"""Unit tests for JD and Search Query processing pipeline."""

from __future__ import annotations

import asyncio
import json
import pytest

from ai_core.embeddings.bge_m3 import BgeM3Embedder
from ai_core.schemas.query import (
    SearchQueryAnalysisResult,
    SearchQueryFilters,
    SearchQueryRequest,
)
from ai_core.search import (
    _prune_jd_noise,
    analyze_query_or_jd,
    build_query_text,
)


class MockLLMClient:
    """Mock LLM client returning deterministic JSON for testing."""

    def __init__(self, response_dict: dict | None = None, raise_error: bool = False) -> None:
        self.response_dict = response_dict or {
            "min_experience_years": 3.0,
            "max_experience_years": 5.0,
            "companies": ["VNG", "FPT"],
            "languages": ["Tiếng Anh"],
            "highest_education": "Bachelor",
            "work_type": "Remote",
            "location": "Hà Nội",
        }
        self.raise_error = raise_error

    def generate(self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.0) -> str:
        if self.raise_error:
            raise RuntimeError("Simulated LLM network error")
        return json.dumps(self.response_dict, ensure_ascii=False)


class TestJdQuerySearchPipeline:
    """Test suite for JD/Query text normalization, filter extraction, and embedding."""

    def test_prune_jd_noise_removes_pii_and_hr_boilerplate(self) -> None:
        raw_text = (
            "Tuyển Backend Developer.\n"
            "Email liên hệ: hr@example.com, SĐT: 0901234567\n"
            "Mức lương thỏa thuận 30-40 triệu, chế độ bảo hiểm PVI tốt."
        )
        cleaned = _prune_jd_noise(raw_text)

        assert "hr@example.com" not in cleaned
        assert "0901234567" not in cleaned
        assert "Mức lương thỏa thuận" not in cleaned
        assert "Tuyển Backend Developer." in cleaned

    def test_build_query_text_formats_canonical_sections(self) -> None:
        filters = SearchQueryFilters(
            min_experience_years=3.0,
            skills=["Python", "FastAPI"],
            job_titles=["Backend Developer"],
            location="Hà Nội",
        )
        canonical = build_query_text(filters, "Requirements summary text")

        assert "[JOB_REQUIREMENTS]" in canonical
        assert "Target roles: Backend Developer" in canonical
        assert "Minimum experience years: 3.0" in canonical
        assert "Location: Hà Nội" in canonical
        assert "[REQUIRED_SKILLS]" in canonical
        assert "Python; FastAPI" in canonical
        assert "[REQUIREMENTS_SUMMARY]" in canonical

    def test_analyze_query_or_jd_returns_1024d_normalized_vector_offline(self) -> None:
        embedder = BgeM3Embedder()
        query = "Backend Developer 3 năm kinh nghiệm Python, FastAPI, Docker"

        result = asyncio.run(analyze_query_or_jd(query, embedder, provider=None))

        assert isinstance(result, SearchQueryAnalysisResult)
        assert result.status == "fallback_taxonomy" or result.status == "succeeded"
        assert result.filters.skills is not None
        assert "Python" in result.filters.skills
        assert "FastAPI" in result.filters.skills
        assert result.filters.job_titles is not None
        assert "Backend Developer" in result.filters.job_titles
        assert result.filters.min_experience_years == 3.0

        # Vector checks
        assert result.embedding.dimension == 1024
        assert result.embedding.normalized is True
        assert len(result.embedding.vector) == 1024
        assert result.embedding.model == "BAAI/bge-m3"

    def test_analyze_query_or_jd_with_mock_llm_provider(self) -> None:
        embedder = BgeM3Embedder()
        mock_provider = MockLLMClient()
        jd_text = (
            "Tuyển dụng Senior Backend Developer Python, FastAPI từ 3 đến 5 năm kinh nghiệm.\n"
            "Ưu tiên ứng viên từng làm việc tại VNG hoặc FPT. Yêu cầu Tiếng Anh giao tiếp tốt.\n"
            "Trình độ học vấn: Tốt nghiệp Đại học trở lên. Hình thức làm việc: Remote tại Hà Nội."
        )

        result = asyncio.run(analyze_query_or_jd(jd_text, embedder, provider=mock_provider))

        assert result.status == "succeeded"
        assert result.filters.min_experience_years == 3.0
        assert result.filters.max_experience_years == 5.0
        assert result.filters.work_type == "Remote"
        assert result.filters.location == "Hà Nội"
        assert result.filters.highest_education == "Bachelor"
        assert result.filters.companies == ["VNG", "FPT"]
        assert result.filters.languages == ["Tiếng Anh"]
        assert "Python" in (result.filters.skills or [])
        assert "FastAPI" in (result.filters.skills or [])
        assert result.embedding.dimension == 1024

    def test_analyze_query_or_jd_with_faulty_llm_provider_fallback(self) -> None:
        embedder = BgeM3Embedder()
        faulty_provider = MockLLMClient(raise_error=True)
        query = "Python Developer 2 năm kinh nghiệm"

        result = asyncio.run(analyze_query_or_jd(query, embedder, provider=faulty_provider))

        assert result.status == "fallback_taxonomy"
        assert result.filters.min_experience_years == 2.0
        assert "Python" in (result.filters.skills or [])
        assert result.embedding.dimension == 1024
        assert len(result.embedding.vector) == 1024

    def test_search_query_request_alias(self) -> None:
        req = SearchQueryRequest.model_validate({"Text": "Data Scientist PyTorch"})
        assert req.text == "Data Scientist PyTorch"
