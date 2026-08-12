"""Unit tests for JD and Search Query processing pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

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

    def test_analyze_query_or_jd_returns_1024d_normalized_vector(self) -> None:
        import asyncio

        embedder = BgeM3Embedder()
        query = "Backend Developer 3 năm kinh nghiệm Python, FastAPI, Docker"

        result = asyncio.run(analyze_query_or_jd(query, embedder))

        assert isinstance(result, SearchQueryAnalysisResult)
        assert result.status == "succeeded"
        assert "Python" in result.filters.skills
        assert "FastAPI" in result.filters.skills
        assert "Backend Developer" in result.filters.job_titles
        assert result.filters.min_experience_years == 3.0

        # Vector checks
        assert result.embedding.dimension == 1024
        assert result.embedding.normalized is True
        assert len(result.embedding.vector) == 1024
        assert result.embedding.model == "BAAI/bge-m3"


    def test_search_query_request_alias(self) -> None:
        req = SearchQueryRequest.model_validate({"Text": "Data Scientist PyTorch"})
        assert req.text == "Data Scientist PyTorch"
