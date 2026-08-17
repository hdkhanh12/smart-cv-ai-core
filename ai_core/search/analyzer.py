"""Real-time JD & Search Query analysis and 1024d BGE-M3 vector embedding module."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Literal

import requests
from fastapi.concurrency import run_in_threadpool

from ai_core.embeddings.bge_m3 import BgeM3Embedder, _sanitize_pii_text
from ai_core.schemas.query import (
    SearchQueryAnalysisResult,
    SearchQueryEmbeddingInfo,
    SearchQueryFilters,
)
from ai_core.taxonomy import JOB_TITLES, SKILLS_DICT, find_entities

logger = logging.getLogger(__name__)

# Pattern matching HR contact, salary, benefits boilerplate lines to prune from JD text before embedding
_JD_NOISE_PATTERN = re.compile(
    r"\b(mức lương|lương thỏa thuận|chế độ bảo hiểm|bảo hiểm pvi|phúc lợi|liên hệ hr|"
    r"gửi cv về|recruiter|hr contact|perks|benefits|competitive salary|email hr)\b",
    re.IGNORECASE | re.UNICODE,
)

_LLM_QUERY_SYSTEM_PROMPT = """Bạn là hệ thống trích xuất filter tuyển dụng từ văn bản đầu vào.
Đầu vào có thể là câu query ngắn (VD: "Backend Developer 3 năm, biết Python và .NET, remote tại Hà Nội") HOẶC toàn bộ Job Description (JD) dài.
Trong cả 2 trường hợp, hãy trích xuất các field sau thành 1 JSON object:

- min_experience_years (float|null): số năm kinh nghiệm tối thiểu yêu cầu (VD: 3, 1.5, 2)
- max_experience_years (float|null): số năm kinh nghiệm tối đa yêu cầu (null nếu không nêu hoặc không có upper bound)
- companies (list[string]|null): tên công ty cụ thể được nêu (VD: công ty tuyển dụng hoặc công ty từng làm)
- languages (list[string]|null): ngoại ngữ yêu cầu (VD: "Tiếng Anh", "Tiếng Nhật") - KHÔNG phải ngôn ngữ lập trình
- highest_education (string|null): trình độ học vấn yêu cầu ("High School" | "Associate" | "Bachelor" | "Master" | "PhD" | "Đại học" | "Cao đẳng" | "Thạc sĩ" | "Tiến sĩ")
- work_type (string|null): hình thức làm việc ("Remote" | "Onsite" | "Hybrid")
- location (string|null): địa điểm làm việc / tỉnh thành nếu có nêu (VD: "Hà Nội", "Hồ Chí Minh")

Nếu là JD dài, đọc toàn bộ để tổng hợp.
Chỉ trả DUY NHẤT 1 JSON hợp lệ, không giải thích, không dùng markdown fence. Nếu field không xuất hiện, trả null.
"""


def _prune_jd_noise(text: str) -> str:
    """Sanitize PII and prune boilerplate recruitment/benefit lines from JD text."""
    if not text:
        return ""
    filtered_lines: list[str] = []
    for line in text.splitlines():
        line_str = line.strip()
        if not line_str or _JD_NOISE_PATTERN.search(line_str):
            continue
        filtered_lines.append(line_str)

    joined_text = "\n".join(filtered_lines)
    return _sanitize_pii_text(joined_text)


def build_query_text(filters: SearchQueryFilters, clean_text: str) -> str:
    """Build a canonical non-PII query representation for BGE-M3 (1024d).

    This representation matches build_profile_text() section structure in bge_m3.py
    to ensure 100% Vector Space Alignment for pgvector Cosine Similarity.
    """
    lines = ["[JOB_REQUIREMENTS]"]
    if filters.job_titles:
        lines.append(f"Target roles: {'; '.join(filters.job_titles)}")
    if filters.min_experience_years is not None:
        lines.append(f"Minimum experience years: {filters.min_experience_years}")
    if filters.location:
        lines.append(f"Location: {filters.location}")
    if filters.work_type:
        lines.append(f"Work type: {filters.work_type}")

    if filters.skills:
        lines.extend(["", "[REQUIRED_SKILLS]", "; ".join(filters.skills)])

    if clean_text:
        lines.extend(["", "[REQUIREMENTS_SUMMARY]", clean_text])

    return "\n".join(lines)


def _extract_first_json_object(raw: str) -> str | None:
    """Extract substring of the first valid JSON object in a raw response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def _extract_exp_from_text(text: str) -> float | None:
    """Heuristic extraction of experience years if LLM is not called or unavailable."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:năm|nam|years?)\s*(?:kinh\s*nghiệm|exp)?", text, re.I)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_llm_query_fields_sync(text: str, provider: Any | None = None) -> tuple[dict[str, Any], bool]:
    """Execute LLM call (temperature=0, max_tokens=400) to extract structured query fields.

    Returns:
        (extracted_dict, is_success)
    """
    if not text.strip():
        return {}, False

    # Extract connection settings from provider or environment
    session: requests.Session | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: int = 15

    if provider is not None and hasattr(provider, "_settings"):
        base_url = getattr(provider._settings, "base_url", None)
        api_key = getattr(provider._settings, "api_key", None)
        model = getattr(provider._settings, "model", None)
        session = getattr(provider, "_session", None)
    elif provider is not None and hasattr(provider, "generate"):
        # Custom LLM client interface support
        try:
            raw = provider.generate(
                messages=[
                    {"role": "system", "content": _LLM_QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=400,
                temperature=0.0,
            )
            json_str = _extract_first_json_object(str(raw))
            if json_str:
                return json.loads(json_str), True
            return {}, False
        except Exception as exc:
            logger.warning("Custom LLM client extraction failed: %s", exc)
            return {}, False
    else:
        try:
            from ai_core.extraction.beeknoee import load_beeknoee_settings
            settings = load_beeknoee_settings()
            base_url = settings.base_url
            api_key = settings.api_key
            model = settings.model
        except Exception:
            # Beeknoee settings not configured (e.g. offline/mock environment)
            return {}, False

    if not base_url or not api_key:
        return {}, False

    http_session = session or requests.Session()
    payload: dict[str, Any] = {
        "model": model or "gemini-2.5-flash-lite",
        "messages": [
            {"role": "system", "content": _LLM_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }

    try:
        response = http_session.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            logger.warning("LLM query extraction returned HTTP %s: %s", response.status_code, response.text[:200])
            return {}, False

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content, True

        json_str = _extract_first_json_object(str(content))
        if json_str:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed, True
        return {}, False
    except Exception as exc:
        logger.warning("LLM query extraction request failed, falling back to heuristics: %s", exc)
        return {}, False


async def _extract_llm_query_fields(text: str, provider: Any | None = None) -> tuple[dict[str, Any], bool]:
    """Asynchronous wrapper for LLM query field extraction."""
    return await run_in_threadpool(_extract_llm_query_fields_sync, text, provider)


async def analyze_query_or_jd(
    text: str,
    embedder: BgeM3Embedder,
    provider: Any | None = None,
) -> SearchQueryAnalysisResult:
    """Hybrid pipeline: Fast FlashText taxonomy matching + Parallel LLM extraction + BGE-M3 1024d embedding."""
    start_time = time.perf_counter()
    clean_text = _prune_jd_noise(text)

    # 1. Deterministic Taxonomy Extraction (FlashText)
    skills = list(find_entities(clean_text, SKILLS_DICT).keys())
    job_titles = list(find_entities(clean_text, JOB_TITLES).keys())

    # Build initial representation string for BGE-M3
    initial_filters = SearchQueryFilters(
        skills=skills if skills else None,
        job_titles=job_titles if job_titles else None,
    )
    canonical_text = build_query_text(initial_filters, clean_text)

    # 2. Parallel Processing: LLM Field Extraction Task + BGE-M3 Vector Embedding Task
    llm_task = _extract_llm_query_fields(clean_text, provider)
    embedding_task = run_in_threadpool(embedder.embed_text, canonical_text)

    (llm_data, is_llm_success), embed_result = await asyncio.gather(llm_task, embedding_task)

    # 3. Merge Filter Results
    if is_llm_success and llm_data:
        status_flag: Literal["succeeded", "fallback_taxonomy"] = "succeeded"
        min_exp_raw = llm_data.get("min_experience_years")
        try:
            min_exp = float(min_exp_raw) if min_exp_raw is not None else _extract_exp_from_text(clean_text)
        except (ValueError, TypeError):
            min_exp = _extract_exp_from_text(clean_text)

        max_exp_raw = llm_data.get("max_experience_years")
        try:
            max_exp = float(max_exp_raw) if max_exp_raw is not None else None
        except (ValueError, TypeError):
            max_exp = None

        companies = llm_data.get("companies")
        languages = llm_data.get("languages")
        highest_education = llm_data.get("highest_education")
        work_type = llm_data.get("work_type")
        location = llm_data.get("location")
    else:
        status_flag = "fallback_taxonomy"
        min_exp = _extract_exp_from_text(clean_text)
        max_exp = None
        companies = None
        languages = None
        highest_education = None
        work_type = None
        location = None

    final_filters = SearchQueryFilters(
        min_experience_years=min_exp,
        max_experience_years=max_exp,
        skills=skills if skills else None,
        job_titles=job_titles if job_titles else None,
        companies=companies if isinstance(companies, list) and companies else None,
        languages=languages if isinstance(languages, list) and languages else None,
        highest_education=str(highest_education) if highest_education else None,
        work_type=str(work_type) if work_type else None,
        location=str(location) if location else None,
    )

    total_latency_ms = (time.perf_counter() - start_time) * 1000

    return SearchQueryAnalysisResult(
        status=status_flag,
        filters=final_filters,
        embedding=SearchQueryEmbeddingInfo(
            vector=embed_result.vector,
            dimension=embed_result.dimension,
            normalized=embed_result.normalized,
            model=embed_result.model,
            model_revision=embed_result.model_revision,
            template_version="bge-m3-query-v1",
            source_hash=embed_result.source_hash,
            duration_ms=embed_result.duration_ms,
        ),
        latency_ms=round(total_latency_ms, 2),
    )
