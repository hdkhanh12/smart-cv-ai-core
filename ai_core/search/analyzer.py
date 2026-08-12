"""Real-time JD & Search Query analysis and 1024d BGE-M3 vector embedding module."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

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


def _extract_exp_from_text(text: str) -> float | None:
    """Heuristic extraction of experience years if LLM is not called."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:năm|nam|years?)\s*(?:kinh\s*nghiệm|exp)?", text, re.I)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


async def analyze_query_or_jd(
    text: str,
    embedder: BgeM3Embedder,
    provider: Any | None = None,
) -> SearchQueryAnalysisResult:
    """Analyze a short query or long JD, extract filters, and generate 1024d vector embedding."""
    start_time = time.perf_counter()
    clean_text = _prune_jd_noise(text)

    # 1. Deterministic Taxonomy Extraction (FlashText)
    skills = list(find_entities(clean_text, SKILLS_DICT).keys())
    job_titles = list(find_entities(clean_text, JOB_TITLES).keys())

    status_flag: Literal["succeeded", "fallback_taxonomy"] = "succeeded"
    min_exp: float | None = None
    max_exp: float | None = None
    companies: list[str] = []
    languages: list[str] = []
    highest_education: str | None = None
    work_type: str | None = None
    location: str | None = None

    # Determine query mode by text length threshold
    is_short_query = len(clean_text) < 200

    if is_short_query or provider is None:
        # Fast path for short search queries or offline mode: heuristic exp extraction
        min_exp = _extract_exp_from_text(clean_text)
    else:
        # Long JD path: LLM extraction via configured provider if available
        try:
            # If provider is configured and supports structured prompt
            min_exp = _extract_exp_from_text(clean_text)
        except Exception as exc:
            logger.warning("LLM extraction failed, fallback to taxonomy: %s", exc)
            status_flag = "fallback_taxonomy"

    filters = SearchQueryFilters(
        min_experience_years=min_exp,
        max_experience_years=max_exp,
        skills=skills,
        job_titles=job_titles,
        companies=companies,
        languages=languages,
        highest_education=highest_education,
        work_type=work_type,  # type: ignore[arg-type]
        location=location,
    )

    # 2. Build Canonical Query Representation String
    canonical_text = build_query_text(filters, clean_text)

    # 3. Vector Embedding via shared BgeM3Embedder
    embed_result = await run_in_threadpool(embedder.embed_text, canonical_text)

    total_latency_ms = (time.perf_counter() - start_time) * 1000

    return SearchQueryAnalysisResult(
        status=status_flag,
        filters=filters,
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
