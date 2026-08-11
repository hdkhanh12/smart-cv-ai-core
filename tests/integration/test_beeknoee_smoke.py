"""Live Beeknoee smoke test; skipped unless a local `.env` has BEE_API_KEY."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest

from ai_core.errors import CoreError, ErrorCode
from ai_core.extraction.beeknoee import BeeknoeeStructuredExtractionProvider
from ai_core.extraction.provider import build_request
from ai_core.parsers.unified import unified_from_lines
from ai_core.reconciliation import reconcile_profile
from ai_core.validation import validate_input


def _same_normalized_text(left: str | None, right: str) -> bool:
    if left is None:
        return False
    normalized_left = " ".join(unicodedata.normalize("NFC", left).casefold().split())
    normalized_right = " ".join(unicodedata.normalize("NFC", right).casefold().split())
    return normalized_left == normalized_right


def test_beeknoee_structured_extraction_with_synthetic_document(tmp_path: Path) -> None:
    if os.getenv("SMART_CV_RUN_LIVE_SMOKE", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Live Beeknoee smoke is disabled; set SMART_CV_RUN_LIVE_SMOKE=1 to run it.")
    try:
        provider = BeeknoeeStructuredExtractionProvider()
    except CoreError as exc:
        if exc.issue.code == ErrorCode.LLM_CONFIGURATION_ERROR:
            pytest.skip("BEE_API_KEY is not configured; live Beeknoee smoke test skipped.")
        raise

    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    document = unified_from_lines(
        validate_input(source),
        "synthetic-smoke",
        [
            [
                "ALEX MORGAN",
                "SKILLS",
                "Python, SQL",
                "PERSONAL PROJECTS",
                "CV Parser Toolkit - Built a local parser for structured CV data.",
            ]
        ],
    )

    mapped_profile = provider.extract(build_request(document))
    profile = reconcile_profile(
        mapped_profile,
        document,
        provider_name=provider.name,
        allow_identity_fallback=False,
    )

    # Candidate identity is intentionally not a remote extraction target.
    assert profile.candidate_name is None
    assert {skill.canonical_name for skill in profile.skills} >= {"Python", "SQL"}
    # Live providers may classify this short fixture as a project or a skills
    # only profile; the smoke gate verifies transport, schema and grounding.
    assert profile.skills or profile.projects
    assert profile.experiences == []
    assert profile.total_experience_years is None
