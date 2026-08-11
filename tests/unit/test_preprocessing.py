from __future__ import annotations

import unicodedata

from ai_core.errors import WarningCode
from ai_core.preprocessing import normalize_text, preprocess_pages
from ai_core.schemas import PageContent


def test_normalizes_unicode_spacing_bullets_and_blank_lines() -> None:
    decomposed = unicodedata.normalize("NFD", "Kỹ năng tiếng Việt")
    source = f"  {decomposed}  \n•  Python\t FastAPI\n\n\n\nKinh nghiệm "

    result = normalize_text(source)

    assert result == "Kỹ năng tiếng Việt\n- Python FastAPI\n\nKinh nghiệm"
    assert unicodedata.is_normalized("NFC", result)


def test_repairs_split_email_and_url() -> None:
    source = "Email: candidate @ example . com\nWeb: https : / / example . vn/profile"
    result = normalize_text(source)
    assert "candidate@example.com" in result
    assert "https://example.vn/profile" in result


def test_removes_repeated_header_footer_only_from_normalized_pages() -> None:
    pages = [
        PageContent(
            page_number=index,
            text=f"CONFIDENTIAL CV\nPage-specific content {index}\nCandidate footer",
        )
        for index in range(1, 4)
    ]

    result = preprocess_pages(pages)

    assert "CONFIDENTIAL CV" in result.raw_text
    assert "Candidate footer" in result.raw_text
    assert "CONFIDENTIAL CV" not in result.normalized_text
    assert "Candidate footer" not in result.normalized_text
    assert all(f"Page-specific content {index}" in result.normalized_text for index in range(1, 4))
    assert result.warnings[0].code == WarningCode.REPEATED_MARGIN_REMOVED
