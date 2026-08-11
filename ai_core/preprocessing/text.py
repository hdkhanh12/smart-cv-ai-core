"""Deterministic text normalization with conservative layout cleanup."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from ai_core.errors import Issue, WarningCode
from ai_core.schemas import PageContent

_BULLETS = str.maketrans(
    {
        "•": "-",
        "●": "-",
        "▪": "-",
        "◦": "-",
        "‣": "-",
        "∙": "-",
        "\uf0b7": "-",
    }
)
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_EMAIL = re.compile(
    r"(?P<local>[A-Z0-9._%+-]+)\s*@\s*(?P<domain>[A-Z0-9-]+(?:\s*\.\s*[A-Z0-9-]+)+)",
    re.IGNORECASE,
)
_URL_SCHEME = re.compile(r"\b(?P<scheme>https?)\s*:\s*/\s*/\s*", re.IGNORECASE)
_WWW = re.compile(r"\bwww\s*\.\s*", re.IGNORECASE)
_DOMAIN_SPACE = re.compile(r"(?<=[A-Z0-9])\s*\.\s*(?=[A-Z]{2,}(?:\b|/))", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    raw_text: str
    normalized_text: str
    normalized_pages: list[PageContent]
    warnings: list[Issue]


def _join_email(match: re.Match[str]) -> str:
    domain = re.sub(r"\s+", "", match.group("domain"))
    return f"{match.group('local')}@{domain}"


def normalize_text(text: str) -> str:
    """Normalize Unicode and layout noise without removing Vietnamese characters."""

    value = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    value = value.translate(_BULLETS)
    value = _EMAIL.sub(_join_email, value)
    value = _URL_SCHEME.sub(lambda match: f"{match.group('scheme').lower()}://", value)
    value = _WWW.sub("www.", value)
    value = _DOMAIN_SPACE.sub(".", value)
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in value.splitlines()]
    value = "\n".join(lines)
    return _EXCESS_BLANK_LINES.sub("\n\n", value).strip()


def _margin_line(page: PageContent, *, first: bool) -> str | None:
    lines = [line.strip() for line in page.text.splitlines() if line.strip()]
    if not lines:
        return None
    return normalize_text(lines[0] if first else lines[-1]).casefold()


def _repeated_margins(pages: list[PageContent]) -> set[str]:
    if len(pages) < 2:
        return set()
    threshold = max(2, (len(pages) * 3 + 4) // 5)
    candidates = [
        line
        for page in pages
        for line in (_margin_line(page, first=True), _margin_line(page, first=False))
        if line
    ]
    return {
        line
        for line, count in Counter(candidates).items()
        if count >= threshold and len(line) <= 160
    }


def preprocess_pages(pages: list[PageContent]) -> PreprocessingResult:
    """Normalize pages and suppress repeated header/footer lines from normalized text."""

    raw_text = "\n\n".join(page.text.rstrip() for page in pages).strip()
    repeated = _repeated_margins(pages)
    normalized_pages: list[PageContent] = []
    for page in pages:
        kept_lines = [
            line
            for line in page.text.splitlines()
            if normalize_text(line).casefold() not in repeated
        ]
        normalized_pages.append(
            PageContent(page_number=page.page_number, text=normalize_text("\n".join(kept_lines)))
        )
    normalized_text = "\n\n".join(page.text for page in normalized_pages if page.text).strip()
    warnings = []
    if repeated:
        warnings.append(
            Issue(
                code=WarningCode.REPEATED_MARGIN_REMOVED,
                message="Repeated page header/footer lines were removed from normalized text.",
                stage="preprocessing",
                details={"lineCount": len(repeated)},
            )
        )
    return PreprocessingResult(
        raw_text=raw_text,
        normalized_text=normalized_text,
        normalized_pages=normalized_pages,
        warnings=warnings,
    )
