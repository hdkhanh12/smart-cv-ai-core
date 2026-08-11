"""Vietnamese/English heading-based CV section detection."""

from __future__ import annotations

import re
import unicodedata

from ai_core.schemas import CVSection, SectionKind

HEADING_ALIASES: dict[SectionKind, tuple[str, ...]] = {
    SectionKind.SUMMARY: (
        "summary",
        "professional summary",
        "profile",
        "career objective",
        "objective",
        "tóm tắt",
        "giới thiệu",
        "mục tiêu nghề nghiệp",
    ),
    SectionKind.SKILLS: (
        "skills",
        "technical skills",
        "technical skill",
        "skills & technologies",
        "core competencies",
        "professional skills",
        "expertise",
        "technologies",
        "tech stack",
        "skill",
        "kỹ năng",
        "kỹ năng chuyên môn",
        "công nghệ",
    ),
    SectionKind.EXPERIENCE: (
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "working experience",
        "work experiences",
        "career history",
        "professional background",
        "employment",
        "kinh nghiệm",
        "kinh nghiệm làm việc",
        "quá trình công tác",
    ),
    SectionKind.EDUCATION: (
        "education",
        "academic background",
        "educational background",
        "education & training",
        "academic qualifications",
        "qualifications",
        "training",
        "học vấn",
        "giáo dục",
        "quá trình học tập",
    ),
    SectionKind.PROJECTS: (
        "projects",
        "personal projects",
        "selected projects",
        "project experience",
        "dự án",
        "dự án tiêu biểu",
    ),
    SectionKind.CERTIFICATIONS: (
        "certifications",
        "certificates",
        "certificate",
        "certification",
        "licenses & certifications",
        "chứng chỉ",
        "chứng nhận",
    ),
    SectionKind.LANGUAGES: (
        "languages",
        "language skills",
        "language",
        "language proficiency",
        "ngoại ngữ",
        "ngôn ngữ",
    ),
}


def _heading_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold().strip()
    normalized = re.sub(r"^[#\-–—•|:]+|[#\-–—•|:]+$", "", normalized).strip()
    normalized = re.sub(r"^\d{1,2}[.)\s-]+", "", normalized).strip()
    return re.sub(r"\s+", " ", normalized)


_ALIAS_TO_KIND = {
    _heading_key(alias): kind for kind, aliases in HEADING_ALIASES.items() for alias in aliases
}


def heading_kind(line: str) -> SectionKind | None:
    key = _heading_key(line)
    if len(key) > 60:
        return None
    direct = _ALIAS_TO_KIND.get(key)
    if direct is not None:
        return direct
    # Handle multi-column interleaved headings like "Work Experience Contact"
    for alias, kind in _ALIAS_TO_KIND.items():
        if len(alias) >= 4 and (key.startswith(alias + " ") or key.endswith(" " + alias)):
            return kind
    return None


def detect_sections(text: str) -> list[CVSection]:
    lines = text.splitlines()
    headings = [
        (index, kind, line.strip())
        for index, line in enumerate(lines)
        if (kind := heading_kind(line)) is not None
    ]
    if not headings:
        content = text.strip()
        return [
            CVSection(
                kind=SectionKind.OTHER,
                text=content,
                start_line=1,
                end_line=max(1, len(lines)),
                confidence=0.35,
            )
        ]

    sections: list[CVSection] = []
    first_index = headings[0][0]
    if first_index > 0 and any(line.strip() for line in lines[:first_index]):
        sections.append(
            CVSection(
                kind=SectionKind.HEADER,
                text="\n".join(lines[:first_index]).strip(),
                start_line=1,
                end_line=first_index,
                confidence=0.9,
            )
        )
    for position, (line_index, kind, heading) in enumerate(headings):
        next_index = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        sections.append(
            CVSection(
                kind=kind,
                heading=heading,
                text="\n".join(lines[line_index + 1 : next_index]).strip(),
                start_line=line_index + 1,
                end_line=max(line_index + 1, next_index),
                confidence=0.95,
            )
        )
    return sections
