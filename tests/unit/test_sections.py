from __future__ import annotations

import pytest

from ai_core.schemas import SectionKind
from ai_core.sections import detect_sections, heading_kind


@pytest.mark.parametrize(
    ("heading", "kind"),
    [
        ("PROFESSIONAL SUMMARY", SectionKind.SUMMARY),
        ("Mục tiêu nghề nghiệp:", SectionKind.SUMMARY),
        ("Skills & Technologies", SectionKind.SKILLS),
        ("KỸ NĂNG CHUYÊN MÔN", SectionKind.SKILLS),
        ("Employment History", SectionKind.EXPERIENCE),
        ("02. Working Experience", SectionKind.EXPERIENCE),
        ("KINH NGHIỆM LÀM VIỆC", SectionKind.EXPERIENCE),
        ("Academic Background", SectionKind.EDUCATION),
        ("HỌC VẤN", SectionKind.EDUCATION),
        ("Selected Projects", SectionKind.PROJECTS),
        ("DỰ ÁN TIÊU BIỂU", SectionKind.PROJECTS),
        ("Licenses & Certifications", SectionKind.CERTIFICATIONS),
        ("CHỨNG CHỈ", SectionKind.CERTIFICATIONS),
        ("Language Skills", SectionKind.LANGUAGES),
        ("NGOẠI NGỮ", SectionKind.LANGUAGES),
    ],
)
def test_heading_variants(heading: str, kind: SectionKind) -> None:
    assert heading_kind(heading) == kind


def test_detects_header_and_section_ranges() -> None:
    sections = detect_sections("NGUYỄN VĂN AN\nEngineer\n\nKỸ NĂNG\nPython\n\nHỌC VẤN\nUniversity")
    assert [section.kind for section in sections] == [
        SectionKind.HEADER,
        SectionKind.SKILLS,
        SectionKind.EDUCATION,
    ]
    assert sections[0].text == "NGUYỄN VĂN AN\nEngineer"
    assert sections[1].text == "Python"


def test_falls_back_to_other_section_without_standard_heading() -> None:
    text = "NGUYỄN VĂN AN\nPython developer with five years of experience."
    sections = detect_sections(text)
    assert len(sections) == 1
    assert sections[0].kind == SectionKind.OTHER
    assert sections[0].confidence == 0.35
