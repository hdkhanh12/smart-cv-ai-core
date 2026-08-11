"""Deterministic CV profile extraction without external services."""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai_core.errors import Issue, WarningCode
from ai_core.schemas import (
    CVProfile,
    CVSection,
    Education,
    Experience,
    SectionKind,
    Skill,
)
from ai_core.sections import detect_sections, heading_kind

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ().-]{7,}\d)(?!\d)")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>{}\[\]]+", re.IGNORECASE)
_DOB = re.compile(
    r"(?:date\s+of\s+birth|dob|ngày\s+sinh)\s*[:\-]?\s*"
    r"(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-](?P<year>\d{4})",
    re.IGNORECASE,
)
_ADDRESS = re.compile(r"^(?:address|địa chỉ|location)\s*[:\-]\s*(?P<value>.+)$", re.IGNORECASE)
_DATE_RANGE = re.compile(
    r"(?P<start>(?:\d{1,2}[./-])?\d{4})\s*"
    r"(?:-|–|—|to|đến)\s*"
    r"(?P<end>present|current|now|hiện tại|(?:\d{1,2}[./-])?\d{4})",
    re.IGNORECASE,
)
_INSTITUTION = re.compile(
    r"\b(university|college|institute|academy|đại học|cao đẳng|học viện)\b",
    re.IGNORECASE,
)
_DEGREE = re.compile(
    r"\b(bachelor|master|engineer|bsc|msc|phd|cử nhân|kỹ sư|thạc sĩ|tiến sĩ)\b",
    re.IGNORECASE,
)
_JOB_TITLE = re.compile(
    r"\b(engineer|developer|scientist|analyst|manager|intern|architect|consultant|"
    r"specialist|lead|trưởng|kỹ sư|lập trình|phân tích)\b",
    re.IGNORECASE,
)
_COMPANY = re.compile(
    r"\b(company|corporation|corp|inc|ltd|llc|jsc|co\.|công ty|tập đoàn)\b",
    re.IGNORECASE,
)
_PRESENT = {"present", "current", "now", "hiện tại"}
_LEVELS = ("beginner", "intermediate", "advanced", "expert", "junior", "senior")


def _section(sections: list[CVSection], kind: SectionKind) -> str:
    return "\n".join(section.text for section in sections if section.kind == kind)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip(" \t-•,;|")
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _candidate_name(lines: list[str]) -> tuple[str | None, float]:
    for index, line in enumerate(lines[:12]):
        value = line.strip(" \t|•-")
        tokens = value.split()
        if (
            2 <= len(tokens) <= 7
            and 3 <= len(value) <= 70
            and "@" not in value
            and not _PHONE.search(value)
            and not _URL.search(value)
            and heading_kind(value) is None
            and not _COMPANY.search(value)
            and all(any(character.isalpha() for character in token) for token in tokens)
            and not any(character.isdigit() for character in value)
        ):
            uppercase_letters = [character for character in value if character.isalpha()]
            is_upper = uppercase_letters and all(
                character.isupper() for character in uppercase_letters
            )
            confidence = 0.95 if index <= 2 and is_upper else 0.8 if index <= 4 else 0.65
            return value, confidence
    return None, 0.0


def _phone(text: str) -> str | None:
    for match in _PHONE.finditer(text):
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        if 9 <= len(digits) <= 15 and candidate.count("/") < 2:
            return ("+" if candidate.lstrip().startswith("+") else "") + digits
    return None


def _date_value(value: str) -> date:
    parts = re.split(r"[./-]", value)
    if len(parts) == 1:
        return date(int(parts[0]), 1, 1)
    return date(int(parts[1]), int(parts[0]), 1)


def _date_range(match: re.Match[str]) -> tuple[date, date | None, bool]:
    start = _date_value(match.group("start"))
    end_text = match.group("end").casefold()
    is_current = end_text in _PRESENT
    return start, None if is_current else _date_value(end_text), is_current


@lru_cache(maxsize=1)
def _skill_config() -> dict[str, list[str]]:
    path = Path(__file__).resolve().parents[2] / "configs" / "skills.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    skills = payload["skills"]
    if not isinstance(skills, dict):
        raise ValueError("skills config must contain an object")
    return {str(name): [str(alias) for alias in aliases] for name, aliases in skills.items()}


def canonical_skill_name(name: str) -> str:
    """Map an LLM-extracted skill to the existing versioned taxonomy."""

    normalized = name.strip().casefold()
    for canonical, aliases in _skill_config().items():
        if normalized == canonical.casefold() or any(
            normalized == alias.casefold() for alias in aliases
        ):
            return canonical
    return name.strip()


def canonical_skill_mentions(text: str) -> list[tuple[str, str]]:
    """Return versioned-taxonomy skills explicitly present in one text fragment."""

    matches: list[tuple[str, str]] = []
    for canonical, aliases in _skill_config().items():
        for alias in sorted({canonical, *aliases}, key=len, reverse=True):
            if canonical in {"C", "R", "Go"} and alias == canonical:
                continue
            match = re.search(rf"(?<![\w+#]){re.escape(alias)}(?![\w+#])", text, re.IGNORECASE)
            if match:
                matches.append((canonical, match.group(0)))
                break
    return matches


def _skills(text: str, sections: list[CVSection]) -> list[Skill]:
    skill_text = _section(sections, SectionKind.SKILLS)
    sources: list[tuple[str, SectionKind | None]] = (
        [(skill_text, SectionKind.SKILLS)] if skill_text else []
    )
    sources.append((text, None))
    found: dict[str, Skill] = {}
    for canonical, aliases in _skill_config().items():
        for source, section_kind in sources:
            if canonical in found:
                break
            if canonical in {"C", "R", "Go"} and section_kind != SectionKind.SKILLS:
                continue
            for alias in sorted(aliases, key=len, reverse=True):
                pattern = re.compile(rf"(?<![\w+#.]){re.escape(alias)}(?![\w+#.])", re.IGNORECASE)
                match = pattern.search(source)
                if not match:
                    continue
                line = source[match.start() :].splitlines()[0][:180]
                vicinity = source[max(0, match.start() - 30) : match.end() + 30]
                years_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|năm)",
                    vicinity,
                    re.IGNORECASE,
                )
                level = next(
                    (level for level in _LEVELS if re.search(rf"\b{level}\b", vicinity, re.I)),
                    None,
                )
                found[canonical] = Skill(
                    name=match.group(0),
                    canonical_name=canonical,
                    years=float(years_match.group(1)) if years_match else None,
                    level=level,
                    evidence=[line.strip()],
                    section=section_kind,
                    confidence=0.95 if section_kind == SectionKind.SKILLS else 0.75,
                )
                break
    return sorted(found.values(), key=lambda skill: skill.canonical_name.casefold())


def _experience(sections: list[CVSection]) -> tuple[list[Experience], list[Issue]]:
    text = _section(sections, SectionKind.EXPERIENCE)
    if not text:
        return [], []
    lines = [line.strip(" \t•-") for line in text.splitlines() if line.strip()]
    entries: list[Experience] = []
    warnings: list[Issue] = []
    date_indexes = [index for index, line in enumerate(lines) if _DATE_RANGE.search(line)]
    for position, index in enumerate(date_indexes):
        match = _DATE_RANGE.search(lines[index])
        if match is None:
            continue
        descriptor = lines[index - 1] if index else "Unknown role"
        job_title = descriptor
        company: str | None = None
        parts = re.split(r"\s*[|@]\s*|\s+at\s+|\s+tại\s+", descriptor, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            job_title, company = parts
        elif index >= 2 and (_JOB_TITLE.search(lines[index - 2]) or _COMPANY.search(descriptor)):
            job_title, company = lines[index - 2], descriptor
        elif _COMPANY.search(descriptor):
            company = descriptor
            job_title = "Intern / Developer"
        if company is None and _COMPANY.search(lines[index]):
            clean_line = _DATE_RANGE.sub("", lines[index]).strip(" \t|•-")
            clean_line = re.sub(r"^(?:email|phone|address|contact|github|linkedin)\s*", "", clean_line, flags=re.I).strip(" \t|•-")
            if clean_line:
                company = clean_line
                if job_title == descriptor or not job_title or job_title == "Unknown role":
                    job_title = "Intern / Developer"
        next_index = date_indexes[position + 1] if position + 1 < len(date_indexes) else len(lines)
        description_end = next_index
        if next_index < len(lines):
            description_end -= 1
            if description_end >= 1 and _JOB_TITLE.search(lines[description_end - 1]):
                description_end -= 1
        description = _unique(lines[index + 1 : description_end])
        try:
            start, end, current = _date_range(match)
            if end and end < start:
                warnings.append(
                    Issue(
                        code=WarningCode.INVALID_TIMELINE,
                        message="Experience date range ends before it starts.",
                        stage="profile_validation",
                        details={"entryIndex": position},
                    )
                )
                continue
            entries.append(
                Experience(
                    job_title=job_title,
                    company=company,
                    start_date=start,
                    end_date=end,
                    is_current=current,
                    description=description,
                    achievements=[
                        line
                        for line in description
                        if re.search(r"\b\d+(?:[.,]\d+)?%|\b\d+\+?\b", line)
                    ],
                    confidence=0.85 if company else 0.7,
                )
            )
        except ValueError:
            warnings.append(
                Issue(
                    code=WarningCode.INVALID_TIMELINE,
                    message="Experience contains an invalid calendar date.",
                    stage="profile_validation",
                    details={"entryIndex": position},
                )
            )
    return entries, warnings


def _total_experience(entries: list[Experience]) -> float | None:
    intervals = sorted(
        (entry.start_date, entry.end_date or date.today())
        for entry in entries
        if entry.start_date is not None
    )
    if not intervals:
        return None
    merged: list[tuple[date, date]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    days = sum((end - start).days for start, end in merged)
    return round(days / 365.25, 1)


def _education(sections: list[CVSection]) -> list[Education]:
    text = _section(sections, SectionKind.EDUCATION)
    lines = [line.strip(" \t•-") for line in text.splitlines() if line.strip()]
    entries: list[Education] = []
    for index, line in enumerate(lines):
        if not _INSTITUTION.search(line):
            continue
        nearby = lines[max(0, index - 1) : min(len(lines), index + 3)]
        degree = next((value for value in nearby if _DEGREE.search(value)), None)
        field_of_study = None
        if degree:
            field_match = re.search(
                r"(?:\bof\b|\bin\b|chuyên\s+ngành)\s*[:\-]?\s*(.+)$",
                degree,
                re.IGNORECASE,
            )
            field_of_study = field_match.group(1).strip() if field_match else None
        date_match = next(
            (_DATE_RANGE.search(value) for value in nearby if _DATE_RANGE.search(value)), None
        )
        start = end = None
        if date_match:
            try:
                start, end, _ = _date_range(date_match)
            except ValueError:
                start = end = None
        entries.append(
            Education(
                institution=line,
                degree=degree if degree != line else None,
                field_of_study=field_of_study,
                start_date=start,
                end_date=end,
                confidence=0.85,
            )
        )
    return entries


def _section_items(sections: list[CVSection], kind: SectionKind) -> list[str]:
    text = _section(sections, kind)
    values = re.split(r"[,;|\n]+", text)
    return _unique(values)


def _dedupe_experiences(entries: list[Experience]) -> list[Experience]:
    seen: set[tuple[object, ...]] = set()
    result: list[Experience] = []
    for entry in entries:
        key = (
            entry.job_title.casefold(),
            (entry.company or "").casefold(),
            entry.start_date,
            entry.end_date,
        )
        if key not in seen:
            seen.add(key)
            result.append(entry)
    return result


def _dedupe_education(entries: list[Education]) -> list[Education]:
    seen: set[tuple[str, str]] = set()
    result: list[Education] = []
    for entry in entries:
        key = (entry.institution.casefold(), (entry.degree or "").casefold())
        if key not in seen:
            seen.add(key)
            result.append(entry)
    return result


def extract_cv_profile(text: str) -> CVProfile:
    """Extract a validated profile from normalized CV text."""

    sections = detect_sections(text)
    lines = [line for line in text.splitlines() if line.strip()]
    name, name_confidence = _candidate_name(lines)
    email_match = _EMAIL.search(text)
    email = email_match.group(0).lower() if email_match else None
    phone = _phone(text)
    urls = _unique([match.group(0).rstrip(".,);") for match in _URL.finditer(text)])
    urls = [f"https://{url}" if url.lower().startswith("www.") else url for url in urls]
    dob_match = _DOB.search(text)
    date_of_birth = None
    if dob_match:
        try:
            date_of_birth = date(
                int(dob_match.group("year")),
                int(dob_match.group("month")),
                int(dob_match.group("day")),
            )
        except ValueError:
            date_of_birth = None
    address = next(
        (
            match.group("value").strip()
            for line in lines[:20]
            if (match := _ADDRESS.match(line.strip()))
        ),
        None,
    )
    experiences, warnings = _experience(sections)
    experiences = _dedupe_experiences(experiences)
    education = _dedupe_education(_education(sections))
    if name is None:
        warnings.append(
            Issue(
                code=WarningCode.LOW_CONFIDENCE_NAME,
                message="A reliable candidate name could not be identified.",
                stage="profile_validation",
            )
        )
    if email is None and phone is None:
        warnings.append(
            Issue(
                code=WarningCode.MISSING_CONTACT,
                message="No valid email or phone number was found.",
                stage="profile_validation",
            )
        )
    summary = _section(sections, SectionKind.SUMMARY).strip() or None
    header_lines = next(
        (section.text.splitlines() for section in sections if section.kind == SectionKind.HEADER),
        lines[:5],
    )
    headline = next(
        (line.strip() for line in header_lines if line.strip() != name and _JOB_TITLE.search(line)),
        None,
    )
    confidence = {
        "candidateName": name_confidence,
        "email": 0.98 if email else 0.0,
        "phone": 0.9 if phone else 0.0,
        "skills": 0.9 if _section(sections, SectionKind.SKILLS) else 0.7,
        "experience": 0.85 if experiences else 0.0,
        "education": 0.85 if _section(sections, SectionKind.EDUCATION) else 0.0,
    }
    return CVProfile(
        candidate_name=name,
        email=email,
        phone=phone,
        urls=urls,
        address=address,
        date_of_birth=date_of_birth,
        headline=headline,
        summary=summary,
        skills=_skills(text, sections),
        experiences=experiences,
        education=education,
        languages=_section_items(sections, SectionKind.LANGUAGES),
        certifications=_section_items(sections, SectionKind.CERTIFICATIONS),
        sections=sections,
        total_experience_years=_total_experience(experiences),
        field_confidence=confidence,
        warnings=warnings,
    )
