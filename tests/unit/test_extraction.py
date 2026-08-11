from __future__ import annotations

import json
from pathlib import Path

from ai_core.errors import WarningCode
from ai_core.extraction import extract_cv_profile

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _golden_projection(profile: object) -> dict[str, object]:
    payload = profile.model_dump(mode="json", by_alias=True)  # type: ignore[attr-defined]
    return {
        "candidateName": payload["candidateName"],
        "email": payload["email"],
        "phone": payload["phone"],
        "urls": payload["urls"],
        "address": payload["address"],
        "dateOfBirth": payload["dateOfBirth"],
        "headline": payload["headline"],
        "summary": payload["summary"],
        "skills": [skill["canonicalName"] for skill in payload["skills"]],
        "experiences": [
            {key: entry[key] for key in ("jobTitle", "company", "startDate", "endDate")}
            for entry in payload["experiences"]
        ],
        "education": [
            {key: entry[key] for key in ("institution", "degree", "startDate", "endDate")}
            for entry in payload["education"]
        ],
        "languages": payload["languages"],
        "certifications": payload["certifications"],
        "totalExperienceYears": payload["totalExperienceYears"],
    }


def test_golden_profile_matches_reviewed_expected_json() -> None:
    text = (FIXTURES / "golden_cv.txt").read_text(encoding="utf-8")
    expected = json.loads((FIXTURES / "golden_profile.json").read_text(encoding="utf-8"))
    profile = extract_cv_profile(text)
    assert _golden_projection(profile) == expected


def test_short_skill_names_only_match_in_explicit_skills_section() -> None:
    prose = extract_cv_profile("Jane Example\nEngineer\nI go to work and use a plan for research.")
    explicit = extract_cv_profile(
        "Jane Example\nEngineer\nSKILLS\nC, R, Go\nEXPERIENCE\nEngineer | Example\n2020 - 2022"
    )
    assert not {"C", "R", "Go"} & {skill.canonical_name for skill in prose.skills}
    assert {"C", "R", "Go"} <= {skill.canonical_name for skill in explicit.skills}


def test_invalid_experience_timeline_is_warned_and_excluded() -> None:
    profile = extract_cv_profile("JANE EXAMPLE\nEXPERIENCE\nEngineer | Example Inc\n2024 - 2020")
    assert profile.experiences == []
    assert WarningCode.INVALID_TIMELINE in {warning.code for warning in profile.warnings}


def test_missing_contact_and_unreliable_name_are_warned() -> None:
    profile = extract_cv_profile("SKILLS\nPython")
    codes = {warning.code for warning in profile.warnings}
    assert WarningCode.MISSING_CONTACT in codes
    assert WarningCode.LOW_CONFIDENCE_NAME in codes


def test_extracts_education_field_and_deduplicates_entries() -> None:
    profile = extract_cv_profile(
        "JANE EXAMPLE\nEDUCATION\nExample University\n"
        "Bachelor of Computer Science\n2018 - 2022\n"
        "Example University\nBachelor of Computer Science\n2018 - 2022"
    )
    assert len(profile.education) == 1
    assert profile.education[0].field_of_study == "Computer Science"
