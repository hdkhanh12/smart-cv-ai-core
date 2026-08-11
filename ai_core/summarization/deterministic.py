"""Traceable deterministic summaries derived only from CVProfile fields."""

from __future__ import annotations

import re

from ai_core.schemas import CVProfile, SummaryResult

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONTACT = re.compile(r"@|https?://|www\.|\+?\d[\d ().-]{7,}\d", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip() for sentence in _SENTENCE_SPLIT.split(text.strip()) if sentence.strip()
    ]


def _usable_existing_summary(value: str | None) -> bool:
    if not value or not 40 <= len(value) <= 600 or _CONTACT.search(value):
        return False
    return len(_sentences(value)) in {2, 3}


def summarize_profile(profile: CVProfile) -> SummaryResult:
    """Build a two-sentence profile summary with explicit source-field traceability."""

    if _usable_existing_summary(profile.summary):
        existing = profile.summary or ""
        return SummaryResult(
            text=existing,
            sentence_count=len(_sentences(existing)),
            source_fields=["profile.summary"],
            reused_existing=True,
        )

    subject = profile.candidate_name or "The candidate"
    sources = ["profile.candidateName"] if profile.candidate_name else []
    first_parts: list[str] = []
    if profile.headline:
        first_parts.append(f"works as {profile.headline}")
        sources.append("profile.headline")
    if profile.total_experience_years is not None:
        first_parts.append(f"has {profile.total_experience_years:g} years of recorded experience")
        sources.append("profile.totalExperienceYears")
    if first_parts:
        first = f"{subject} {' and '.join(first_parts)}."
    else:
        first = f"{subject} is the candidate represented by the extracted profile."
        sources.append("profile")

    second_parts: list[str] = []
    if profile.skills:
        names = [skill.canonical_name for skill in profile.skills[:5]]
        second_parts.append(f"lists skills in {', '.join(names)}")
        sources.append("profile.skills")
    if profile.education:
        second_parts.append(f"records education at {profile.education[0].institution}")
        sources.append("profile.education")
    if profile.certifications:
        second_parts.append(f"includes {len(profile.certifications)} certification(s)")
        sources.append("profile.certifications")
    if second_parts:
        second = f"The profile {' and '.join(second_parts)}."
    else:
        second = "No structured skills, education, or certifications were extracted."
        sources.extend(["profile.skills", "profile.education", "profile.certifications"])

    return SummaryResult(
        text=f"{first} {second}",
        sentence_count=2,
        source_fields=list(dict.fromkeys(sources)),
        reused_existing=False,
    )
