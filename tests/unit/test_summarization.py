from __future__ import annotations

from ai_core.schemas import CVProfile, Education, Skill
from ai_core.summarization import summarize_profile


def test_reuses_existing_summary_only_when_it_is_two_or_three_safe_sentences() -> None:
    existing = (
        "Data engineer experienced in reliable batch pipelines. "
        "Skilled in Python and SQL for analytics workloads."
    )
    result = summarize_profile(CVProfile(summary=existing))
    assert result.text == existing
    assert result.sentence_count == 2
    assert result.source_fields == ["profile.summary"]
    assert result.reused_existing is True


def test_generates_two_traceable_sentences_without_hallucinated_facts() -> None:
    profile = CVProfile(
        candidate_name="An Nguyen",
        headline="Data Engineer",
        total_experience_years=4,
        skills=[
            Skill(name="Python", canonical_name="Python"),
            Skill(name="SQL", canonical_name="SQL"),
        ],
        education=[Education(institution="Example University")],
    )
    result = summarize_profile(profile)
    assert result.sentence_count == 2
    assert result.reused_existing is False
    for fact in ("An Nguyen", "Data Engineer", "4 years", "Python", "SQL", "Example University"):
        assert fact in result.text
    for invented in ("AWS", "manager", "certified", "10 years"):
        assert invented not in result.text
    assert {
        "profile.candidateName",
        "profile.headline",
        "profile.totalExperienceYears",
        "profile.skills",
        "profile.education",
    } <= set(result.source_fields)


def test_sparse_profile_still_produces_deterministic_two_sentence_result() -> None:
    first = summarize_profile(CVProfile())
    second = summarize_profile(CVProfile())
    assert first == second
    assert first.sentence_count == 2
    assert "No structured skills" in first.text


def test_rejects_existing_summary_that_contains_contact_information() -> None:
    profile = CVProfile(
        summary=(
            "Contact me at candidate@example.com for data engineering work. "
            "I build reliable data platforms."
        )
    )
    result = summarize_profile(profile)
    assert result.reused_existing is False
    assert "candidate@example.com" not in result.text
