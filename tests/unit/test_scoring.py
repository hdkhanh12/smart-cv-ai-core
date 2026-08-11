from __future__ import annotations

from datetime import date

from ai_core.schemas import (
    CVProfile,
    CVSection,
    DiagnosticStatus,
    Education,
    Experience,
    ExtractionDiagnostics,
    SectionKind,
    Skill,
    UnifiedBlock,
    UnifiedDocument,
    UnifiedPage,
)
from ai_core.scoring import (
    SENSITIVE_FIELDS_EXCLUDED,
    score_cv_quality,
    score_profile_completeness,
)


def _strong_profile() -> CVProfile:
    return CVProfile(
        candidate_name="Synthetic Candidate",
        email="candidate@example.test",
        headline="Data Engineer",
        summary="Data engineer building reliable platforms with measurable business outcomes.",
        skills=[
            Skill(
                name=name,
                canonical_name=name,
                evidence=[f"Used {name} in production"],
                section=SectionKind.SKILLS,
            )
            for name in ("Python", "SQL", "Docker")
        ],
        experiences=[
            Experience(
                job_title="Data Engineer",
                company="Example",
                start_date=date(2020, 1, 1),
                end_date=date(2024, 1, 1),
                description=[
                    "Built reliable batch pipelines processing large production datasets.",
                    "Reduced processing latency by 35%.",
                ],
                achievements=["Reduced processing latency by 35%."],
            )
        ],
        education=[Education(institution="Example University")],
        languages=["English"],
        sections=[
            CVSection(
                kind=kind,
                text="Synthetic evidence",
                start_line=index,
                end_line=index,
                confidence=1,
            )
            for index, kind in enumerate(
                (
                    SectionKind.SUMMARY,
                    SectionKind.SKILLS,
                    SectionKind.EXPERIENCE,
                    SectionKind.EDUCATION,
                    SectionKind.PROJECTS,
                ),
                start=1,
            )
        ],
        field_confidence={"candidateName": 0.95},
    )


def test_completeness_has_configured_breakdown_and_score_levels() -> None:
    empty = score_profile_completeness(CVProfile())
    strong = score_profile_completeness(_strong_profile())
    assert empty.score == 0
    assert strong.score == 100
    assert strong.confidence == 1
    assert set(strong.breakdown) == {
        "candidateIdentity",
        "contact",
        "professionalSummary",
        "skills",
        "experience",
        "education",
        "additional",
    }
    assert len(empty.warnings) == 6


def test_quality_scores_evidence_achievements_timeline_and_structure() -> None:
    weak = score_cv_quality(CVProfile())
    strong = score_cv_quality(_strong_profile())
    assert strong.score > weak.score
    assert strong.breakdown["experienceDescription"] > 0
    assert strong.breakdown["skillEvidence"] == 40
    assert strong.breakdown["quantifiedAchievements"] == 50
    assert strong.breakdown["timelineConsistency"] == 100
    assert strong.breakdown["structure"] == 100
    assert 0 <= strong.confidence <= 1


def test_quality_score_is_invariant_to_sensitive_attributes() -> None:
    base = _strong_profile()
    changed = base.model_copy(
        update={
            "candidate_name": "Different Person",
            "email": "other@example.test",
            "phone": "+84912345678",
            "address": "Different Address",
            "date_of_birth": date(1970, 1, 1),
        }
    )
    assert score_cv_quality(base) == score_cv_quality(changed)
    assert {
        "candidateName",
        "email",
        "phone",
        "urls",
        "address",
        "dateOfBirth",
    } == SENSITIVE_FIELDS_EXCLUDED


def test_scores_are_deterministic_for_same_profile() -> None:
    profile = _strong_profile()
    assert score_profile_completeness(profile) == score_profile_completeness(profile)
    assert score_cv_quality(profile) == score_cv_quality(profile)


def test_document_features_make_scores_independent_of_llm_wording() -> None:
    blocks = [
        "NGUYEN VAN A",
        "candidate@example.com",
        "## SUMMARY",
        "Experienced data engineer building reliable systems.",
        "## TECHNICAL SKILLS",
        "Python, SQL",
        "## WORK EXPERIENCE",
        "Built pipeline that processed 44M records and reduced latency by 35%.",
        "## PROJECTS",
        "Built a data quality project with Python.",
        "## EDUCATION",
        "Example University",
    ]
    document = UnifiedDocument(
        file_hash="a" * 64,
        extractor="fixture",
        markdown="\n".join(blocks),
        pages=[
            UnifiedPage(
                page_number=1,
                blocks=[
                    UnifiedBlock(id=f"p1-b{index}", text=value, reading_order=index)
                    for index, value in enumerate(blocks, start=1)
                ],
            )
        ],
        diagnostics=ExtractionDiagnostics(
            status=DiagnosticStatus.USABLE,
            text_character_count=300,
            block_count=len(blocks),
            heading_count=5,
        ),
    )
    first = CVProfile(
        validation_status="validated",
        skills=[Skill(name="Python", canonical_name="Python")],
    )
    second = CVProfile(validation_status="validated", summary="Different LLM wording.")

    assert score_profile_completeness(first, document) == score_profile_completeness(
        second, document
    )
    assert score_cv_quality(first, document) == score_cv_quality(second, document)
