"""Unit tests for Milestone 4.5 extraction refinements."""

from __future__ import annotations

from datetime import date

from ai_core.reconciliation.profile import (
    _separate_quantified_achievements,
    reconcile_profile,
)
from ai_core.schemas import (
    CVProfile,
    DiagnosticStatus,
    EvidenceRef,
    EvidenceType,
    Experience,
    ExtractionDiagnostics,
    Project,
    Skill,
    UnifiedBlock,
    UnifiedDocument,
    UnifiedPage,
)


def _make_dummy_doc(text_blocks: list[str]) -> UnifiedDocument:
    blocks = [
        UnifiedBlock(id=f"p1-b{i + 1}", type="text", text=text, reading_order=i + 1)
        for i, text in enumerate(text_blocks)
    ]
    return UnifiedDocument(
        file_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        extractor="docling",
        markdown="\n".join(text_blocks),
        pages=[UnifiedPage(page_number=1, blocks=blocks)],
        diagnostics=ExtractionDiagnostics(
            status=DiagnosticStatus.USABLE,
            issues=[],
            text_character_count=sum(len(t) for t in text_blocks),
            block_count=len(blocks),
            heading_count=1,
        ),
    )


def test_skill_evidence_type_classification() -> None:
    doc_text = [
        "## TECHNICAL SKILLS",
        "- Python",
        "## PROFESSIONAL EXPERIENCE",
        "Optimization Researcher at FPT SAP Lab",
        "Conducted research in genetic algorithm, tabu search, and reinforcement learning.",
        "## RESEARCH PAPER",
        "Published study on Educational Data Mining and Computer Vision in IEEE.",
    ]
    doc = _make_dummy_doc(doc_text)
    profile = CVProfile(
        candidate_name="Alex Morgan",
        skills=[
            Skill(name="Python", canonical_name="Python", evidence=["Python"]),
            Skill(
                name="Genetic Algorithm",
                canonical_name="Genetic Algorithm",
                evidence=["genetic algorithm"],
            ),
            Skill(
                name="Educational Data Mining",
                canonical_name="Educational Data Mining",
                evidence=["Educational Data Mining"],
            ),
        ],
        experiences=[
            Experience(
                job_title="Optimization Researcher",
                company="FPT SAP Lab",
                start_date=date(2023, 1, 1),
                end_date=date(2024, 1, 1),
                description=[
                    "Conducted research in genetic algorithm, tabu search, "
                    "and reinforcement learning."
                ],
                evidence=[
                    EvidenceRef(text="Optimization Researcher", page_number=1, block_id="p1-b4")
                ],
            )
        ],
        field_evidence={
            "candidateName": [EvidenceRef(text="Alex Morgan", page_number=1, block_id="p1-b1")]
        },
    )
    reconciled = reconcile_profile(profile, doc, provider_name="beeknoee-openai-compatible")

    skill_map = {s.name: s.evidence_types for s in reconciled.skills}
    assert EvidenceType.WORK_EXPERIENCE in skill_map["Genetic Algorithm"]
    assert EvidenceType.RESEARCH in skill_map["Educational Data Mining"]


def test_reconciliation_enriches_low_cv_domain_skills_from_work_and_research() -> None:
    doc = _make_dummy_doc(
        [
            "PROFESSIONAL EXPERIENCE",
            "Developed Python crawlers for mining data extraction.",
            "Applied metaheuristic algorithms and reinforcement learning.",
            "RESEARCH PAPER",
            "Conducted research in educational data mining and computer vision.",
        ]
    )
    profile = CVProfile(
        candidate_name="Alex Morgan",
        field_evidence={
            "candidateName": [EvidenceRef(text="Alex Morgan", page_number=1, block_id="p1-b1")]
        },
    )

    result = reconcile_profile(profile, doc, provider_name="beeknoee-openai-compatible")
    skills = {skill.canonical_name: skill for skill in result.skills}

    for canonical in (
        "Web Crawling",
        "Metaheuristics",
        "Reinforcement Learning",
        "Educational Data Mining",
        "Computer Vision",
    ):
        assert canonical in skills
    assert EvidenceType.WORK_EXPERIENCE in skills["Web Crawling"].evidence_types
    assert EvidenceType.WORK_EXPERIENCE in skills["Metaheuristics"].evidence_types
    assert EvidenceType.RESEARCH in skills["Computer Vision"].evidence_types


def test_candidate_name_uses_evidence_spelling_after_normalized_comparison() -> None:
    doc = _make_dummy_doc(["ALEX   NGUYỄN", "Data Engineer"])
    profile = CVProfile(
        candidate_name="alex nguyễn",
        field_evidence={
            "candidateName": [EvidenceRef(text="ALEX   NGUYỄN", page_number=1, block_id="p1-b1")]
        },
    )

    result = reconcile_profile(profile, doc, provider_name="beeknoee-openai-compatible")

    assert result.candidate_name == "ALEX NGUYỄN"


def test_quantified_achievement_separation() -> None:
    descriptions = [
        "Processed 1.2M events per day on Kafka pipeline.",
        "Improved system performance by 70%.",
        "Contact +61 414 538 472",
        "Python 3.9 upgraded in 2024.",
        "Developed web crawlers for data mining.",
    ]
    remaining, achievements = _separate_quantified_achievements(descriptions, [])

    assert "Processed 1.2M events per day on Kafka pipeline." in achievements
    assert "Improved system performance by 70%." in achievements
    assert "Contact +61 414 538 472" in remaining
    assert "Python 3.9 upgraded in 2024." in remaining
    assert "Developed web crawlers for data mining." in remaining


def test_project_dates_do_not_inflate_total_experience_years() -> None:
    doc = _make_dummy_doc(["Alex Morgan", "Data Engineer | Company A | 2022 - 2023", "Project X"])
    profile = CVProfile(
        candidate_name="Alex Morgan",
        experiences=[
            Experience(
                job_title="Data Engineer",
                company="Company A",
                start_date=date(2022, 1, 1),
                end_date=date(2023, 1, 1),
                evidence=[
                    EvidenceRef(
                        text="Data Engineer | Company A | 2022 - 2023",
                        page_number=1,
                        block_id="p1-b2",
                    )
                ],
            )
        ],
        projects=[
            Project(
                title="Project X",
                start_date=date(2015, 1, 1),
                end_date=date(2020, 1, 1),
                evidence=[EvidenceRef(text="Project X", page_number=1, block_id="p1-b3")],
            )
        ],
        field_evidence={
            "candidateName": [EvidenceRef(text="Alex Morgan", page_number=1, block_id="p1-b1")]
        },
    )
    reconciled = reconcile_profile(profile, doc, provider_name="beeknoee-openai-compatible")

    # Only 1 year of employment (2022-2023); project date range (2015-2020)
    # MUST NOT inflate total_experience_years
    assert reconciled.total_experience_years == 1.0


def test_project_date_without_project_evidence_is_removed() -> None:
    doc = _make_dummy_doc(["Alex Morgan", "Project X", "Built a pipeline"])
    profile = CVProfile(
        candidate_name="Alex Morgan",
        projects=[
            Project(
                title="Project X",
                start_date=date(2022, 1, 1),
                end_date=date(2025, 12, 31),
                evidence=[EvidenceRef(text="Project X", page_number=1, block_id="p1-b2")],
            )
        ],
        field_evidence={
            "candidateName": [EvidenceRef(text="Alex Morgan", page_number=1, block_id="p1-b1")]
        },
    )

    result = reconcile_profile(profile, doc, provider_name="beeknoee-openai-compatible")

    assert result.projects[0].start_date is None
    assert result.projects[0].end_date is None


def test_project_without_direct_evidence_is_excluded() -> None:
    doc = _make_dummy_doc(["Alex Morgan", "Projects"])
    profile = CVProfile(
        candidate_name="Alex Morgan",
        projects=[Project(title="Ungrounded Project")],
        field_evidence={
            "candidateName": [EvidenceRef(text="Alex Morgan", page_number=1, block_id="p1-b1")]
        },
    )

    result = reconcile_profile(profile, doc, provider_name="beeknoee-openai-compatible")

    assert result.projects == []
    assert any(warning.code.value == "UNSUPPORTED_PROJECT_CLAIM" for warning in result.warnings)


def test_headline_fallback_behavior() -> None:
    doc = _make_dummy_doc(["Alex Morgan", "Data Science Intern"])
    long_summary = (
        "I am currently pursuing a Master of Data Science and Innovation, "
        "building upon a strong foundation in IT and AI. With a passion for turning data into "
        "actionable insights, I am eager to apply my skills."
    )
    profile_with_long_headline = CVProfile(
        candidate_name="Alex Morgan",
        headline=long_summary,
        summary=long_summary,
        field_evidence={
            "candidateName": [EvidenceRef(text="Alex Morgan", page_number=1, block_id="p1-b1")]
        },
    )
    reconciled = reconcile_profile(
        profile_with_long_headline, doc, provider_name="beeknoee-openai-compatible"
    )
    assert reconciled.headline is None

    profile_with_short_headline = CVProfile(
        candidate_name="Alex Morgan",
        headline="Data Science Intern",
        summary=long_summary,
        field_evidence={
            "candidateName": [EvidenceRef(text="Alex Morgan", page_number=1, block_id="p1-b1")]
        },
    )
    reconciled_short = reconcile_profile(
        profile_with_short_headline, doc, provider_name="beeknoee-openai-compatible"
    )
    assert reconciled_short.headline == "Data Science Intern"
