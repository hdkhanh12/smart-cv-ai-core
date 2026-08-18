"""Deterministic mapping from LLM extraction DTOs to the internal domain model."""

from __future__ import annotations

from ai_core.extraction.dto import LLMExtractedProfile
from ai_core.extraction.profile import canonical_skill_name
from ai_core.schemas import (
    CVProfile,
    Education,
    EvaluatedTiers,
    Experience,
    HonorAward,
    LanguageProficiency,
    Project,
    Skill,
)


def _normalize_url(url: str) -> str | None:
    url_str = url.strip()
    if not url_str or " " in url_str or "." not in url_str:
        return None
    if not (url_str.startswith("http://") or url_str.startswith("https://")):
        return f"https://{url_str}"
    return url_str


def map_llm_profile(extracted: LLMExtractedProfile) -> CVProfile:
    """Populate internal-only fields deterministically after LLM validation."""

    skills = [
        Skill(
            name=skill.name,
            canonical_name=canonical_skill_name(skill.name),
            evidence=[evidence.text for evidence in skill.evidence],
            confidence=1.0 if skill.evidence else 0.0,
        )
        for skill in extracted.skills
    ]
    experiences = [
        Experience(
            job_title=entry.job_title,
            company=entry.company,
            location=entry.location,
            start_date=entry.start_date,
            end_date=entry.end_date,
            is_current=entry.is_current,
            description=entry.description,
            achievements=entry.achievements,
            skills=entry.skills,
            evidence=entry.evidence,
            confidence=1.0 if entry.evidence else 0.0,
        )
        for entry in extracted.experiences
    ]
    honors_awards = [
        HonorAward(
            title=item.title,
            issuer=item.issuer,
            award_date=item.award_date,
            evidence=item.evidence,
            confidence=1.0 if item.evidence else 0.0,
        )
        for item in getattr(extracted, "honors_awards", [])
    ]
    language_proficiencies = [
        LanguageProficiency(
            language=item.language,
            proficiency=item.proficiency,
            exam=item.exam,
            score=item.score,
            scale=item.scale,
            issuer=item.issuer,
            exam_date=item.exam_date,
            evidence=item.evidence,
            confidence=1.0 if item.evidence else 0.0,
        )
        for item in getattr(extracted, "language_proficiencies", [])
    ]
    projects = [
        Project(
            title=entry.title,
            role=entry.role,
            description=entry.description,
            skills=entry.skills,
            achievements=entry.achievements,
            start_date=entry.start_date,
            end_date=entry.end_date,
            evidence=entry.evidence,
            confidence=1.0 if entry.evidence else 0.0,
        )
        for entry in extracted.projects
    ]
    education = [
        Education(
            institution=entry.institution,
            degree=entry.degree,
            field_of_study=entry.field_of_study,
            location=entry.location,
            start_date=entry.start_date,
            end_date=entry.end_date,
            gpa=entry.gpa,
            coursework=entry.coursework,
            evidence=entry.evidence,
            confidence=1.0 if entry.evidence else 0.0,
        )
        for entry in extracted.education
    ]
    normalized_urls = [
        norm
        for u in extracted.urls
        if u and isinstance(u, str) and (norm := _normalize_url(u)) is not None
    ]

    # Map V2 evaluated tiers from LLM DTO to domain model
    evaluated_tiers: EvaluatedTiers | None = None
    llm_tiers = getattr(extracted, "evaluated_tiers", None)
    if llm_tiers is not None:
        evaluated_tiers = EvaluatedTiers(
            education_tier=getattr(llm_tiers, "education_tier", None),
            company_prestige_tier=getattr(llm_tiers, "company_prestige_tier", None),
            skill_evidence_level=getattr(llm_tiers, "skill_evidence_level", None),
            project_quality_tier=getattr(llm_tiers, "project_quality_tier", None),
            certification_tier=getattr(llm_tiers, "certification_tier", None),
            language_proficiency=getattr(llm_tiers, "language_proficiency", None),
        )

    return CVProfile(
        candidate_name=extracted.candidate_name,
        headline=extracted.headline,
        summary=extracted.professional_summary,
        email=extracted.email,
        phone=extracted.phone,
        urls=normalized_urls,
        skills=skills,
        experiences=experiences,
        projects=projects,
        education=education,
        languages=extracted.languages,
        language_proficiencies=language_proficiencies,
        certifications=extracted.certifications,
        honors_awards=honors_awards,
        field_evidence=extracted.evidence,
        field_confidence={
            field_name: 1.0 if evidence else 0.0
            for field_name, evidence in extracted.evidence.items()
        },
        # V2 IT Scoring fields
        primary_role_domain=getattr(extracted, "primary_role_domain", None),
        evaluated_tiers=evaluated_tiers,
        executive_summary=getattr(extracted, "executive_summary", None),
    )

