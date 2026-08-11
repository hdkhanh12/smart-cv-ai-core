"""Explainable deterministic scoring with Hybrid v2 quality gates."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai_core.errors import Issue, WarningCode
from ai_core.schemas import (
    CVProfile,
    EvidenceType,
    ScoreResult,
    ScoreStatus,
    SectionKind,
    UnifiedDocument,
    ValidationStatus,
)
from ai_core.scoring.features import DocumentScoreFeatures, document_score_features

SENSITIVE_FIELDS_EXCLUDED = frozenset(
    {"candidateName", "email", "phone", "urls", "address", "dateOfBirth"}
)


@lru_cache(maxsize=1)
def _config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "configs" / "scoring.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    excluded = frozenset(str(value) for value in payload["excludedSensitiveFields"])
    if excluded != SENSITIVE_FIELDS_EXCLUDED:
        raise ValueError("scoring sensitive-field exclusion config is invalid")
    for group in ("completeness", "quality"):
        if abs(sum(float(value) for value in payload[group].values()) - 1.0) > 1e-9:
            raise ValueError(f"{group} weights must sum to 1")
    return payload


def _weighted_score(
    breakdown: dict[str, float | None],
    weights: dict[str, Any],
) -> float:
    applicable = {
        name: float(weight) for name, weight in weights.items() if breakdown.get(name) is not None
    }
    total_weight = sum(applicable.values())
    if not total_weight:
        return 0.0
    return round(
        sum(float(breakdown[name] or 0) * weight for name, weight in applicable.items())
        / total_weight,
        2,
    )


def score_profile_completeness(
    profile: CVProfile,
    document: UnifiedDocument | None = None,
) -> ScoreResult:
    """Measure completeness of extracted data, not candidate quality."""

    weights: dict[str, Any] = _config()["completeness"]
    features = document_score_features(document) if document is not None else None
    valid_email = bool(
        profile.email and re.fullmatch(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", profile.email)
    )
    phone_digits = re.sub(r"\D", "", profile.phone or "")
    education_value = (
        (100.0 if features.has_education else None)
        if features is not None
        else (100.0 if profile.education else None)
    )
    has_additional = (
        features.has_additional
        if features is not None
        else bool(profile.languages or profile.certifications or profile.urls)
    )
    additional_value = 100.0 if has_additional else 0.0
    identity_value = (
        (100.0 if features.has_identity else 0.0)
        if features is not None
        else 100.0
        if profile.candidate_name and profile.field_confidence.get("candidateName", 0) >= 0.6
        else 0.0
    )
    contact_value = (
        (100.0 if features.has_contact else 0.0)
        if features is not None
        else 100.0
        if valid_email or 9 <= len(phone_digits) <= 15
        else 0.0
    )
    summary_value = (
        (100.0 if features.has_summary else 0.0)
        if features is not None
        else 100.0
        if profile.summary and len(profile.summary) >= 40
        else 60.0
        if profile.headline
        else 0.0
    )
    experience_value = (
        (100.0 if features.has_experience else 0.0)
        if features is not None
        else 100.0
        if any(entry.job_title and entry.start_date for entry in profile.experiences)
        else 40.0
        if profile.experiences
        else 0.0
    )
    breakdown: dict[str, float | None] = {
        "candidateIdentity": identity_value,
        "contact": contact_value,
        "professionalSummary": summary_value,
        "skills": min(100.0, len(features.skill_types) / 3 * 100)
        if features
        else min(100.0, len(profile.skills) / 3 * 100),
        "experience": experience_value,
        # Education is optional: absent is N/A, not a profile defect.
        "education": education_value,
        "additional": additional_value,
    }
    warnings = [
        Issue(
            code=WarningCode.MISSING_PROFILE_FIELD,
            message=f"Extracted profile group '{name}' is missing or invalid.",
            stage="completeness_scoring",
            details={"group": name},
        )
        for name, value in breakdown.items()
        if value == 0
    ]
    applicable = [value for value in breakdown.values() if value is not None]
    return ScoreResult(
        name="profile_completeness.v2",
        score=_weighted_score(breakdown, weights),
        breakdown=breakdown,
        confidence=round(sum(value > 0 for value in applicable) / len(applicable), 2),
        warnings=warnings,
    )


def _professional_description_score(profile: CVProfile) -> float | None:
    descriptions = [entry.description for entry in profile.experiences] + [
        project.description for project in profile.projects
    ]
    if not descriptions:
        return None
    scores = []
    for values in descriptions:
        clean = [value for value in values if value.strip()]
        average_length = sum(map(len, clean)) / len(clean) if clean else 0
        scores.append(min(100.0, 40.0 + average_length) if clean else 0.0)
    return sum(scores) / len(scores)


def _achievement_count(profile: CVProfile) -> tuple[int, bool]:
    values = [value for entry in profile.experiences for value in entry.achievements] + [
        value for project in profile.projects for value in project.achievements
    ]
    impact = re.compile(
        r"\b(increased|reduced|improved|saved|grew|processed|served|tăng|giảm|cải thiện)\b",
        re.I,
    )
    quantity = re.compile(
        r"\b\d+(?:[.,]\d+)?(?:%|[kmb]\b|\s+(?:users|records|events|requests))",
        re.I,
    )
    return sum(bool(impact.search(value) and quantity.search(value)) for value in values), bool(
        values
    )


def score_cv_quality(profile: CVProfile, document: UnifiedDocument | None = None) -> ScoreResult:
    """Score professional evidence, suppressing output on manual review."""

    if profile.validation_status == ValidationStatus.MANUAL_REVIEW:
        return ScoreResult(
            name="cv_quality.v2",
            status=ScoreStatus.INSUFFICIENT_DATA,
            score=None,
            breakdown={
                "experienceDescription": None,
                "skillEvidence": None,
                "quantifiedAchievements": None,
                "timelineConsistency": None,
                "structure": None,
            },
            confidence=0.25,
            warnings=[
                Issue(
                    code=WarningCode.MANUAL_REVIEW_REQUIRED,
                    message="Quality score is suppressed until extraction is manually reviewed.",
                    stage="quality_scoring",
                )
            ],
        )

    weights: dict[str, Any] = _config()["quality"]
    features: DocumentScoreFeatures | None = (
        document_score_features(document) if document is not None else None
    )
    evidence_scores = []
    evidence_sets = (
        ((set(types), True) for types in features.skill_types.values())
        if features is not None
        else ((set(skill.evidence_types), bool(skill.evidence)) for skill in profile.skills)
    )
    for types, has_evidence in evidence_sets:
        evidence_scores.append(
            100.0
            if EvidenceType.WORK_EXPERIENCE in types
            else 80.0
            if types & {EvidenceType.PROJECT, EvidenceType.RESEARCH}
            else 40.0
            if EvidenceType.LISTED in types or has_evidence
            else 0.0
        )
    achievement_count, has_achievement_values = (
        (features.achievement_count, features.has_experience)
        if features is not None
        else _achievement_count(profile)
    )
    has_timeline_warning = any(
        warning.code == WarningCode.INVALID_TIMELINE for warning in profile.warnings
    )
    structural = {
        SectionKind.SUMMARY,
        SectionKind.SKILLS,
        SectionKind.EXPERIENCE,
        SectionKind.EDUCATION,
        SectionKind.PROJECTS,
    }
    section_kinds = {section.kind for section in profile.sections}
    timeline_value = (
        0.0
        if has_timeline_warning
        else 100.0
        if (features.has_experience if features is not None else profile.experiences)
        else None
    )
    breakdown: dict[str, float | None] = {
        "experienceDescription": (
            features.description_score
            if features is not None
            else _professional_description_score(profile)
        ),
        "skillEvidence": sum(evidence_scores) / len(evidence_scores) if evidence_scores else None,
        "quantifiedAchievements": min(100.0, achievement_count * 50.0)
        if has_achievement_values
        else None,
        "timelineConsistency": timeline_value,
        "structure": features.structure_count / len(structural) * 100
        if features is not None
        else len(section_kinds & structural) / len(structural) * 100
        if section_kinds
        else None,
    }
    warnings = []
    if not (
        features.has_experience if features is not None else profile.experiences or profile.projects
    ):
        warnings.append(
            Issue(
                code=WarningCode.INSUFFICIENT_EVIDENCE,
                message="No professional experience or project evidence is available.",
                stage="quality_scoring",
            )
        )
    if profile.validation_status == ValidationStatus.PARTIAL:
        warnings.append(
            Issue(
                code=WarningCode.INSUFFICIENT_EVIDENCE,
                message="Quality score is based on a partially validated extraction.",
                stage="quality_scoring",
            )
        )
    applicable = sum(value is not None for value in breakdown.values())
    confidence = applicable / len(breakdown)
    if profile.validation_status == ValidationStatus.PARTIAL:
        confidence *= 0.7
    return ScoreResult(
        name="cv_quality.v2",
        score=_weighted_score(breakdown, weights),
        breakdown={
            name: round(value, 2) if value is not None else None
            for name, value in breakdown.items()
        },
        confidence=round(confidence, 2),
        warnings=warnings,
    )
