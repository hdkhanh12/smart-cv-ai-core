"""Unit tests for IT Scoring Rubric V3 (Calibrated Rubric & Ceiling Gate Guardrails)."""

from __future__ import annotations

import pytest

from ai_core.api.response_contract import to_scored_payload
from ai_core.api.scoring_v2 import (
    _TIER_CEILING_GATES,
    _V3_WEIGHTS,
    _validate_ceiling_gate,
    compute_it_rubric_scoring,
)
from ai_core.schemas import CriterionScore, CVProfile, ProcessingResult


def _sample_v3_profile_senior() -> CVProfile:
    """Create a mock Senior CVProfile with V3 rubric scores from AI."""
    return CVProfile(
        candidate_name="Phong Tran",
        executive_summary="[EN] Senior Data Engineer with 6+ years...\n[VI] Kỹ sư dữ liệu cấp cao với hơn 6 năm...",
        total_experience_years=6.5,
        rubric_scores={
            "TECHNICAL_DEPTH": CriterionScore(
                name="Kỹ năng & Kiến trúc",
                score=96.0,
                tier="PRINCIPAL_ARCHITECT",
                explanation="Tự thiết kế hệ thống Data Lakehouse trên Azure Synapse và ADF.",
                evidence_summary=["Data Lakehouse", "Azure Synapse", "ADF ETL Pipelines"],
            ),
            "IMPACT_METRICS": CriterionScore(
                name="Dự án & Thành tựu",
                score=94.0,
                tier="HIGH_IMPACT_ELITE",
                explanation="Tối ưu chi phí Cloud giảm 40%, xử lý hàng triệu bản ghi/ngày.",
                evidence_summary=["Giảm 40% chi phí Cloud", "Hàng triệu bản ghi/ngày"],
            ),
            "ENTERPRISE_SCALE": CriterionScore(
                name="Quy mô Doanh nghiệp",
                score=95.0,
                tier="TIER_1_BIGTECH_ELITE",
                explanation="Từng làm việc tại Version 1 và các đối tác Enterprise toàn cầu.",
                evidence_summary=["Version 1", "Global Clients"],
            ),
            "EDUCATION": CriterionScore(
                name="Học vấn & CS Foundation",
                score=95.0,
                tier="TIER_1A_ELITE_PLUS",
                explanation="Tốt nghiệp Thạc sĩ MSc Information Systems tại University College Dublin.",
                evidence_summary=["University College Dublin (MSc)"],
            ),
            "CERTIFICATIONS": CriterionScore(
                name="Chứng chỉ Chuyên môn",
                score=85.0,
                tier="ASSOCIATE_PRACTITIONER",
                explanation="Sở hữu các chứng chỉ Microsoft Fabric & Azure Data Engineer Associate.",
                evidence_summary=["Fabric Data Engineer Associate", "Azure Data Engineer Associate"],
            ),
            "LANGUAGE_PROFICIENCY": CriterionScore(
                name="Năng lực Ngoại ngữ",
                score=92.0,
                tier="EXPERT_FLUENT",
                explanation="Làm việc trực tiếp tại Ireland và môi trường quốc tế 100% tiếng Anh.",
                evidence_summary=["Ireland market", "English native environment"],
            ),
        },
    )


def test_v3_scoring_calculation():
    """Verify that V3 weighted score is computed accurately using industry weights."""
    profile = _sample_v3_profile_senior()
    result = compute_it_rubric_scoring(profile)

    assert result["scoringVersion"] == "v3_rubric"
    assert "criteriaScores" in result
    cs = result["criteriaScores"]

    assert cs["TECHNICAL_DEPTH"]["score"] == 96.0
    assert cs["IMPACT_METRICS"]["score"] == 94.0
    assert cs["ENTERPRISE_SCALE"]["score"] == 95.0
    assert cs["EDUCATION"]["score"] == 95.0
    assert cs["CERTIFICATIONS"]["score"] == 85.0
    assert cs["LANGUAGE_PROFICIENCY"]["score"] == 92.0

    # Calculate expected weighted score:
    # 96*0.30 (28.8) + 94*0.25 (23.5) + 95*0.15 (14.25) + 95*0.15 (14.25) + 85*0.10 (8.5) + 92*0.05 (4.6) = 93.9
    expected_score = round(
        96.0 * 0.30 + 94.0 * 0.25 + 95.0 * 0.15 + 95.0 * 0.15 + 85.0 * 0.10 + 92.0 * 0.05,
        1,
    )
    assert result["Score"] == expected_score
    assert result["seniorityCalibratedLevel"] == "SENIOR_LEAD"
    assert result["hiringSignal"] == "STRONG_RECOMMEND"


def test_ceiling_gate_clamping():
    """Verify that Ceiling Gate guardrail clamps scores exceeding the tier upper bound."""
    # When tier has lower ceiling (e.g. COMPETENT_PRODUCTION capped at 83), score 95 is clamped to 83
    assert _validate_ceiling_gate("TECHNICAL_DEPTH", 95.0, "COMPETENT_PRODUCTION") == 83.0
    # When tier has lower ceiling (e.g. STANDARD_COMPLETED_DESCRIBED capped at 79), score 88 is clamped to 79
    assert _validate_ceiling_gate("IMPACT_METRICS", 88.0, "STANDARD_COMPLETED_DESCRIBED") == 79.0
    # When tier is NON_DEGREE capped at 54, score 70 is clamped to 54
    assert _validate_ceiling_gate("EDUCATION", 70.0, "NON_DEGREE") == 54.0
    # When tier is NONE capped at 39, score 60 is clamped to 39
    assert _validate_ceiling_gate("CERTIFICATIONS", 60.0, "NONE") == 39.0
    # When tier is STANDARD_SME capped at 64, score 80 is clamped to 64
    assert _validate_ceiling_gate("ENTERPRISE_SCALE", 80.0, "STANDARD_SME") == 64.0
    # When tier is ELITE/PRINCIPAL, high score is preserved
    assert _validate_ceiling_gate("TECHNICAL_DEPTH", 96.0, "PRINCIPAL_ARCHITECT") == 96.0


def test_v3_to_scored_payload():
    """Verify that response_contract.to_scored_payload properly formats V3 fields."""
    from ai_core.schemas import ProcessingStatus
    profile = _sample_v3_profile_senior()
    proc_result = ProcessingResult(
        source_id="test-doc-001",
        status=ProcessingStatus.SUCCEEDED,
        profile=profile,
    )
    payload = to_scored_payload(proc_result)

    assert payload["status"] == "Scored"
    assert payload["scoringVersion"] == "v3_rubric"
    assert payload["Score"] > 90.0
    assert payload["candidateName"] == "Phong Tran"
    assert "criteriaScores" in payload
    assert "scoringRationale" in payload
    assert "seniorityCalibratedLevel" in payload
    assert "hiringSignal" in payload
    assert len(payload["criteriaScores"]) == 6


def test_v3_fallback_to_v2():
    """Verify that when rubric_scores is None, the system cleanly falls back to V2."""
    profile_without_v3 = CVProfile(
        candidate_name="Fresher Test",
        total_experience_years=0.5,
        rubric_scores=None,
    )
    result = compute_it_rubric_scoring(profile_without_v3)
    assert result["scoringVersion"] == "v2_tier_mapping"
    assert "criteriaScores" in result
