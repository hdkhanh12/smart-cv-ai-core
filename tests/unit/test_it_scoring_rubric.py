"""Unit tests for V2 IT Scoring Rubric Engine (6-criteria x 100-point scale).

Covers 3 archetypes:
  1. Senior BigTech (95-100)
  2. Junior Fresher (55-70)
  3. Non-tech Self-taught (35-50)

Plus: edge cases, heuristic fallback, tier-based scoring, and performance.
"""

from __future__ import annotations

import time
from datetime import date

import pytest

from ai_core.api.scoring_v2 import (
    CRITERIA_NAMES,
    compute_it_rubric_scoring,
)
from ai_core.schemas import (
    CVProfile,
    Education,
    EvaluatedTiers,
    Experience,
    LanguageProficiency,
    Project,
    Skill,
)


# ── Helper: construct test profiles ──────────────────────────────────────────


def _senior_bigtech_profile() -> CVProfile:
    """Senior Backend Engineer at BigTech with 6+ years, advanced skills, metrics."""
    return CVProfile(
        candidate_name="Nguyễn Văn A",
        email="senior@example.com",
        headline="Senior Backend Engineer",
        summary="Senior Backend Engineer with 6+ years",
        skills=[
            Skill(
                name=name,
                canonical_name=name,
                evidence=[f"Used {name} in production at scale"],
            )
            for name in (
                "Golang", "Kubernetes", "Kafka", "PostgreSQL", "Docker",
                "Redis", "AWS", "Microservices", "System Design",
            )
        ],
        experiences=[
            Experience(
                job_title="Senior Backend Engineer",
                company="Viettel High Tech",
                start_date=date(2019, 2, 1),
                is_current=True,
                description=["Designed distributed microservice architecture processing 50k RPS"],
                achievements=["Reduced system latency by 40%", "Handled 100k concurrent users"],
                skills=["Golang", "Kubernetes", "Kafka"],
            ),
            Experience(
                job_title="Backend Engineer",
                company="FPT Software",
                start_date=date(2017, 6, 1),
                end_date=date(2019, 1, 31),
                description=["Built REST APIs serving mobile application"],
                achievements=["Increased API throughput by 30%"],
                skills=["Java", "Spring Boot"],
            ),
        ],
        education=[
            Education(
                institution="Đại học Bách Khoa Hà Nội",
                degree="Kỹ sư Giỏi",
                field_of_study="Công nghệ Thông tin",
                start_date=date(2013, 9, 1),
                end_date=date(2017, 6, 1),
            ),
        ],
        projects=[
            Project(
                title="High-Performance Trading System",
                description=["Built low-latency trading platform"],
                achievements=["Processed 500k transactions/day with 99.99% uptime"],
                skills=["Golang", "Redis", "Kafka"],
            ),
            Project(
                title="Data Pipeline Platform",
                description=["ETL pipeline processing 10TB daily"],
                achievements=["Reduced processing cost by 35%"],
                skills=["Apache Spark", "Airflow"],
            ),
            Project(
                title="API Gateway",
                description=["Centralized gateway for microservices"],
                achievements=["Handled 100k concurrent requests"],
                skills=["Kong", "Kubernetes"],
            ),
        ],
        languages=["English"],
        language_proficiencies=[
            LanguageProficiency(
                language="English", exam="TOEIC", score=870, scale=990
            ),
        ],
        certifications=["AWS Certified Solutions Architect - Associate"],
        total_experience_years=6.5,
        primary_role_domain="BACKEND_CLOUD",
        evaluated_tiers=EvaluatedTiers(
            education_tier="TIER_1A_ELITE",
            company_prestige_tier="TIER_1_BIGTECH_ENTERPRISE",
            skill_evidence_level="ADVANCED_EVIDENCE_BASED",
            project_quality_tier="HIGH_IMPACT_METRICS",
            certification_tier="ASSOCIATE_PRACTITIONER",
            language_proficiency="EXPERT_FLUENT",
        ),
        executive_summary=(
            "Kỹ sư Backend Cấp cao với hơn 6 năm kinh nghiệm chuyên sâu về kiến trúc "
            "Microservices và hệ thống phân tán chịu tải cao tại Viettel. Nắm vững Golang, "
            "Kubernetes và Kafka với thành tựu tối ưu hóa độ trễ hệ thống 40%."
        ),
    )


def _junior_fresher_profile() -> CVProfile:
    """Junior Developer, ~1 year, basic skills, no certifications."""
    return CVProfile(
        candidate_name="Trần Thị B",
        email="junior@example.com",
        headline="Junior Frontend Developer",
        skills=[
            Skill(name=name, canonical_name=name)
            for name in ("JavaScript", "React", "CSS")
        ],
        experiences=[
            Experience(
                job_title="Junior Frontend Developer",
                company="Công ty Nhỏ ABC",
                start_date=date(2023, 6, 1),
                is_current=True,
                description=["Built company website frontend"],
                skills=["React", "JavaScript"],
            ),
        ],
        education=[
            Education(
                institution="Đại học Sư Phạm",
                degree="Cử nhân",
                field_of_study="Sư phạm Toán",
            ),
        ],
        total_experience_years=1.2,
    )


def _self_taught_profile() -> CVProfile:
    """Self-taught developer, no formal education, intern level."""
    return CVProfile(
        candidate_name="Lê Văn C",
        skills=[
            Skill(name="HTML", canonical_name="HTML"),
            Skill(name="CSS", canonical_name="CSS"),
        ],
        education=[],
        experiences=[],
        total_experience_years=0.0,
    )


# ── Test: Senior BigTech Profile ─────────────────────────────────────────────


class TestSeniorBigTechProfile:
    """Senior BigTech profile should score 95-100 on most criteria."""

    def test_all_criteria_present(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        assert set(result["criteriaScores"].keys()) == set(CRITERIA_NAMES.keys())

    def test_education_score_is_elite(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        assert result["criteriaScores"]["EDUCATION"]["score"] == 100.0

    def test_experience_score_is_senior(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        assert result["criteriaScores"]["EXPERIENCE_YEARS"]["score"] == 100.0

    def test_enterprise_score_is_bigtech(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        assert result["criteriaScores"]["ENTERPRISE_SCALE"]["score"] == 100.0

    def test_skills_score_is_advanced(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        assert result["criteriaScores"]["TECHNICAL_SKILLS"]["score"] == 95.0

    def test_projects_score_is_high_impact(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        assert result["criteriaScores"]["PROJECTS_PORTFOLIO"]["score"] == 95.0

    def test_certifications_score_is_expert(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        # max(ASSOCIATE_PRACTITIONER=85, EXPERT_FLUENT=95) = 95
        assert result["criteriaScores"]["CERTIFICATIONS_LANGUAGE"]["score"] == 95.0

    def test_executive_summary_present(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        assert result["summary"]
        assert "Backend" in result["summary"]

    def test_max_score_is_100(self) -> None:
        result = compute_it_rubric_scoring(_senior_bigtech_profile())
        assert result["maxScore"] == 100.0


# ── Test: Junior Fresher Profile ─────────────────────────────────────────────


class TestJuniorFresherProfile:
    """Junior profile should score 55-70 on most criteria (heuristic fallback)."""

    def test_education_score_in_range(self) -> None:
        result = compute_it_rubric_scoring(_junior_fresher_profile())
        score = result["criteriaScores"]["EDUCATION"]["score"]
        assert 60.0 <= score <= 90.0, f"Education score {score} out of range"

    def test_experience_score_is_junior(self) -> None:
        result = compute_it_rubric_scoring(_junior_fresher_profile())
        score = result["criteriaScores"]["EXPERIENCE_YEARS"]["score"]
        assert 55.0 <= score <= 70.0, f"Experience score {score} out of range"

    def test_enterprise_score_is_sme(self) -> None:
        result = compute_it_rubric_scoring(_junior_fresher_profile())
        assert result["criteriaScores"]["ENTERPRISE_SCALE"]["score"] == 60.0

    def test_skills_score_is_basic(self) -> None:
        result = compute_it_rubric_scoring(_junior_fresher_profile())
        score = result["criteriaScores"]["TECHNICAL_SKILLS"]["score"]
        assert 60.0 <= score <= 85.0, f"Skills score {score} out of range"

    def test_projects_score_is_low(self) -> None:
        result = compute_it_rubric_scoring(_junior_fresher_profile())
        score = result["criteriaScores"]["PROJECTS_PORTFOLIO"]["score"]
        assert 50.0 <= score <= 80.0, f"Projects score {score} out of range"

    def test_certifications_score_is_low(self) -> None:
        result = compute_it_rubric_scoring(_junior_fresher_profile())
        score = result["criteriaScores"]["CERTIFICATIONS_LANGUAGE"]["score"]
        assert score <= 50.0, f"Certifications score {score} too high for no certs"

    def test_reason_strings_are_not_empty(self) -> None:
        result = compute_it_rubric_scoring(_junior_fresher_profile())
        for key, detail in result["criteriaScores"].items():
            assert detail["reason"], f"{key} has empty reason"
            assert isinstance(detail["reason"], str)


# ── Test: Self-taught Non-tech Profile ───────────────────────────────────────


class TestSelfTaughtProfile:
    """Self-taught profile should score 35-50 on most criteria."""

    def test_education_score_is_non_degree(self) -> None:
        result = compute_it_rubric_scoring(_self_taught_profile())
        assert result["criteriaScores"]["EDUCATION"]["score"] == 40.0

    def test_experience_score_is_intern(self) -> None:
        result = compute_it_rubric_scoring(_self_taught_profile())
        assert result["criteriaScores"]["EXPERIENCE_YEARS"]["score"] == 35.0

    def test_enterprise_score_is_default_sme(self) -> None:
        result = compute_it_rubric_scoring(_self_taught_profile())
        assert result["criteriaScores"]["ENTERPRISE_SCALE"]["score"] == 60.0

    def test_certifications_is_none(self) -> None:
        result = compute_it_rubric_scoring(_self_taught_profile())
        assert result["criteriaScores"]["CERTIFICATIONS_LANGUAGE"]["score"] == 20.0


# ── Test: Edge Cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case coverage for empty profiles, partial tiers, etc."""

    def test_empty_profile_returns_all_criteria(self) -> None:
        result = compute_it_rubric_scoring(CVProfile())
        assert len(result["criteriaScores"]) == 6
        for detail in result["criteriaScores"].values():
            assert "name" in detail
            assert "score" in detail
            assert "reason" in detail
            assert isinstance(detail["score"], float)

    def test_partial_tiers_fallback_to_heuristic(self) -> None:
        """Profile with only education_tier set should use heuristic for others."""
        profile = CVProfile(
            evaluated_tiers=EvaluatedTiers(education_tier="TIER_1A_ELITE"),
            total_experience_years=3.0,
        )
        result = compute_it_rubric_scoring(profile)
        assert result["criteriaScores"]["EDUCATION"]["score"] == 100.0
        # Others should still be scored via heuristic
        assert result["criteriaScores"]["EXPERIENCE_YEARS"]["score"] == 85.0

    def test_invalid_tier_label_falls_back(self) -> None:
        """Unknown tier label should trigger heuristic fallback."""
        profile = CVProfile(
            evaluated_tiers=EvaluatedTiers(education_tier="INVALID_TIER"),
        )
        result = compute_it_rubric_scoring(profile)
        # Should use heuristic (no education → 40.0)
        assert result["criteriaScores"]["EDUCATION"]["score"] == 40.0

    def test_criterion_names_match_spec(self) -> None:
        """All criteria names must match the Vietnamese spec."""
        expected_keys = {
            "EDUCATION", "EXPERIENCE_YEARS", "ENTERPRISE_SCALE",
            "TECHNICAL_SKILLS", "PROJECTS_PORTFOLIO", "CERTIFICATIONS_LANGUAGE",
        }
        result = compute_it_rubric_scoring(CVProfile())
        assert set(result["criteriaScores"].keys()) == expected_keys

    def test_scores_are_deterministic(self) -> None:
        """Same profile should always produce same scores."""
        profile = _senior_bigtech_profile()
        r1 = compute_it_rubric_scoring(profile)
        r2 = compute_it_rubric_scoring(profile)
        assert r1 == r2

    def test_all_scores_in_valid_range(self) -> None:
        """All scores must be between 0 and 100."""
        for factory in (_senior_bigtech_profile, _junior_fresher_profile, _self_taught_profile):
            result = compute_it_rubric_scoring(factory())
            for key, detail in result["criteriaScores"].items():
                assert 0.0 <= detail["score"] <= 100.0, f"{key} score {detail['score']} out of range"


# ── Test: Performance Benchmark ──────────────────────────────────────────────


class TestPerformance:
    """CPU scoring must complete in < 0.2ms per invocation."""

    def test_scoring_under_200_microseconds(self) -> None:
        profile = _senior_bigtech_profile()
        # Warmup
        compute_it_rubric_scoring(profile)
        compute_it_rubric_scoring(profile)

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            compute_it_rubric_scoring(profile)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.2, f"Average scoring time {avg_ms:.4f}ms exceeds 0.2ms budget"

    def test_heuristic_fallback_also_fast(self) -> None:
        """Heuristic mode (no tiers) should also be fast."""
        profile = _senior_bigtech_profile().model_copy(update={"evaluated_tiers": None})

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            compute_it_rubric_scoring(profile)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.5, f"Average heuristic scoring time {avg_ms:.4f}ms exceeds 0.5ms budget"


# ── Test: Backward Compatibility ─────────────────────────────────────────────


class TestBackwardCompatibility:
    """CVProfile without V2 fields should still work perfectly."""

    def test_legacy_profile_without_v2_fields(self) -> None:
        """A profile created without any V2 fields should score without errors."""
        profile = CVProfile(
            candidate_name="Legacy User",
            email="legacy@test.com",
            skills=[Skill(name="Python", canonical_name="Python")],
            experiences=[
                Experience(
                    job_title="Developer",
                    company="Startup",
                    start_date=date(2022, 1, 1),
                    is_current=True,
                )
            ],
            education=[Education(institution="Some University")],
            total_experience_years=2.0,
        )
        result = compute_it_rubric_scoring(profile)
        assert len(result["criteriaScores"]) == 6
        assert result["summary"] == ""  # No executive_summary set

    def test_v2_fields_default_to_none(self) -> None:
        """Default CVProfile should have V2 fields as None."""
        profile = CVProfile()
        assert profile.primary_role_domain is None
        assert profile.evaluated_tiers is None
        assert profile.executive_summary is None
