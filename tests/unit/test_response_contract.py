"""Unit tests for Milestone 1 response contract and scoring rubric adapters."""

from datetime import date
from ai_core.api.response_contract import (
    to_embedded_payload,
    to_extracted_payload,
    to_scored_payload,
)
from ai_core.api.scoring_rubric import compute_rubric_scoring
from ai_core.schemas import (
    CVProfile,
    Education,
    EmbeddingResult,
    Experience,
    ProcessingResult,
    ProcessingStatus,
    Skill,
    SummaryResult,
)


def _sample_profile() -> CVProfile:
    return CVProfile(
        candidate_name="Nguyễn Văn A",
        email="nguyenvana@example.com",
        phone="0901234567",
        address="Quận 1, TP.HCM",
        headline="Senior Backend Engineer",
        summary="Kỹ sư Backend với 5 năm kinh nghiệm Python.",
        skills=[
            Skill(name="Python", canonical_name="Python"),
            Skill(name="FastAPI", canonical_name="FastAPI"),
            Skill(name="AWS", canonical_name="AWS"),
            Skill(name="Docker", canonical_name="Docker"),
        ],
        experiences=[
            Experience(
                job_title="Senior Backend Engineer",
                company="FPT Software",
                start_date=date(2021, 1, 1),
                end_date=date(2024, 1, 1),
                description=["Phát triển API microservices."],
            ),
            Experience(
                job_title="Backend Developer",
                company="VNG Corporation",
                start_date=date(2019, 1, 1),
                end_date=date(2020, 12, 31),
                description=["Viết dịch vụ backend Python."],
            ),
        ],
        education=[
            Education(
                institution="Đại học Bách Khoa TP.HCM",
                degree="Kỹ sư",
                field_of_study="Công nghệ Thông tin",
                start_date=date(2014, 9, 1),
                end_date=date(2018, 6, 1),
            )
        ],
        languages=["Tiếng Việt", "Tiếng Anh"],
        total_experience_years=5.0,
    )


def _sample_result() -> ProcessingResult:
    return ProcessingResult(
        status=ProcessingStatus.SUCCEEDED,
        source_id="a" * 64,
        profile=_sample_profile(),
        summary=SummaryResult(text="Kỹ sư Backend 5 năm kinh nghiệm.", sentence_count=2, source_fields=["skills"]),
        embedding=EmbeddingResult(
            model="BAAI/bge-m3",
            model_revision="main",
            dimension=1024,
            normalized=True,
            vector=[0.1] * 1024,
            source_hash="a" * 64,
            template_version="bge-m3-profile-v1",
            duration_ms=1.5,
        ),
    )


def test_compute_rubric_scoring():
    result = _sample_result()
    score, details = compute_rubric_scoring(result)

    assert 0.0 <= score <= 100.0
    assert len(details) == 4
    criterion_names = [d["criterionName"] for d in details]
    assert criterion_names == ["Học vấn", "Kinh nghiệm", "Kỹ năng", "Chứng chỉ"]


def test_to_extracted_payload():
    result = _sample_result()
    payload = to_extracted_payload(result)

    assert payload["status"] == "Extracted"
    assert payload["candidateName"] == "Nguyễn Văn A"
    assert payload["email"] == "nguyenvana@example.com"
    assert payload["phone"] == "0901234567"
    assert payload["estimatedBirthYear"] == 1996  # 2014 - 18
    assert payload["highestEducation"] == "Đại học Bách Khoa TP.HCM"
    assert payload["yearsOfExperience"] == 5.0
    assert payload["companies"] == ["FPT Software", "VNG Corporation"]
    assert "Backend Developer" in payload["jobTitles"]
    assert len(payload["skills"]) == 4
    assert payload["skills"][0] == {"skillName": "Python", "years": None, "level": None}
    assert "profileJson" in payload
    assert "piiFields" in payload
    assert payload["piiFields"]["containsPii"] is True


def test_to_scored_payload():
    result = _sample_result()
    payload = to_scored_payload(result)

    assert payload["status"] == "Scored"
    assert "Score" in payload
    assert "criteriaScores" in payload
    assert "summary" in payload


def test_to_embedded_payload():
    result = _sample_result()
    payload = to_embedded_payload(result)

    assert payload["status"] == "Embedded"
    assert payload["reason"] is None
    assert "embedding" in payload
    assert payload["dimension"] == 1024
    assert payload["model"] == "BAAI/bge-m3"
    assert payload["version"] == "v2.3.0-bgem3"
    assert len(payload["embedding"]) == 1024
