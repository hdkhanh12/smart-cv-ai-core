from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ai_core.extraction import ExtractionRequest
from ai_core.parsers.unified import (
    diagnose,
    effective_document_text,
    extract_with_fallback,
    unified_from_lines,
)
from ai_core.reconciliation import reconcile_profile
from ai_core.schemas import (
    CVProfile,
    DiagnosticStatus,
    EvidenceRef,
    Experience,
    Project,
    ScoreStatus,
    UnifiedDocument,
    ValidationStatus,
)
from ai_core.scoring import score_cv_quality
from ai_core.validation import ValidatedInput

FIXTURE = Path(__file__).parents[1] / "fixtures" / "hybrid_v2_cases.json"


def _case(case_id: str) -> dict[str, object]:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return next(case for case in cases if case["id"] == case_id)


def _document(tmp_path: Path, case: dict[str, object]) -> UnifiedDocument:
    source = tmp_path / f"{case['id']}.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    from ai_core.validation import validate_input

    validated = validate_input(source)
    blocks = [str(value) for value in case["blocks"]]  # type: ignore[union-attr]
    return unified_from_lines(validated, "fixture", [blocks])


def test_low_fixture_does_not_confuse_location_with_name(tmp_path: Path) -> None:
    case = _case("low_location_name")
    document = _document(tmp_path, case)
    profile = CVProfile.model_validate(case["profile"])
    result = reconcile_profile(profile, document, provider_name="fixture-llm")
    assert result.validation_status == ValidationStatus.VALIDATED
    assert result.candidate_name == "ALEX MORGAN"
    assert len(result.experiences) == 3
    assert len(result.education) == 2
    assert {"Python", "SQL", "Spark"} <= {skill.canonical_name for skill in result.skills}


def test_location_returned_as_name_requires_manual_review(tmp_path: Path) -> None:
    case = _case("low_location_name")
    document = _document(tmp_path, case)
    profile = CVProfile(
        candidate_name="Sydney, Australia",
        field_confidence={"candidateName": 0.99},
        field_evidence={
            "candidateName": [{"text": "Sydney, Australia", "pageNumber": 1, "blockId": "p1-b1"}]
        },
    )
    result = reconcile_profile(profile, document, provider_name="fixture-llm")
    assert result.candidate_name is None
    assert result.validation_status == ValidationStatus.MANUAL_REVIEW
    quality = score_cv_quality(result)
    assert quality.status == ScoreStatus.INSUFFICIENT_DATA
    assert quality.score is None


def test_mid_fixture_keeps_three_projects_out_of_employment_years(tmp_path: Path) -> None:
    case = _case("mid_personal_projects")
    result = reconcile_profile(
        CVProfile.model_validate(case["profile"]),
        _document(tmp_path, case),
        provider_name="fixture-llm",
    )
    assert result.validation_status == ValidationStatus.VALIDATED
    assert len(result.projects) == 3
    assert result.experiences == []
    assert result.total_experience_years is None
    assert any(
        "project" in {value.value for value in skill.evidence_types} for skill in result.skills
    )
    assert score_cv_quality(result).score is not None


def test_project_only_evidence_cannot_become_employment(tmp_path: Path) -> None:
    case = _case("mid_personal_projects")
    document = _document(tmp_path, case)
    hallucinated = CVProfile(
        experiences=[
            Experience(
                job_title="Data Engineer",
                company="Invented Employer",
                start_date=date(2022, 1, 1),
                is_current=True,
                evidence=[
                    EvidenceRef(
                        text="Realtime Analytics | Kafka, Spark | Processed 1.2M events daily",
                        page_number=1,
                        block_id="p1-b3",
                    )
                ],
            )
        ]
    )

    dispositions: list[dict[str, object]] = []
    result = reconcile_profile(
        hallucinated,
        document,
        provider_name="fixture-llm",
        entity_dispositions=dispositions,
    )

    assert result.experiences == []
    assert result.total_experience_years is None
    assert result.validation_status == ValidationStatus.MANUAL_REVIEW
    assert any(warning.code.value == "UNSUPPORTED_EXPERIENCE_CLAIM" for warning in result.warnings)
    assert dispositions[0]["decision"] == "dropped"
    assert dispositions[0]["reasonCode"] == "ROLE_NOT_SUPPORTED_BY_EVIDENCE"
    assert "identityFingerprint" in dispositions[0]


def test_high_fixture_preserves_role_company_location_and_cleans_unicode(
    tmp_path: Path,
) -> None:
    case = _case("high_role_company_location")
    result = reconcile_profile(
        CVProfile.model_validate(case["profile"]),
        _document(tmp_path, case),
        provider_name="fixture-llm",
    )
    experience = result.experiences[0]
    assert experience.job_title == "Platform Engineer"
    assert experience.company == "Northstar Labs"
    assert experience.location == "Berlin, Germany"
    assert result.education[0].institution == "Example University"


def test_ungrounded_evidence_forces_manual_review(tmp_path: Path) -> None:
    case = _case("high_role_company_location")
    profile = CVProfile.model_validate(case["profile"])
    bad = profile.model_copy(
        update={
            "field_evidence": {
                "candidateName": [
                    EvidenceRef(
                        text="NOT IN DOCUMENT",
                        page_number=1,
                        block_id="p1-b1",
                    )
                ]
            }
        }
    )
    result = reconcile_profile(bad, _document(tmp_path, case), provider_name="fixture-llm")
    assert result.validation_status == ValidationStatus.MANUAL_REVIEW


def test_diagnostics_routes_to_fallback_only_after_unusable_main(tmp_path: Path) -> None:
    source = tmp_path / "fallback.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    from ai_core.validation import validate_input

    validated = validate_input(source)

    class Main:
        name = "main"

        def extract(self, document: ValidatedInput) -> UnifiedDocument:
            return unified_from_lines(document, self.name, [["short"]])

    class Fallback:
        name = "fallback"

        def __init__(self) -> None:
            self.calls = 0

        def extract(
            self,
            document: ValidatedInput,
            failed: UnifiedDocument,
        ) -> UnifiedDocument:
            self.calls += 1
            return unified_from_lines(
                document,
                self.name,
                [
                    [
                        "CANDIDATE NAME",
                        "SKILLS",
                        "Python and SQL experience in production systems with reliable evidence",
                    ]
                ],
            )

    fallback = Fallback()
    result = extract_with_fallback(validated, Main(), fallback)
    assert fallback.calls == 1
    assert result.extractor == "fallback"
    assert result.diagnostics.status == DiagnosticStatus.USABLE


def test_full_document_provider_contract_is_one_request(tmp_path: Path) -> None:
    case = _case("mid_personal_projects")
    document = _document(tmp_path, case)
    calls: list[ExtractionRequest] = []

    class Provider:
        name = "fixture-llm"
        model = "fixture-model"
        revision = "1"

        def extract(self, request: ExtractionRequest) -> CVProfile:
            calls.append(request)
            return CVProfile.model_validate(case["profile"])

    from ai_core.extraction import build_request

    profile = Provider().extract(build_request(document))
    assert len(calls) == 1
    assert "MAYA TRAN" not in calls[0].document.markdown
    assert "[HEADER_REDACTED]" in calls[0].document.markdown
    assert "p1-b1" in calls[0].document.model_dump_json(by_alias=True)
    assert "properties" in calls[0].json_schema
    assert len(profile.projects) == 3


def test_versions_dates_and_phone_are_not_quantified_achievements() -> None:
    profile = CVProfile(
        validation_status=ValidationStatus.VALIDATED,
        projects=[
            Project(
                title="Synthetic",
                achievements=["Python 3.11", "Released in 2024", "Phone +84 912 345 678"],
            )
        ],
    )
    score = score_cv_quality(profile)
    assert score.breakdown["quantifiedAchievements"] == 0


def test_diagnostics_detects_non_monotonic_reading_order(tmp_path: Path) -> None:
    case = _case("high_role_company_location")
    pages = _document(tmp_path, case).pages
    pages[0].blocks[1].reading_order = 1
    result = diagnose("long enough document content for diagnostics" * 3, pages)
    assert result.status == DiagnosticStatus.FALLBACK_REQUIRED


def test_diagnostics_treats_docling_image_placeholders_as_no_text(tmp_path: Path) -> None:
    case = _case("high_role_company_location")
    pages = _document(tmp_path, case).pages
    markdown = "\n".join("<!-- image -->" for _ in range(8))

    result = diagnose(markdown, pages)

    assert effective_document_text(markdown) == ""
    assert result.status == DiagnosticStatus.FALLBACK_REQUIRED
    assert result.text_character_count == 0
    assert result.issues[0].details["imagePlaceholderCount"] == 8
