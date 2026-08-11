"""Synthetic OCR fallback routing unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_core.parsers.unified import (
    extract_with_fallback,
)
from ai_core.reconciliation.profile import reconcile_profile
from ai_core.schemas import (
    CVProfile,
    DiagnosticStatus,
    DocumentMetadata,
    EvidenceRef,
    ExtractionDiagnostics,
    UnifiedBlock,
    UnifiedDocument,
    UnifiedPage,
    ValidationStatus,
)
from ai_core.scoring.profile import score_cv_quality
from ai_core.validation import ValidatedInput


def _make_validated_input() -> ValidatedInput:
    meta = DocumentMetadata(
        source_name="synthetic_scan.pdf",
        extension=".pdf",
        media_type="application/pdf",
        size_bytes=1024,
        sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        page_count=1,
    )
    return ValidatedInput(path=Path("synthetic_scan.pdf"), metadata=meta)


class FakeMainExtractor:
    name = "fake_main"

    def __init__(self, status: DiagnosticStatus, text: str = "Sample Text") -> None:
        self.status = status
        self.text = text
        self.called = False

    def extract(self, document: ValidatedInput) -> UnifiedDocument:
        self.called = True
        return UnifiedDocument(
            file_hash=document.metadata.sha256,
            extractor=self.name,
            markdown=self.text,
            pages=[
                UnifiedPage(
                    page_number=1,
                    blocks=[UnifiedBlock(id="p1-b1", type="text", text=self.text, reading_order=1)],
                )
            ],
            diagnostics=ExtractionDiagnostics(
                status=self.status,
                issues=[],
                text_character_count=len(self.text),
                block_count=1,
                heading_count=1,
            ),
        )


class FakeOCRFallbackExtractor:
    name = "fake_rapidocr_fallback"

    def __init__(self, status: DiagnosticStatus, text: str = "OCR Extracted Text") -> None:
        self.status = status
        self.text = text
        self.called = False

    def extract(self, document: ValidatedInput, failed: UnifiedDocument) -> UnifiedDocument:
        self.called = True
        return UnifiedDocument(
            file_hash=document.metadata.sha256,
            extractor=self.name,
            markdown=self.text,
            pages=[
                UnifiedPage(
                    page_number=1,
                    blocks=[UnifiedBlock(id="p1-b1", type="text", text=self.text, reading_order=1)],
                )
            ],
            diagnostics=ExtractionDiagnostics(
                status=self.status,
                issues=[],
                text_character_count=len(self.text),
                block_count=1,
                heading_count=1,
            ),
        )


def test_routing_main_extractor_usable_skips_fallback() -> None:
    inp = _make_validated_input()
    main = FakeMainExtractor(status=DiagnosticStatus.USABLE, text="Detailed CV content")
    fallback = FakeOCRFallbackExtractor(status=DiagnosticStatus.USABLE, text="OCR content")

    res = extract_with_fallback(inp, main, fallback)
    assert res.extractor == "fake_main"
    assert main.called is True
    assert fallback.called is False


def test_routing_main_extractor_fallback_required_triggers_ocr() -> None:
    inp = _make_validated_input()
    main = FakeMainExtractor(status=DiagnosticStatus.FALLBACK_REQUIRED, text="Short")
    fallback = FakeOCRFallbackExtractor(
        status=DiagnosticStatus.USABLE, text="OCR Recovered Detailed CV Content"
    )

    res = extract_with_fallback(inp, main, fallback)
    assert res.extractor == "fake_rapidocr_fallback"
    assert main.called is True
    assert fallback.called is True
    assert res.diagnostics.status == DiagnosticStatus.USABLE


def test_image_placeholder_diagnostics_trigger_ocr_fallback() -> None:
    inp = _make_validated_input()

    class ImageOnlyMain:
        name = "docling"

        def extract(self, document: ValidatedInput) -> UnifiedDocument:
            from ai_core.parsers.unified import unified_from_lines

            return unified_from_lines(
                document,
                self.name,
                [["<!-- image -->"] for _ in range(8)],
                markdown="\n".join("<!-- image -->" for _ in range(8)),
            )

    fallback = FakeOCRFallbackExtractor(
        DiagnosticStatus.USABLE,
        text="Candidate has Python and SQL experience in production systems.",
    )
    res = extract_with_fallback(inp, ImageOnlyMain(), fallback)

    assert fallback.called is True
    assert res.extractor == fallback.name


def test_routing_ocr_fallback_still_fails_results_in_manual_review() -> None:
    inp = _make_validated_input()
    main = FakeMainExtractor(status=DiagnosticStatus.FALLBACK_REQUIRED, text="Short")
    fallback = FakeOCRFallbackExtractor(
        status=DiagnosticStatus.FALLBACK_REQUIRED, text="OCR Still Bad"
    )

    res = extract_with_fallback(inp, main, fallback)
    assert res.extractor == "fake_rapidocr_fallback"
    assert res.diagnostics.status == DiagnosticStatus.FALLBACK_REQUIRED

    # Verify reconciliation sets manual_review and quality score suppresses output
    dummy_profile = CVProfile(
        candidate_name="Alex Morgan",
        field_evidence={
            "candidateName": [EvidenceRef(text="Alex Morgan", page_number=1, block_id="p1-b1")]
        },
    )
    reconciled = reconcile_profile(dummy_profile, res, provider_name="beeknoee-openai-compatible")
    assert reconciled.validation_status == ValidationStatus.MANUAL_REVIEW

    score_result = score_cv_quality(reconciled)
    assert score_result.score is None
    assert score_result.confidence == 0.25


def test_pipeline_does_not_call_llm_after_second_diagnostics_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_core.pipeline import process_document

    source = tmp_path / "synthetic-scan.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    main = FakeMainExtractor(DiagnosticStatus.FALLBACK_REQUIRED, text="Short")
    fallback = FakeOCRFallbackExtractor(DiagnosticStatus.FALLBACK_REQUIRED, text="Still bad")

    class Provider:
        name = "must-not-be-called"
        model = "fixture"
        revision = "1"

        def extract(self, request: object) -> CVProfile:
            raise AssertionError("LLM extraction must not run after failed OCR diagnostics")

    monkeypatch.setattr("ai_core.pipeline.configured_main_extractor", lambda: main)
    monkeypatch.setattr("ai_core.pipeline.configured_fallback_extractor", lambda: fallback)

    result = process_document(source, extraction_provider=Provider())

    assert main.called is True
    assert fallback.called is True
    assert result.profile is not None
    assert result.profile.validation_status == ValidationStatus.MANUAL_REVIEW
    quality = next(score for score in result.scores if score.name == "cv_quality.v2")
    assert quality.status.value == "insufficient_data"
    assert quality.score is None
