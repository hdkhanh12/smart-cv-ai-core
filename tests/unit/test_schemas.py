from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from ai_core.errors import ErrorCode, Issue
from ai_core.schemas import (
    CVProfile,
    CVSection,
    DocumentMetadata,
    Education,
    EmbeddingResult,
    Experience,
    MatchResult,
    PageContent,
    ParsedDocument,
    ProcessingResult,
    ProcessingStatus,
    ScoreResult,
    SectionKind,
    Skill,
)

SHA256 = "a" * 64


def metadata() -> DocumentMetadata:
    return DocumentMetadata(
        source_name="synthetic.pdf",
        extension=".pdf",
        media_type="application/pdf",
        size_bytes=128,
        sha256=SHA256,
        page_count=1,
    )


def test_processing_result_json_round_trip_uses_camel_case() -> None:
    profile = CVProfile(
        candidate_name="Nguyễn Văn An",
        email="an@example.test",
        urls=["https://github.com/example"],
        skills=[
            Skill(
                name="Python",
                canonical_name="Python",
                evidence=["Built an API"],
                section=SectionKind.SKILLS,
            )
        ],
        experiences=[
            Experience(
                job_title="Software Engineer",
                company="Example",
                start_date=date(2022, 1, 1),
                is_current=True,
            )
        ],
        education=[Education(institution="Example University", degree="BSc")],
        field_confidence={"candidateName": 0.9},
    )
    parsed = ParsedDocument(
        metadata=metadata(),
        pages=[PageContent(page_number=1, text="Nguyễn Văn An")],
        raw_text="Nguyễn Văn An",
        normalized_text="Nguyễn Văn An",
    )
    score = ScoreResult(name="completeness", score=85, confidence=0.9)
    result = ProcessingResult(
        status=ProcessingStatus.SUCCEEDED,
        source_id=SHA256,
        metadata=metadata(),
        parsed_document=parsed,
        profile=profile,
        embedding=EmbeddingResult(
            model="BAAI/bge-m3",
            model_revision="test-revision",
            dimension=3,
            normalized=True,
            vector=[1.0, 0.0, 0.0],
            source_hash=SHA256,
            template_version="1",
            duration_ms=1.2,
        ),
        scores=[score],
    )

    payload = result.model_dump_json(by_alias=True)
    restored = ProcessingResult.model_validate_json(payload)

    assert '"schemaVersion":"1.0.0"' in payload
    assert '"candidateName":"Nguyễn Văn An"' in payload
    assert restored == result


def test_match_result_round_trip() -> None:
    match = MatchResult(
        candidate_id="candidate-1",
        rank=1,
        raw_similarity=0.72,
        score=ScoreResult(name="hybrid", score=78, confidence=0.8),
        matched_requirements=["Python"],
        missing_requirements=["Kubernetes"],
    )
    assert MatchResult.model_validate_json(match.model_dump_json(by_alias=True)) == match


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            lambda: DocumentMetadata(
                source_name="x.pdf",
                extension=".pdf",
                media_type="application/pdf",
                size_bytes=0,
                sha256=SHA256,
            ),
            "greater than 0",
        ),
        (
            lambda: ParsedDocument(
                metadata=metadata(),
                pages=[
                    PageContent(page_number=2, text="b"),
                    PageContent(page_number=1, text="a"),
                ],
            ),
            "unique ascending",
        ),
        (
            lambda: CVSection(
                kind=SectionKind.SKILLS,
                text="Python",
                start_line=3,
                end_line=2,
                confidence=0.8,
            ),
            "endLine",
        ),
        (
            lambda: Experience(
                job_title="Engineer",
                start_date=date(2024, 1, 1),
                end_date=date(2023, 1, 1),
            ),
            "endDate",
        ),
        (
            lambda: EmbeddingResult(
                model="model",
                model_revision="revision",
                dimension=2,
                normalized=True,
                vector=[1.0],
                source_hash=SHA256,
                template_version="1",
                duration_ms=0,
            ),
            "dimension",
        ),
        (
            lambda: ProcessingResult(
                status=ProcessingStatus.FAILED,
                source_id="source",
            ),
            "at least one error",
        ),
        (
            lambda: ProcessingResult(
                status=ProcessingStatus.SUCCEEDED,
                source_id="source",
                errors=[
                    Issue(
                        code=ErrorCode.PROCESSING_FAILED,
                        message="failed",
                        stage="test",
                    )
                ],
            ),
            "only failed",
        ),
    ],
)
def test_invalid_schema_data_is_rejected(factory: object, expected: str) -> None:
    with pytest.raises(ValidationError, match=expected):
        factory()  # type: ignore[operator]
