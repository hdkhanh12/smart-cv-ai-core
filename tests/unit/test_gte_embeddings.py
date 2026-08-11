from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from ai_core.embeddings import EMBEDDING_DIMENSION, GteMultilingualEmbedder, build_profile_text
from ai_core.extraction import CompatibilityFullDocumentProvider
from ai_core.pipeline import process_document
from ai_core.schemas import (
    CVProfile,
    DiagnosticStatus,
    EmbeddingResult,
    Experience,
    ExtractionDiagnostics,
    Skill,
    UnifiedBlock,
    UnifiedDocument,
    UnifiedPage,
)


class FakeEncoder:
    def encode(self, sentences: Sequence[str], **kwargs: object) -> list[list[float]]:
        assert sentences
        assert kwargs["normalize_embeddings"] is False
        return [[float(index + 1) for index in range(EMBEDDING_DIMENSION)]]


def test_profile_text_is_stable_and_excludes_pii() -> None:
    profile = CVProfile(
        candidate_name="Candidate Name",
        email="candidate@example.com",
        phone="0123456789",
        headline="Data Engineer",
        skills=[Skill(name="Python", canonical_name="Python")],
        experiences=[
            Experience(
                job_title="Engineer",
                company="Example",
                description=["Built data pipelines"],
            )
        ],
    )

    text = build_profile_text(profile)

    assert "[SKILLS]" in text
    assert "Built data pipelines" in text
    assert "candidate@example.com" not in text
    assert "Candidate Name" not in text


def test_gte_embedder_returns_normalized_vector(tmp_path: Path) -> None:
    embedder = GteMultilingualEmbedder(cache_dir=tmp_path, encoder=FakeEncoder())
    result = embedder.embed_text("Python data")

    assert result.dimension == EMBEDDING_DIMENSION
    assert result.normalized is True
    assert len(result.vector) == EMBEDDING_DIMENSION
    assert math.isclose(sum(value * value for value in result.vector), 1.0, rel_tol=1e-9)
    assert result.duration_ms >= 0


def test_pipeline_records_embedding_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")

    def usable_unified(*args: object, **kwargs: object) -> UnifiedDocument:
        del args, kwargs
        return UnifiedDocument(
            file_hash="b" * 64,
            extractor="fixture",
            markdown="Candidate profile with Python experience and evidence.",
            pages=[
                UnifiedPage(
                    page_number=1,
                    blocks=[UnifiedBlock(id="p1-b1", text="Python", reading_order=1)],
                )
            ],
            diagnostics=ExtractionDiagnostics(
                status=DiagnosticStatus.USABLE,
                text_character_count=53,
                block_count=1,
                heading_count=0,
            ),
        )

    monkeypatch.setattr("ai_core.pipeline.extract_with_fallback", usable_unified)

    class FakeProfileEmbedder:
        def embed_profile(self, profile: CVProfile) -> EmbeddingResult:
            del profile
            return EmbeddingResult(
                model="BAAI/bge-m3",
                model_revision="main",
                dimension=EMBEDDING_DIMENSION,
                normalized=True,
                vector=[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
                source_hash="a" * 64,
                template_version="bge-m3-profile-v1",
                duration_ms=3.2,
            )

    result = process_document(
        source,
        extraction_provider=CompatibilityFullDocumentProvider(),
        embed=True,
        embedder=FakeProfileEmbedder(),  # type: ignore[arg-type]
    )

    assert result.embedding is not None
    assert result.audit["scoreFeatures"]["version"] == "document-score-v1"
    assert len(result.audit["scoreFeatures"]["hash"]) == 64
    assert result.audit["embedding"] == {
        "model": "BAAI/bge-m3",
        "dimension": EMBEDDING_DIMENSION,
        "latencyMs": 3.2,
        "sourceHash": "a" * 64,
        "timestamp": result.audit["embedding"]["timestamp"],
        "operation": "profile_embedding",
    }
