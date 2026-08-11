"""Contract tests for the narrow, PII-safe embedding bridge."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from ai_core.api.server import create_app
from ai_core.embeddings import GteMultilingualEmbedder
from ai_core.schemas import (
    CVProfile,
    EmbeddingResult,
    Experience,
    ProcessingResult,
    ProcessingStatus,
    ValidationStatus,
)


class FakeEncoder:
    def encode(self, sentences: Sequence[str], **kwargs: object) -> list[list[float]]:
        assert len(sentences) == 1
        return [[1.0] * 768]


def _embedder() -> GteMultilingualEmbedder:
    return GteMultilingualEmbedder(encoder=FakeEncoder())


def _processing_result(*, manual_review: bool = False) -> ProcessingResult:
    profile = CVProfile(
        candidate_name="Synthetic Candidate",
        email="synthetic@example.invalid",
        phone="0900000000",
        headline="Backend Engineer",
        summary="Builds reliable APIs.",
        experiences=[
            Experience(
                job_title="Backend Engineer",
                company="Example Systems",
                description=["Built Python APIs."],
                skills=["Python", "PostgreSQL"],
            )
        ],
        validation_status=(
            ValidationStatus.MANUAL_REVIEW if manual_review else ValidationStatus.PARTIAL
        ),
    )
    return ProcessingResult(
        status=ProcessingStatus.SUCCEEDED,
        source_id="0" * 64,
        profile=profile,
        embedding=(
            None
            if manual_review
            else EmbeddingResult(
                model="Alibaba-NLP/gte-multilingual-base",
                model_revision="main",
                dimension=768,
                normalized=True,
                vector=[1 / math.sqrt(768)] * 768,
                source_hash="1" * 64,
                template_version="gte-profile-v1",
                duration_ms=1,
            )
        ),
        audit={
            "rawOcrText": "must never leave the API",
            "tokens": {"documentAfter": 101, "completionTokens": 22},
        },
    )


def test_text_endpoint_returns_normalized_768d_embedding() -> None:
    client = TestClient(create_app(embedder=_embedder()))

    response = client.post("/v1/embeddings/text", files={"input": (None, "Python engineer")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["dimension"] == 768
    assert len(payload["embedding"]) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in payload["embedding"])), 1.0)


def test_openapi_exposes_exactly_four_public_api_routes() -> None:
    client = TestClient(create_app(embedder=_embedder()))

    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {
        "/v1/cv/process",
        "/v1/embeddings/file",
        "/v1/embeddings/profile",
        "/v1/embeddings/text",
    }


def test_text_endpoint_rejects_empty_input() -> None:
    client = TestClient(create_app(embedder=_embedder()))

    assert client.post("/v1/embeddings/text", files={"input": (None, " ")}).status_code == 422


def test_cv_process_returns_rich_safe_profile_and_initial_embedding() -> None:
    calls = 0

    def fake_process(path: Path, **kwargs: object) -> ProcessingResult:
        nonlocal calls
        calls += 1
        assert path.suffix == ".pdf"
        assert kwargs["embed"] is True
        assert kwargs["embedder"] is not None
        return _processing_result()

    client = TestClient(create_app(embedder=_embedder(), process=fake_process))
    response = client.post(
        "/v1/cv/process",
        files={"input": ("synthetic.pdf", b"%PDF-1.4\nsynthetic\n%%EOF", "application/pdf")},
    )

    assert response.status_code == 200
    assert calls == 1
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["profile"]["candidateName"] == "Synthetic Candidate"
    assert payload["profile"]["experiences"][0]["company"] == "Example Systems"
    assert payload["meta"]["inputTokenEstimate"] == 101
    assert payload["embedding"]["dimension"] == 768
    assert len(payload["embedding"]["embedding"]) == 768
    assert payload["embedding"]["model"] == "Alibaba-NLP/gte-multilingual-base"
    assert "rawOcrText" not in response.text
    assert "fieldEvidence" not in response.text


def test_cv_process_rejects_manual_review_without_leaking_profile() -> None:
    def fake_process(path: Path, **kwargs: object) -> ProcessingResult:
        assert kwargs["embed"] is True
        return _processing_result(manual_review=True)

    client = TestClient(create_app(embedder=_embedder(), process=fake_process))
    response = client.post(
        "/v1/cv/process",
        files={"input": ("synthetic.pdf", b"%PDF-1.4\nsynthetic\n%%EOF", "application/pdf")},
    )

    assert response.status_code == 422
    assert "Synthetic Candidate" not in response.text


def test_profile_embedding_uses_stored_profile_without_processing_file() -> None:
    def must_not_process(*args: object, **kwargs: object) -> ProcessingResult:
        raise AssertionError("profile embedding must not call the extraction pipeline")

    profile = _processing_result().profile
    assert profile is not None
    client = TestClient(create_app(embedder=_embedder(), process=must_not_process))
    response = client.post(
        "/v1/embeddings/profile",
        json={"profile": profile.model_dump(by_alias=True, mode="json")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dimension"] == 768
    assert len(payload["embedding"]) == 768
    assert payload["model"] == "Alibaba-NLP/gte-multilingual-base"
    assert payload["templateVersion"] == "gte-profile-v2"
    assert len(payload["semanticProfileHash"]) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in payload["embedding"])), 1.0)


def test_profile_embedding_hash_excludes_contact_information() -> None:
    first = _processing_result().profile
    assert first is not None
    second = first.model_copy(update={"email": "other@example.invalid", "phone": "0911111111"})
    client = TestClient(create_app(embedder=_embedder()))

    first_response = client.post(
        "/v1/embeddings/profile",
        json={"profile": first.model_dump(by_alias=True, mode="json")},
    )
    second_response = client.post(
        "/v1/embeddings/profile",
        json={"profile": second.model_dump(by_alias=True, mode="json")},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_hash = first_response.json()["semanticProfileHash"]
    second_hash = second_response.json()["semanticProfileHash"]
    assert first_hash == second_hash


def test_file_endpoint_returns_only_embedding_contract(tmp_path: Path) -> None:
    calls = 0

    def fake_process(path: Path, **kwargs: object) -> ProcessingResult:
        nonlocal calls
        calls += 1
        assert path.suffix == ".pdf"
        assert kwargs["embed"] is True
        return _processing_result()

    client = TestClient(create_app(embedder=_embedder(), process=fake_process))
    response = client.post(
        "/v1/embeddings/file",
        files={"input": ("synthetic.pdf", b"%PDF-1.4\nsynthetic\n%%EOF", "application/pdf")},
    )

    assert response.status_code == 200
    assert calls == 1
    payload = response.json()
    assert set(payload) == {"embedding", "dimension"}
    assert len(payload["embedding"]) == 768
    assert "Synthetic Candidate" not in response.text
    assert "rawOcrText" not in response.text


def test_file_endpoint_rejects_missing_or_unsupported_input() -> None:
    client = TestClient(create_app(embedder=_embedder()))

    assert client.post("/v1/embeddings/file").status_code == 422
    response = client.post(
        "/v1/embeddings/file",
        files={"input": ("notes.txt", b"not a CV", "text/plain")},
    )
    assert response.status_code == 422
    assert "notes.txt" not in response.text
    assert "not a CV" not in response.text


def test_file_endpoint_rejects_manual_review_without_embedding() -> None:
    def fake_process(path: Path, **kwargs: object) -> ProcessingResult:
        return _processing_result(manual_review=True)

    client = TestClient(create_app(embedder=_embedder(), process=fake_process))
    response = client.post(
        "/v1/embeddings/file",
        files={"input": ("synthetic.pdf", b"%PDF-1.4\nsynthetic\n%%EOF", "application/pdf")},
    )

    assert response.status_code == 422
    assert "Synthetic Candidate" not in response.text
