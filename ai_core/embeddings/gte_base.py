"""CPU-local GTE multilingual embeddings for candidate profiles and JDs."""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from ai_core.schemas import CVProfile, EmbeddingResult

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
EMBEDDING_DIMENSION = 768
TEMPLATE_VERSION = "gte-profile-v2"
# This is the local Hugging Face snapshot used by the checked-in runtime cache.
# Deployments may override it explicitly, but must use the same revision for CV
# and query vectors in one semantic index.
MODEL_REVISION = os.getenv("SMART_CV_GTE_REVISION", "9bbca17d9273fd0d03d5725c7a4b0f6b45142062")


class SentenceEncoder(Protocol):
    """Minimal surface used from sentence-transformers, enabling offline tests."""

    def encode(self, sentences: Sequence[str], **kwargs: object) -> object: ...


def _nonempty(values: Sequence[str | None]) -> list[str]:
    return [value.strip() for value in values if value and value.strip()]


def build_profile_text(profile: CVProfile) -> str:
    """Create a stable, non-PII retrieval representation of a profile."""

    lines = ["[CANDIDATE_PROFILE]"]
    if profile.headline:
        lines.append(f"Target roles: {profile.headline.strip()}")
    if profile.summary:
        lines.append(f"Professional summary: {profile.summary.strip()}")
    skills = _nonempty([skill.canonical_name for skill in profile.skills])
    if skills:
        lines.extend(["", "[SKILLS]", "; ".join(dict.fromkeys(skills))])
    if profile.experiences:
        lines.extend(["", "[WORK_EXPERIENCE]"])
        for experience in profile.experiences:
            employer = f" at {experience.company}" if experience.company else ""
            lines.append(f"{experience.job_title}{employer}")
            lines.extend(f"- {item}" for item in _nonempty(experience.description))
            lines.extend(f"- {item}" for item in _nonempty(experience.achievements))
            if experience.skills:
                lines.append(f"- Skills: {'; '.join(_nonempty(experience.skills))}")
    if profile.projects:
        lines.extend(["", "[PROJECTS]"])
        for project in profile.projects:
            lines.append(project.title)
            lines.extend(f"- {item}" for item in _nonempty(project.description))
            if project.skills:
                lines.append(f"- Skills: {'; '.join(_nonempty(project.skills))}")
    if profile.education:
        lines.extend(["", "[EDUCATION]"])
        for education in profile.education:
            degree = education.degree or education.field_of_study or "Education"
            lines.append(f"{degree} at {education.institution}")
            if education.coursework:
                lines.append(f"- Relevant coursework: {'; '.join(_nonempty(education.coursework))}")
    if profile.languages:
        lines.extend(["", "[LANGUAGES]", "; ".join(_nonempty(profile.languages))])
    if profile.language_proficiencies:
        details = []
        for language in profile.language_proficiencies:
            value = language.language
            if language.proficiency:
                value += f" ({language.proficiency})"
            elif language.exam and language.score is not None:
                value += f" ({language.exam} {language.score:g})"
            details.append(value)
        lines.extend(["", "[LANGUAGE_PROFICIENCY]", "; ".join(details)])
    return "\n".join(lines)


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_profile_hash(profile: CVProfile) -> str:
    """Hash the canonical non-PII semantic representation of a profile."""

    return _source_hash(build_profile_text(profile))


def _normalize(vector: Sequence[float]) -> list[float]:
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding model returned a non-finite vector value.")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding model returned a zero vector.")
    return [float(value / norm) for value in vector]


class GteMultilingualEmbedder:
    """Lazy, process-cached local CPU embedder for Alibaba GTE multilingual base."""

    _cached_encoder: SentenceEncoder | None = None

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        encoder: SentenceEncoder | None = None,
    ) -> None:
        self.cache_dir = cache_dir or Path(os.getenv("SMART_CV_MODEL_CACHE", "outputs/model-cache"))
        self._encoder = encoder

    def _load_encoder(self) -> SentenceEncoder:
        if self._encoder is not None:
            return self._encoder
        if type(self)._cached_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment boundary
                raise RuntimeError(
                    "Install the embedding extra: pip install -e '.[embedding]'"
                ) from exc
            from huggingface_hub import snapshot_download

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            allow_download = os.getenv("SMART_CV_MODEL_ALLOW_DOWNLOAD", "").lower() in {
                "1",
                "true",
                "yes",
            }
            model_path = snapshot_download(
                MODEL_NAME,
                cache_dir=str(self.cache_dir),
                revision=MODEL_REVISION,
                local_files_only=not allow_download,
            )
            type(self)._cached_encoder = SentenceTransformer(
                model_path,
                cache_folder=str(self.cache_dir),
                device="cpu",
                trust_remote_code=True,
                revision=MODEL_REVISION,
                local_files_only=True,
            )
        cached = type(self)._cached_encoder
        assert cached is not None
        return cached

    def embed_text(self, text: str) -> EmbeddingResult:
        """Encode one text to a 768d L2-normalized vector with audit metadata."""

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Cannot embed empty text.")
        started = time.perf_counter()
        raw = self._load_encoder().encode(
            [normalized_text], convert_to_numpy=True, normalize_embeddings=False
        )
        row = raw[0]  # type: ignore[index]
        vector = _normalize([float(value) for value in row])
        duration_ms = (time.perf_counter() - started) * 1000
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions from {MODEL_NAME}, got {len(vector)}."
            )
        return EmbeddingResult(
            model=MODEL_NAME,
            model_revision=MODEL_REVISION,
            dimension=EMBEDDING_DIMENSION,
            normalized=True,
            vector=vector,
            source_hash=_source_hash(normalized_text),
            template_version=TEMPLATE_VERSION,
            duration_ms=round(duration_ms, 3),
        )

    def embed_profile(self, profile: CVProfile) -> EmbeddingResult:
        return self.embed_text(build_profile_text(profile))
