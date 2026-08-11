"""Local, versioned cache for deterministic score results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ai_core.schemas import ScoreResult, ValidationStatus
from ai_core.scoring.features import FEATURE_VERSION

SCORE_CACHE_VERSION = "score-cache-v1"


def cache_key(
    source_hash: str,
    *,
    anonymize: bool,
    feature_hash: str,
    validation_status: ValidationStatus,
) -> str:
    payload = {
        "version": SCORE_CACHE_VERSION,
        "featureVersion": FEATURE_VERSION,
        "sourceHash": source_hash,
        "mode": "anonymized" if anonymize else "fresh",
        "featureHash": feature_hash,
        "validationStatus": validation_status.value,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_scores(path: Path) -> list[ScoreResult] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return None
        return [ScoreResult.model_validate(item) for item in payload]
    except (OSError, ValueError):
        return None


def store_scores(path: Path, scores: list[ScoreResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([score.model_dump(mode="json") for score in scores], ensure_ascii=False),
        encoding="utf-8",
    )
