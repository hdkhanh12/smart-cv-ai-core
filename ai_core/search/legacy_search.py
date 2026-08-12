"""Local dense search over persisted ProcessingResult JSON artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from ai_core.embeddings import GteMultilingualEmbedder
from ai_core.schemas import ProcessingResult


@dataclass(frozen=True)
class SearchMatch:
    rank: int
    source: Path
    candidate_name: str
    similarity: float
    top_skills: list[str]


def read_jd(path: Path) -> str:
    """Read a UTF-8 text JD or extract its text from a PDF."""

    if not path.is_file():
        raise ValueError(f"JD file does not exist: {path}")
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as document:
            text = "\n".join(page.get_text() for page in document)
    elif path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError("JD must be a .txt or .pdf file.")
    if not text.strip():
        raise ValueError("JD contains no extractable text.")
    return text


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match.")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        raise ValueError("Cannot calculate cosine similarity for a zero vector.")
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def search_results(
    jd_text: str,
    cv_dir: Path,
    *,
    top_k: int,
    embedder: GteMultilingualEmbedder | None = None,
) -> list[SearchMatch]:
    """Rank valid output JSON files; fill and persist missing CV embeddings."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not cv_dir.is_dir():
        raise ValueError(f"CV directory does not exist: {cv_dir}")
    active_embedder = embedder or GteMultilingualEmbedder()
    jd_embedding = active_embedder.embed_text(jd_text)
    matches: list[SearchMatch] = []
    for json_path in sorted(cv_dir.rglob("*.json")):
        try:
            result = ProcessingResult.model_validate_json(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if result.profile is None:
            continue
        embedding = result.embedding or active_embedder.embed_profile(result.profile)
        if result.embedding is None:
            result.embedding = embedding
            result.audit["embedding"] = {
                "model": embedding.model,
                "dimension": embedding.dimension,
                "latencyMs": embedding.duration_ms,
                "sourceHash": embedding.source_hash,
                "operation": "search_backfill",
            }
            json_path.write_text(
                result.model_dump_json(by_alias=True, indent=2) + "\n",
                encoding="utf-8",
            )
        name = result.profile.candidate_name or "Unknown candidate"
        skills = [skill.canonical_name for skill in result.profile.skills[:5]]
        matches.append(
            SearchMatch(0, json_path, name, _cosine(jd_embedding.vector, embedding.vector), skills)
        )
    matches.sort(key=lambda match: match.similarity, reverse=True)
    return [
        SearchMatch(index, match.source, match.candidate_name, match.similarity, match.top_skills)
        for index, match in enumerate(matches[:top_k], start=1)
    ]


def write_search_report(matches: list[SearchMatch], jd_path: Path, report_path: Path) -> None:
    """Write a Markdown result report with standard file URIs for all artifacts."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Dense Search Report",
        "",
        f"- Job description: [{jd_path.name}]({jd_path.resolve().as_uri()})",
        "- Model: `Alibaba-NLP/gte-multilingual-base` (768d, L2-normalized)",
        "",
        "| Rank | Candidate Name | File Name | Cosine Similarity Score | Top Skills |",
        "|---:|---|---|---:|---|",
    ]
    rows.extend(
        f"| {match.rank} | {match.candidate_name} | "
        f"[{match.source.name}]({match.source.resolve().as_uri()}) | "
        f"{match.similarity * 100:.2f}% | {', '.join(match.top_skills) or '-'} |"
        for match in matches
    )
    if not matches:
        rows.append("| - | No valid candidate artifacts found | - | - | - |")
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
