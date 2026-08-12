"""Stage-by-stage inspection tool for JD and Search Query processing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_core.embeddings.bge_m3 import BgeM3Embedder
from ai_core.search import _prune_jd_noise, analyze_query_or_jd, build_query_text
from ai_core.schemas.query import SearchQueryFilters
from ai_core.taxonomy import JOB_TITLES, SKILLS_DICT, find_entities


def inspect_stages(text: str) -> None:
    print("==========================================================================")
    print("             SMART-CV-AI-CORE : JD & SEARCH QUERY INSPECTOR               ")
    print("==========================================================================")
    print(f"\n[STAGE 1] RAW INPUT (len={len(text)} chars):\n{text[:300]!r}...")

    # Stage 2: Normalization & Noise Pruning
    clean_text = _prune_jd_noise(text)
    print(f"\n[STAGE 2] SANITIZED & PRUNED TEXT (len={len(clean_text)} chars):\n{clean_text[:300]!r}...")

    # Stage 3: Taxonomy Matching
    skills = list(find_entities(clean_text, SKILLS_DICT).keys())
    job_titles = list(find_entities(clean_text, JOB_TITLES).keys())
    print(f"\n[STAGE 3] FAST TAXONOMY MATCH (FlashText):")
    print(f"  - Extracted Skills:     {skills}")
    print(f"  - Extracted JobTitles:  {job_titles}")

    # Stage 4: Canonical Representation
    filters = SearchQueryFilters(skills=skills, job_titles=job_titles)
    canonical_text = build_query_text(filters, clean_text)
    print(f"\n[STAGE 4] CANONICAL QUERY TEXT (BGE-M3 Input Formatter):\n{canonical_text}")

    # Stage 5: BGE-M3 Embedding Vector
    print("\n[STAGE 5] GENERATING BGE-M3 1024D EMBEDDING VECTOR...")
    embedder = BgeM3Embedder()
    res = embedder.embed_text(canonical_text)
    print(f"  - Model:          {res.model}")
    print(f"  - Revision:       {res.model_revision}")
    print(f"  - Dimension:      {res.dimension}")
    print(f"  - L2-Normalized:  {res.normalized}")
    print(f"  - Duration:       {res.duration_ms:.2f} ms")
    print(f"  - Vector Hash:    {res.source_hash[:16]}...")
    print(f"  - Vector Sample:  [{res.vector[0]:.4f}, {res.vector[1]:.4f}, {res.vector[2]:.4f}, ..., {res.vector[-1]:.4f}]")
    print("==========================================================================")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Inspect JD/Query processing stages.")

    parser.add_argument("--text", type=str, help="Search query string or JD snippet.")
    parser.add_argument("--file", type=Path, help="Path to a .txt or .pdf JD file.")

    args = parser.parse_args()
    if args.file and args.file.is_file():
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = "Tuyển Backend Developer 3 năm kinh nghiệm Python, FastAPI, Docker, PostgreSQL tại Hà Nội"

    inspect_stages(text)


if __name__ == "__main__":
    main()
