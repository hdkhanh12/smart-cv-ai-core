"""Batch runner to process all CV files in 7_samples and output summary table."""

import sys
import time
import json
from pathlib import Path

# Force UTF-8 output encoding for Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ai_core.pipeline import process_document
from ai_core.embeddings import BgeM3Embedder, build_profile_text
from ai_core.extraction.provider import configured_provider
from ai_core.api.response_contract import (
    to_embedded_payload,
    to_extracted_payload,
    to_scored_payload,
)


def main():
    samples_dir = Path(r"..\7_samples")
    if len(sys.argv) > 1:
        samples_dir = Path(sys.argv[1])

    if not samples_dir.exists():
        print(f"[ERROR] Directory not found: {samples_dir}")
        sys.exit(1)

    pdf_files = sorted([f for f in samples_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
    if not pdf_files:
        print(f"[WARN] No PDF files found in: {samples_dir}")
        sys.exit(0)

    print(f"\n" + "=" * 75)
    print(f"🚀 BATCH CV PROCESSING: {len(pdf_files)} files in {samples_dir.resolve()}")
    print("=" * 75)

    embedder = BgeM3Embedder()
    provider = configured_provider()

    summary_records = []

    for idx, cv_path in enumerate(pdf_files, 1):
        stem_name = cv_path.stem
        out_dir = Path("outputs") / f"sample_{stem_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(pdf_files)}] 📄 Processing: {cv_path.name}")
        start_time = time.perf_counter()

        try:
            result = process_document(
                cv_path,
                extraction_provider=provider,
                local_evidence_binding="exact-entity",
                embed=True,
                embedder=embedder,
            )
            elapsed = time.perf_counter() - start_time

            # Save stage 1: Parsed text
            if result.unified_document:
                (out_dir / "step1_parsed_document.txt").write_text(
                    result.unified_document.markdown, encoding="utf-8"
                )

            # Save stage 2: Canonical profile
            if result.profile:
                (out_dir / "step2_canonical_profile.json").write_text(
                    result.profile.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
                )

            # Save stage 3: Extracted payload
            extracted_data = to_extracted_payload(result)
            (out_dir / "step3_extracted_payload.json").write_text(
                json.dumps(extracted_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Save stage 4: Scored payload
            scored_data = to_scored_payload(result)
            (out_dir / "step4_scored_payload.json").write_text(
                json.dumps(scored_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Save stage 5: Profile text
            if result.profile:
                profile_text = build_profile_text(result.profile)
                (out_dir / "step5_embedding_profile_text.txt").write_text(
                    profile_text, encoding="utf-8"
                )

            # Save stage 6: Embedded payload
            embedded_data = to_embedded_payload(result)
            (out_dir / "step6_embedded_payload.json").write_text(
                json.dumps(embedded_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Extract summary stats
            candidate_name = extracted_data.get("candidateName") or "N/A"
            yoe = extracted_data.get("yearsOfExperience", 0.0)
            skills_count = len(extracted_data.get("skills", []))
            companies_count = len(extracted_data.get("companies", []))
            criteria = scored_data.get("criteriaScores", {})
            edu_score = criteria.get("EDUCATION", {}).get("score", 0.0)
            exp_score = criteria.get("EXPERIENCE_YEARS", {}).get("score", 0.0)
            skill_score = criteria.get("TECHNICAL_SKILLS", {}).get("score", 0.0)

            print(f"    ✅ Done in {elapsed:.1f}s | Name: {candidate_name} | YoE: {yoe} | Skills: {skills_count} | Companies: {companies_count}")
            summary_records.append({
                "file": cv_path.name,
                "name": candidate_name,
                "yoe": f"{yoe:.1f}y",
                "skills": skills_count,
                "companies": companies_count,
                "edu_score": edu_score,
                "exp_score": exp_score,
                "skill_score": skill_score,
                "time": f"{elapsed:.1f}s",
                "status": "SUCCESS",
            })

        except Exception as err:
            elapsed = time.perf_counter() - start_time
            print(f"    ❌ Failed in {elapsed:.1f}s: {err}")
            summary_records.append({
                "file": cv_path.name,
                "name": "N/A",
                "yoe": "N/A",
                "skills": 0,
                "companies": 0,
                "edu_score": 0,
                "exp_score": 0,
                "skill_score": 0,
                "time": f"{elapsed:.1f}s",
                "status": f"FAILED ({err})",
            })

    print("\n" + "=" * 105)
    print("📊 BATCH PROCESSING SUMMARY TABLE")
    print("=" * 105)
    header = f"{'STT':<4} | {'File Name':<35} | {'Candidate Name':<22} | {'YoE':<6} | {'Skills':<6} | {'Time':<6} | {'Status'}"
    print(header)
    print("-" * 105)
    for i, r in enumerate(summary_records, 1):
        print(f"{i:<4} | {r['file'][:35]:<35} | {r['name'][:22]:<22} | {r['yoe']:<6} | {r['skills']:<6} | {r['time']:<6} | {r['status']}")
    print("=" * 105)
    print(f"✨ All detailed 7-stage outputs saved to: outputs/sample_<filename>/\n")


if __name__ == "__main__":
    main()
