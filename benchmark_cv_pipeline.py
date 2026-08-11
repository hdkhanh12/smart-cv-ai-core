"""Diagnostic & Benchmark Suite for AI Core CV Extraction Workflow across all 7 sample CVs."""

import sys
import time
import json
from pathlib import Path

# Force UTF-8 output encoding for Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ai_core.pipeline import process_document
from ai_core.extraction.provider import CompatibilityFullDocumentProvider
from ai_core.embeddings import BgeM3Embedder
from ai_core.api.response_contract import (
    to_extracted_payload,
    to_scored_payload,
    to_embedded_payload,
)

def run_benchmark():
    samples_dir = Path(r"C:\Users\Thinkpad\Desktop\smart-cv\7_samples")
    pdf_files = sorted(list(samples_dir.glob("*.pdf")))

    out_dir = Path("outputs") / "benchmark_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print(f"[BENCHMARK] RUNNING DIAGNOSTIC SUITE ON {len(pdf_files)} SAMPLE CVs")
    print(f"[MODE] Resilient Pipeline Provider (Fast & Deterministic Analysis)")
    print("============================================================")

    embedder = BgeM3Embedder()
    results_summary = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        start_time = time.perf_counter()

        try:
            provider = CompatibilityFullDocumentProvider()
            res = process_document(
                pdf_path,
                extraction_provider=provider,
                local_evidence_binding="exact-entity",
                embed=True,
                embedder=embedder,
            )
            elapsed = time.perf_counter() - start_time

            extracted = to_extracted_payload(res)
            scored = to_scored_payload(res)
            embedded = to_embedded_payload(res)

            issues_list = [w.message for w in res.warnings] + [e.message for e in res.errors]

            report_entry = {
                "fileName": pdf_path.name,
                "elapsedSeconds": round(elapsed, 2),
                "extracted": {
                    "candidateName": extracted.get("candidateName"),
                    "email": extracted.get("email"),
                    "phone": extracted.get("phone"),
                    "address": extracted.get("address"),
                    "highestEducation": extracted.get("highestEducation"),
                    "yearsOfExperience": extracted.get("yearsOfExperience"),
                    "companies": extracted.get("companies", []),
                    "jobTitles": extracted.get("jobTitles", []),
                    "skillsCount": len(extracted.get("skills", [])),
                },
                "scored": {
                    "totalScore": scored.get("Score"),
                    "scoringDetails": scored.get("ScoringDetails", []),
                },
                "embedded": {
                    "status": embedded.get("status"),
                    "dimension": embedded.get("dimension"),
                },
                "issues": issues_list,
            }
            results_summary.append(report_entry)

            print(f"\n[{idx}/{len(pdf_files)}] File: {pdf_path.name}")
            print(f"   ⏱️ Time: {elapsed:.2f}s | Name: {extracted.get('candidateName')} | Exp Years: {extracted.get('yearsOfExperience')} | Score: {scored.get('Score')}/100")
            print(f"   🏢 Companies ({len(extracted.get('companies', []))}): {extracted.get('companies')}")
            print(f"   🎓 Education: {extracted.get('highestEducation')}")
            print(f"   🛠️ Skills ({len(extracted.get('skills', []))}): {[s['skillName'] for s in extracted.get('skills', [])[:8]]}...")

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            print(f"   ❌ Failed to process {pdf_path.name}: {exc}")
            results_summary.append({
                "fileName": pdf_path.name,
                "elapsedSeconds": round(elapsed, 2),
                "error": str(exc),
            })

    report_file = out_dir / "benchmark_summary.json"
    report_file.write_text(json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n============================================================")
    print(f"[SUCCESS] BENCHMARK COMPLETE! Full report saved to: {report_file.resolve()}")
    print("============================================================")

if __name__ == "__main__":
    run_benchmark()
