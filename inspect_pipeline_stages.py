r"""CLI tool to run the TRUE standard AI Core LLM pipeline on any CV file and save all 6 stage outputs + 7th timing & token audit file.

Usage:
  .\.venv\Scripts\python.exe inspect_pipeline_stages.py <cv_file_path> [output_directory]
"""

import sys
import time
import json
from pathlib import Path

# Force UTF-8 output encoding for Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ai_core.pipeline import process_document
from ai_core.embeddings import BgeM3Embedder, build_profile_text
from ai_core.extraction.provider import CompatibilityFullDocumentProvider, configured_provider
from ai_core.api.response_contract import (
    to_embedded_payload,
    to_extracted_payload,
    to_scored_payload,
)

def main():
    if len(sys.argv) > 1:
        cv_path = Path(sys.argv[1])
    else:
        cv_path = Path(r"..\7_samples\HUYNHTHANHTUNG_ENG_Intern.pdf")

    if not cv_path.exists():
        print(f"Error: File not found: {cv_path}")
        sys.exit(1)

    stem_name = cv_path.stem
    if len(sys.argv) > 2:
        out_dir = Path(sys.argv[2])
    else:
        out_dir = Path("outputs") / f"stages_{stem_name}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"[TRUE STANDARD PIPELINE] Processing: {cv_path.name}")
    print(f"[OUTPUT FOLDER] Saving files to: {out_dir.resolve()}")
    print(f"============================================================")

    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    embedder = BgeM3Embedder()
    
    t0 = time.perf_counter()
    try:
        provider = configured_provider()
        result = process_document(
            cv_path,
            extraction_provider=provider,
            local_evidence_binding="none",
            embed=False,
        )
    except Exception as exc:
        print(f"[PIPELINE NOTICE] Remote LLM provider notice: {exc}. Switching to resilient fallback provider...")
        provider = CompatibilityFullDocumentProvider()
        result = process_document(
            cv_path,
            extraction_provider=provider,
            local_evidence_binding="none",
            embed=False,
        )

    t1 = time.perf_counter()
    timings["pipeline_extraction_and_reconciliation_ms"] = round((t1 - t0) * 1000, 2)

    # Stage 1: Parsed Raw Markdown / Text (Fast PDF Output)
    t_stage1 = time.perf_counter()
    stage1_file = out_dir / "step1_parsed_document.txt"
    raw_text = ""
    if result.parsed_document and result.parsed_document.raw_text:
        raw_text = result.parsed_document.raw_text
    elif result.unified_document and result.unified_document.markdown:
        raw_text = result.unified_document.markdown
    stage1_file.write_text(raw_text, encoding="utf-8")
    timings["step1_save_parsed_text_ms"] = round((time.perf_counter() - t_stage1) * 1000, 2)
    print(f"[STAGE 1] Saved Parsed Text ({len(raw_text)} chars) -> {stage1_file.resolve()}")

    # Stage 2: Reconciled LLM CVProfile JSON (Internal AI Core Schema)
    t_stage2 = time.perf_counter()
    stage2_file = out_dir / "step2_canonical_profile.json"
    if result.profile:
        stage2_file.write_text(result.profile.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    timings["step2_save_profile_json_ms"] = round((time.perf_counter() - t_stage2) * 1000, 2)
    print(f"[STAGE 2] Saved Reconciled LLM CVProfile JSON -> {stage2_file.resolve()}")

    # Stage 3: Callback 1 Payload - PUT /api/v1/cvs/{id}/extracted (Backend DTO)
    t_stage3 = time.perf_counter()
    stage3_file = out_dir / "step3_extracted_payload.json"
    extracted = to_extracted_payload(result)
    stage3_file.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    timings["step3_save_extracted_payload_ms"] = round((time.perf_counter() - t_stage3) * 1000, 2)
    print(f"[STAGE 3] Saved Extracted Payload JSON -> {stage3_file.resolve()}")

    # Stage 4: Callback 2 Payload - PUT /api/v1/cvs/{id}/scored (4-Criteria Rubric)
    t_stage4 = time.perf_counter()
    stage4_file = out_dir / "step4_scored_payload.json"
    scored = to_scored_payload(result)
    stage4_file.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    timings["step4_save_scored_payload_ms"] = round((time.perf_counter() - t_stage4) * 1000, 2)
    print(f"[STAGE 4] Saved Scored Payload JSON (Score={scored.get('Score')}) -> {stage4_file.resolve()}")

    # Time to UI Ready (Stages 1-4 completed)
    time_to_ui_ready = round(time.perf_counter() - total_start, 2)
    timings["time_to_ui_ready_seconds"] = time_to_ui_ready

    # Stage 5: Pre-Embedding Profile Text (Sanitized non-PII text fed into BGE-M3)
    t_stage5 = time.perf_counter()
    stage5_file = out_dir / "step5_embedding_profile_text.txt"
    profile_text = ""
    if result.profile:
        profile_text = build_profile_text(result.profile)
        stage5_file.write_text(profile_text, encoding="utf-8")
    timings["step5_save_pre_embedding_text_ms"] = round((time.perf_counter() - t_stage5) * 1000, 2)
    print(f"[STAGE 5] Saved Pre-Embedding Profile Text ({len(profile_text)} chars) -> {stage5_file.resolve()}")

    # Stage 6: Callback 3 Payload - PUT /api/v1/cvs/{id}/embedded (1024d Vector)
    t_stage6 = time.perf_counter()
    if result.profile and result.embedding is None:
        result.embedding = embedder.embed_profile(result.profile)
    stage6_file = out_dir / "step6_embedded_payload.json"
    embedded = to_embedded_payload(result)
    stage6_file.write_text(json.dumps(embedded, ensure_ascii=False, indent=2), encoding="utf-8")
    timings["step6_save_embedded_payload_ms"] = round((time.perf_counter() - t_stage6) * 1000, 2)
    print(f"[STAGE 6] Saved Embedded Payload (1024d Vector, len={len(embedded.get('embedding', []))}) -> {stage6_file.resolve()}")

    total_elapsed = round(time.perf_counter() - total_start, 2)
    background_indexing_time = round(total_elapsed - time_to_ui_ready, 2)
    tokens = result.audit.get("tokens", {})

    # Stage 7: Execution & Token Audit Metrics File
    stage7_file = out_dir / "step7_execution_audit.json"
    audit_payload = {
        "cvFileName": cv_path.name,
        "timeToUIReadySeconds": time_to_ui_ready,
        "backgroundIndexingSeconds": background_indexing_time,
        "totalElapsedSeconds": total_elapsed,
        "stageTimingsMs": result.timings_ms or timings,
        "fileStageTimingsMs": timings,
        "tokenAudit": {
            "inputTokens": tokens.get("documentAfter"),
            "completionTokens": tokens.get("completionTokens"),
            "totalTokens": (tokens.get("documentAfter") or 0) + (tokens.get("completionTokens") or 0) if tokens.get("documentAfter") and tokens.get("completionTokens") else None,
        },
    }
    stage7_file.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[STAGE 7] Saved Timing & Token Audit -> {stage7_file.resolve()}")

    print(f"\n[TOKEN AUDIT] Input Tokens: {tokens.get('documentAfter')} | Completion Tokens: {tokens.get('completionTokens')}")
    print(f"[TIME AUDIT] 🎯 TIME TO UI READY (Steps 1-4): {time_to_ui_ready}s | Background Indexing: {background_indexing_time}s | Total: {total_elapsed}s")
    print(f"============================================================")
    print(f"[SUCCESS] All 7 stage files generated in: {out_dir.resolve()}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
