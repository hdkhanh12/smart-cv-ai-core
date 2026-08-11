"""Command-line interface for the standalone AI core."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore

from ai_core.errors import CoreError, ErrorCode, ExitCode, Issue
from ai_core.evaluation.annotation_intake import build_annotation_intake
from ai_core.evaluation.annotation_metrics import (
    evaluate_annotation_intake,
    write_annotation_evaluation_report,
)
from ai_core.evaluation.binding_review import (
    build_binding_review,
    evaluate_binding_review,
    write_binding_review_evaluation_report,
    write_binding_review_report,
)
from ai_core.evaluation.initial_entity_metrics import (
    evaluate_initial_entities,
    write_initial_entity_report,
)
from ai_core.extraction.beeknoee import BeeknoeeStructuredExtractionProvider
from ai_core.pipeline import process_document
from ai_core.reporting import write_human_review_report, write_process_report
from ai_core.schemas import ProcessingResult, ProcessingStatus
from ai_core.search import read_jd, search_results, write_search_report
from ai_core.validation import DEFAULT_MAX_FILE_SIZE
from evaluation.hybrid_regression import run_regression


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _error_report_path(source: Path, explicit: Path | None) -> Path:
    if explicit is not None and explicit.suffix.lower() == ".json":
        return explicit
    return Path("outputs") / "errors" / f"{source.stem}.error.json"


def _failed_result(source: Path, issue: Issue) -> ProcessingResult:
    details = issue.details if isinstance(issue.details, dict) else {}
    funnel = details.get("funnel")
    audit = {"extractionFunnel": {"provider": funnel}} if isinstance(funnel, dict) else {}
    return ProcessingResult(
        status=ProcessingStatus.FAILED,
        source_id=source.name or "unknown",
        errors=[issue],
        audit=audit,
    )


def _run_process(args: argparse.Namespace) -> int:
    source = Path(args.file)
    try:
        result = process_document(
            source,
            max_size_bytes=int(args.max_size_mb * 1024 * 1024),
            debug_dir=args.debug_dir,
            embed=args.embed,
        )
        destination = args.output or Path("outputs") / f"{source.stem}.json"
        _write_json(destination, result.model_dump(mode="json", by_alias=True))
        if args.report:
            write_process_report(result, args.report)
        print(destination)
        return ExitCode.SUCCESS
    except CoreError as exc:
        report = _error_report_path(source, args.report)
        result = _failed_result(source, exc.issue)
        _write_json(report, result.model_dump(mode="json", by_alias=True))
        print(f"{exc.issue.code}: {exc.issue.message} Report: {report}", file=sys.stderr)
        return ExitCode.INPUT_ERROR
    except Exception as exc:  # pragma: no cover - defensive boundary
        issue = Issue(
            code=ErrorCode.PROCESSING_FAILED,
            message="Unexpected processing failure.",
            stage="cli",
            details={"exceptionType": type(exc).__name__, "traceback": traceback.format_exc()},
        )
        report = _error_report_path(source, args.report)
        _write_json(report, _failed_result(source, issue).model_dump(mode="json", by_alias=True))
        print(f"{issue.code}: {issue.message} Report: {report}", file=sys.stderr)
        return ExitCode.PROCESSING_ERROR


def _not_implemented(args: argparse.Namespace) -> int:
    command = str(args.command)
    print(
        f"{ErrorCode.NOT_IMPLEMENTED}: '{command}' is reserved for a later milestone.",
        file=sys.stderr,
    )
    return ExitCode.NOT_IMPLEMENTED


def _run_batch(args: argparse.Namespace) -> int:
    if args.output.exists() and any(args.output.glob("*.json")) and not args.allow_existing_output:
        print(
            "Refusing to mix batch artifacts. Choose an empty --output directory or pass "
            "--allow-existing-output deliberately.",
            file=sys.stderr,
        )
        return ExitCode.INPUT_ERROR
    files = sorted(
        path for path in args.directory.rglob("*") if path.suffix.lower() in {".pdf", ".docx"}
    )
    semaphore = Semaphore(args.rate_limit)
    results: list[ProcessingResult] = []

    def process_one(source: Path) -> ProcessingResult:
        with semaphore:
            return process_document(source, embed=args.embed)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, source): source for source in files}
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                _write_json(
                    args.output / f"{source.stem}.json",
                    result.model_dump(mode="json", by_alias=True),
                )
                results.append(result)
            except CoreError as exc:
                failed = _failed_result(source, exc.issue)
                _write_json(
                    args.output / f"{source.stem}.error.json",
                    failed.model_dump(mode="json", by_alias=True),
                )
                results.append(failed)
            except Exception as exc:  # pragma: no cover - defensive batch boundary
                issue = Issue(
                    code=ErrorCode.PROCESSING_FAILED,
                    message="Unexpected batch processing failure.",
                    stage="batch",
                    details={"exceptionType": type(exc).__name__},
                )
                failed = _failed_result(source, issue)
                _write_json(
                    args.output / f"{source.stem}.error.json",
                    failed.model_dump(mode="json", by_alias=True),
                )
                results.append(failed)
    report = args.report or args.output / "batch_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    succeeded = sum(result.status == ProcessingStatus.SUCCEEDED for result in results)
    report.write_text(
        "# Batch Processing Report\n\n"
        "| Files | Succeeded | Failed | Workers | Rate limit |\n|---:|---:|---:|---:|---:|\n"
        f"| {len(results)} | {succeeded} | {len(results) - succeeded} | "
        f"{args.workers} | {args.rate_limit} |\n",
        encoding="utf-8",
    )
    human_report_name = "human_review_new.md" if "new" in str(report).lower() else "human_review.md"
    human_report = report.parent / human_report_name
    write_human_review_report(results, human_report)
    print(report)
    print(human_report)
    return ExitCode.SUCCESS if succeeded == len(results) else ExitCode.PROCESSING_ERROR


def _run_benchmark(args: argparse.Namespace) -> int:
    if not args.synthetic_benchmark:
        print("--synthetic-benchmark is required to enable raw capture.", file=sys.stderr)
        return ExitCode.INPUT_ERROR
    if args.workers < 1:
        print("--workers must be at least 1.", file=sys.stderr)
        return ExitCode.INPUT_ERROR
    files = sorted(
        path for path in args.directory.rglob("*") if path.suffix.lower() in {".pdf", ".docx"}
    )
    if args.include_file:
        included = set(args.include_file)
        files = [path for path in files if path.name in included]
    repeat_overrides = dict(item.split("=", 1) for item in args.repeat_file)
    jobs: list[tuple[Path, int]] = []
    for source in files:
        repeats = int(repeat_overrides.get(source.name, args.runs))
        for run_number in range(1, repeats + 1):
            jobs.append((source, run_number))
    _write_json(
        args.output / "benchmark_manifest.json",
        {
            "schemaVersion": "1.0",
            "workerCount": args.workers,
            "inputRepresentation": args.input_representation,
            "evidenceMode": args.evidence_mode,
            "evidenceContract": args.evidence_contract,
            "localEvidenceBinding": args.local_evidence_binding,
            "captureInitialEntities": args.capture_initial_entities,
            "outputScope": args.output_scope,
            "jobs": [
                {"sourceName": source.name, "repeatIndex": run_number}
                for source, run_number in jobs
            ],
        },
    )

    def process_one(source: Path) -> ProcessingResult:
        provider = BeeknoeeStructuredExtractionProvider(
            raw_capture_dir=args.raw_capture_dir,
            input_representation=args.input_representation,
            evidence_mode=args.evidence_mode,
            evidence_contract=args.evidence_contract,
            output_scope=args.output_scope,
            capture_initial_entities=args.capture_initial_entities,
        )
        return process_document(
            source,
            extraction_provider=provider,
            capture_reconciliation_dispositions=True,
            local_evidence_binding=args.local_evidence_binding,
        )

    results: list[ProcessingResult | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, source): (index, source, run_number)
            for index, (source, run_number) in enumerate(jobs)
        }
        for future in as_completed(futures):
            index, source, run_number = futures[future]
            try:
                result = future.result()
                destination = args.output / f"run-{run_number:02d}" / f"{source.stem}.json"
            except CoreError as exc:
                result = _failed_result(source, exc.issue)
                destination = args.output / f"run-{run_number:02d}" / f"{source.stem}.error.json"
            except Exception as exc:  # pragma: no cover - defensive benchmark boundary
                issue = Issue(
                    code=ErrorCode.PROCESSING_FAILED,
                    message="Unexpected benchmark processing failure.",
                    stage="benchmark",
                    details={"exceptionType": type(exc).__name__},
                )
                result = _failed_result(source, issue)
                destination = args.output / f"run-{run_number:02d}" / f"{source.stem}.error.json"
            _write_json(destination, result.model_dump(mode="json", by_alias=True))
            results[index] = result
    ordered_results = [result for result in results if result is not None]
    write_human_review_report(ordered_results, args.output / "human_review.md")
    print(args.output / "human_review.md")
    return ExitCode.SUCCESS


def _run_inventory_ab_benchmark(args: argparse.Namespace) -> int:
    """Run a sequential, interleaved inventory-contract comparison locally.

    Both arms use the same representation and no-evidence policy. They differ
    only in request breadth: the established entities-only contract versus a
    narrow identity-and-date inventory. Sequential execution deliberately
    keeps paired calls adjacent, preventing worker concurrency from turning
    gateway time drift into an apparent prompt effect.
    """

    if not args.synthetic_benchmark:
        print("--synthetic-benchmark is required to enable raw capture.", file=sys.stderr)
        return ExitCode.INPUT_ERROR
    files = sorted(
        path for path in args.directory.rglob("*") if path.suffix.lower() in {".pdf", ".docx"}
    )
    if args.include_file:
        included = set(args.include_file)
        files = [path for path in files if path.name in included]
    repeat_overrides = dict(item.split("=", 1) for item in args.repeat_file)
    if args.ab_dimension == "header-masking":
        arms = (
            ("blanket-legacy", "entities-only", "blanket-legacy"),
            ("selective", "entities-only", "selective"),
        )
    else:
        arms = (
            ("entities-only", "entities-only", "selective"),
            ("entity-inventory", "entity-inventory", "selective"),
        )
    jobs: list[tuple[Path, int, str, str, str]] = []
    for source in files:
        repeats = int(repeat_overrides.get(source.name, args.runs))
        for run_number in range(1, repeats + 1):
            # Reverse the local pair every other repeat; neither arm always
            # receives the first call in a pair.
            arm_order = arms if run_number % 2 else tuple(reversed(arms))
            for arm_name, output_scope, header_masking_mode in arm_order:
                jobs.append((source, run_number, arm_name, output_scope, header_masking_mode))
    _write_json(
        args.output / "benchmark_manifest.json",
        {
            "schemaVersion": "1.0",
            "kind": f"interleaved_{args.ab_dimension}_ab",
            "workerCount": 1,
            "inputRepresentation": args.input_representation,
            "evidenceMode": "baseline",
            "evidenceContract": "optional",
            "captureInitialEntities": True,
            "armOrderPolicy": "alternate per source/repeat: A,B then B,A",
            "arms": [
                {
                    "name": arm_name,
                    "outputScope": output_scope,
                    "headerMaskingMode": header_masking_mode,
                }
                for arm_name, output_scope, header_masking_mode in arms
            ],
            "jobs": [
                {
                    "sequence": index + 1,
                    "sourceName": source.name,
                    "repeatIndex": run_number,
                    "arm": arm_name,
                    "outputScope": output_scope,
                }
                for index, (
                    source,
                    run_number,
                    arm_name,
                    output_scope,
                    header_masking_mode,
                ) in enumerate(jobs)
            ],
        },
    )
    results_by_arm: dict[str, list[ProcessingResult]] = {
        arm_name: [] for arm_name, _, _ in arms
    }
    for source, run_number, arm_name, output_scope, header_masking_mode in jobs:
        destination = args.output / arm_name / f"run-{run_number:02d}" / f"{source.stem}.json"
        try:
            provider = BeeknoeeStructuredExtractionProvider(
                raw_capture_dir=args.raw_capture_dir / arm_name,
                input_representation=args.input_representation,
                evidence_mode="baseline",
                evidence_contract="optional",
                output_scope=output_scope,
                capture_initial_entities=True,
            )
            result = process_document(
                source,
                extraction_provider=provider,
                capture_reconciliation_dispositions=True,
                local_evidence_binding="none",
                header_masking_mode=header_masking_mode,
            )
        except CoreError as exc:
            result = _failed_result(source, exc.issue)
            destination = destination.with_suffix(".error.json")
        except Exception as exc:  # pragma: no cover - defensive benchmark boundary
            issue = Issue(
                code=ErrorCode.PROCESSING_FAILED,
                message="Unexpected inventory A/B processing failure.",
                stage="benchmark",
                details={"exceptionType": type(exc).__name__},
            )
            result = _failed_result(source, issue)
            destination = destination.with_suffix(".error.json")
        _write_json(destination, result.model_dump(mode="json", by_alias=True))
        results_by_arm[arm_name].append(result)
    for arm_name, _, _ in arms:
        report = args.output / arm_name / "human_review.md"
        write_human_review_report(results_by_arm[arm_name], report)
        print(report)
    return ExitCode.SUCCESS


def _run_summarize(args: argparse.Namespace) -> int:
    directory: Path = args.directory
    output_report: Path = args.output or Path("outputs/reports/human_review.md")
    if not directory.exists():
        print(f"Directory not found: {directory}", file=sys.stderr)
        return int(ExitCode.INPUT_ERROR)
    json_files = sorted(directory.glob("*.json"))
    results: list[ProcessingResult] = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            results.append(ProcessingResult.model_validate(data))
        except Exception:
            continue
    write_human_review_report(results, output_report)
    print(output_report)
    return int(ExitCode.SUCCESS)


def _run_search(args: argparse.Namespace) -> int:
    try:
        matches = search_results(read_jd(args.jd), args.cv_dir, top_k=args.top_k)
        print("Rank | Candidate Name | File Name | Cosine Similarity Score | Top Skills")
        for match in matches:
            print(
                f"{match.rank} | {match.candidate_name} | {match.source.name} | "
                f"{match.similarity * 100:.2f}% | {', '.join(match.top_skills) or '-'}"
            )
        report = args.report or Path("outputs/reports/search_report.md")
        write_search_report(matches, args.jd, report)
        print(report)
        return ExitCode.SUCCESS
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return ExitCode.INPUT_ERROR


def _run_evaluate(args: argparse.Namespace) -> int:
    report = run_regression(Path("tests/fixtures/hybrid_v2_cases.json"), args.output)
    markdown = args.report or args.output.with_name("evaluation_report.md")
    hybrid = report["hybridV2"]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        "# Extraction Evaluation Report\n\n"
        "| Golden cases | Precision | Recall | F1 | Exact match accuracy | "
        "Evidence grounding ratio |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"| {report['caseCount']} | {hybrid['precision']} | {hybrid['recall']} | {hybrid['f1']} | "
        f"{hybrid['exactCaseAccuracy']} | {hybrid.get('evidenceGroundingRatio', 0)} |\n",
        encoding="utf-8",
    )
    print(markdown)
    return ExitCode.SUCCESS


def _run_annotation_intake(args: argparse.Namespace) -> int:
    if not args.synthetic_annotations:
        print("--synthetic-annotations is required for local annotation intake.", file=sys.stderr)
        return ExitCode.INPUT_ERROR
    try:
        payload = build_annotation_intake(args.directory)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return ExitCode.INPUT_ERROR
    _write_json(args.output, payload)
    print(args.output)
    return ExitCode.SUCCESS


def _run_annotation_evaluate(args: argparse.Namespace) -> int:
    try:
        payload = evaluate_annotation_intake(args.annotations, args.results, scope=args.scope)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return ExitCode.INPUT_ERROR
    _write_json(args.output, payload)
    report = args.report or args.output.with_suffix(".md")
    write_annotation_evaluation_report(payload, report)
    print(report)
    return ExitCode.SUCCESS


def _run_binding_review(args: argparse.Namespace) -> int:
    try:
        payload = build_binding_review(args.annotations, args.results)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return ExitCode.INPUT_ERROR
    _write_json(args.output, payload)
    report = args.report or args.output.with_suffix(".md")
    write_binding_review_report(payload, report)
    print(report)
    return ExitCode.SUCCESS


def _run_initial_entity_evaluate(args: argparse.Namespace) -> int:
    try:
        payload = evaluate_initial_entities(args.annotations, args.results)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return ExitCode.INPUT_ERROR
    _write_json(args.output, payload)
    report = args.report or args.output.with_suffix(".md")
    write_initial_entity_report(payload, report)
    print(report)
    return ExitCode.SUCCESS


def _run_binding_review_evaluate(args: argparse.Namespace) -> int:
    try:
        payload = evaluate_binding_review(args.review)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return ExitCode.INPUT_ERROR
    _write_json(args.output, payload)
    report = args.report or args.output.with_suffix(".md")
    write_binding_review_evaluation_report(payload, report)
    print(report)
    return ExitCode.SUCCESS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smart-cv-ai", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Validate and process one CV.")
    process_parser.add_argument("file", type=Path)
    process_parser.add_argument("--output", type=Path)
    process_parser.add_argument("--report", type=Path)
    process_parser.add_argument(
        "--embed", action="store_true", help="Create the local GTE profile vector."
    )
    process_parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Explicitly store raw/normalized text for local debugging.",
    )
    process_parser.add_argument(
        "--max-size-mb",
        type=float,
        default=DEFAULT_MAX_FILE_SIZE / 1024 / 1024,
    )
    process_parser.set_defaults(handler=_run_process)

    batch_parser = subparsers.add_parser("batch", help="Process a CV directory concurrently.")
    batch_parser.add_argument("directory", type=Path)
    batch_parser.add_argument("--output", type=Path, default=Path("outputs"))
    batch_parser.add_argument("--report", type=Path)
    batch_parser.add_argument("--workers", type=int, default=4)
    batch_parser.add_argument("--rate-limit", type=int, default=2)
    batch_parser.add_argument("--embed", action="store_true", help="Create local GTE vectors.")
    batch_parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into a non-empty output directory (not recommended for evaluation).",
    )
    batch_parser.set_defaults(handler=_run_batch)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run explicit synthetic CV extraction repeats with expiring raw capture."
    )
    benchmark_parser.add_argument("directory", type=Path)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--raw-capture-dir", type=Path, required=True)
    benchmark_parser.add_argument("--synthetic-benchmark", action="store_true")
    benchmark_parser.add_argument("--runs", type=int, default=1)
    benchmark_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent benchmark jobs; default 1 preserves sequential experiment behavior.",
    )
    benchmark_parser.add_argument(
        "--input-representation",
        choices=("layout-transcript", "layout-native", "canonical-text"),
        default="layout-transcript",
        help="Experiment-only LLM document representation; default remains layout transcript.",
    )
    benchmark_parser.add_argument(
        "--evidence-mode",
        choices=("baseline", "entity-bound"),
        default="baseline",
        help="Experiment-only entity evidence instruction; default remains baseline.",
    )
    benchmark_parser.add_argument(
        "--evidence-contract",
        choices=("optional", "required-initial"),
        default="optional",
        help="Experimental entity evidence schema contract; default remains optional.",
    )
    benchmark_parser.add_argument(
        "--local-evidence-binding",
        choices=("none", "exact-entity"),
        default="none",
        help="Experimental local entity-to-block binding; default remains none.",
    )
    benchmark_parser.add_argument(
        "--capture-initial-entities",
        action="store_true",
        help=(
            "Benchmark-only: retain initial entity identity fields in local artifacts "
            "to diagnose upstream omissions."
        ),
    )
    benchmark_parser.add_argument(
        "--output-scope",
        choices=("full", "entities-only"),
        default="full",
        help="Experiment-only output contract scope; default remains the full profile.",
    )
    benchmark_parser.add_argument("--include-file", action="append", default=[], metavar="FILE")
    benchmark_parser.add_argument(
        "--repeat-file", action="append", default=[], metavar="FILE=COUNT"
    )
    benchmark_parser.set_defaults(handler=_run_benchmark)

    inventory_ab_parser = subparsers.add_parser(
        "benchmark-inventory-ab",
        help=(
            "Run a sequential interleaved benchmark of entities-only against "
            "the narrow entity-inventory contract."
        ),
    )
    inventory_ab_parser.add_argument("directory", type=Path)
    inventory_ab_parser.add_argument("--output", type=Path, required=True)
    inventory_ab_parser.add_argument("--raw-capture-dir", type=Path, required=True)
    inventory_ab_parser.add_argument("--synthetic-benchmark", action="store_true")
    inventory_ab_parser.add_argument("--runs", type=int, default=1)
    inventory_ab_parser.add_argument(
        "--input-representation",
        choices=("layout-transcript", "layout-native", "canonical-text"),
        default="layout-transcript",
        help="Shared experiment-only representation for both arms.",
    )
    inventory_ab_parser.add_argument("--include-file", action="append", default=[], metavar="FILE")
    inventory_ab_parser.add_argument(
        "--repeat-file", action="append", default=[], metavar="FILE=COUNT"
    )
    inventory_ab_parser.set_defaults(
        handler=_run_inventory_ab_benchmark,
        ab_dimension="entity-inventory",
    )

    header_mask_ab_parser = subparsers.add_parser(
        "benchmark-header-mask-ab",
        help=(
            "Run a sequential interleaved benchmark of legacy blanket header masking "
            "against selective PII masking."
        ),
    )
    header_mask_ab_parser.add_argument("directory", type=Path)
    header_mask_ab_parser.add_argument("--output", type=Path, required=True)
    header_mask_ab_parser.add_argument("--raw-capture-dir", type=Path, required=True)
    header_mask_ab_parser.add_argument("--synthetic-benchmark", action="store_true")
    header_mask_ab_parser.add_argument("--runs", type=int, default=1)
    header_mask_ab_parser.add_argument(
        "--input-representation",
        choices=("layout-transcript", "layout-native", "canonical-text"),
        default="layout-transcript",
        help="Shared experiment-only representation for both arms.",
    )
    header_mask_ab_parser.add_argument(
        "--include-file", action="append", default=[], metavar="FILE"
    )
    header_mask_ab_parser.add_argument(
        "--repeat-file", action="append", default=[], metavar="FILE=COUNT"
    )
    header_mask_ab_parser.set_defaults(
        handler=_run_inventory_ab_benchmark,
        ab_dimension="header-masking",
    )

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate executive human review markdown summary."
    )
    summarize_parser.add_argument("directory", type=Path)
    summarize_parser.add_argument(
        "--output", type=Path, default=Path("outputs/reports/human_review.md")
    )
    summarize_parser.set_defaults(handler=_run_summarize)

    search_parser = subparsers.add_parser(
        "search", help="Rank local CVs with GTE dense similarity."
    )
    search_parser.add_argument("--jd", type=Path, required=True)
    search_parser.add_argument("--cv-dir", type=Path, required=True)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument(
        "--report", type=Path, default=Path("outputs/reports/search_report.md")
    )
    search_parser.set_defaults(handler=_run_search)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run evaluation (Milestone 8).")
    evaluate_parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    evaluate_parser.add_argument("--report", type=Path)
    evaluate_parser.set_defaults(handler=_run_evaluate)

    intake_parser = subparsers.add_parser(
        "annotation-intake",
        help="Convert synthetic reviewer Markdown notes into local benchmark intake JSON.",
    )
    intake_parser.add_argument("directory", type=Path)
    intake_parser.add_argument("--output", type=Path, required=True)
    intake_parser.add_argument("--synthetic-annotations", action="store_true")
    intake_parser.set_defaults(handler=_run_annotation_intake)

    annotation_evaluate_parser = subparsers.add_parser(
        "annotation-evaluate",
        help="Compare extraction artifacts with local synthetic annotation intake JSON.",
    )
    annotation_evaluate_parser.add_argument("--annotations", type=Path, required=True)
    annotation_evaluate_parser.add_argument("--results", type=Path, required=True)
    annotation_evaluate_parser.add_argument("--output", type=Path, required=True)
    annotation_evaluate_parser.add_argument("--report", type=Path)
    annotation_evaluate_parser.add_argument(
        "--scope",
        choices=("full", "entities-only"),
        default="full",
        help=(
            "Contract scope to evaluate; entities-only excludes education "
            "and credentials from recall."
        ),
    )
    annotation_evaluate_parser.set_defaults(handler=_run_annotation_evaluate)

    binding_review_parser = subparsers.add_parser(
        "binding-review",
        help="Create a local human-review packet for binder block selections.",
    )
    binding_review_parser.add_argument("--annotations", type=Path, required=True)
    binding_review_parser.add_argument("--results", type=Path, required=True)
    binding_review_parser.add_argument("--output", type=Path, required=True)
    binding_review_parser.add_argument("--report", type=Path)
    binding_review_parser.set_defaults(handler=_run_binding_review)

    binding_review_evaluate_parser = subparsers.add_parser(
        "binding-review-evaluate",
        help="Score completed local human judgments of binder block selections.",
    )
    binding_review_evaluate_parser.add_argument("--review", type=Path, required=True)
    binding_review_evaluate_parser.add_argument("--output", type=Path, required=True)
    binding_review_evaluate_parser.add_argument("--report", type=Path)
    binding_review_evaluate_parser.set_defaults(handler=_run_binding_review_evaluate)

    initial_entity_parser = subparsers.add_parser(
        "initial-entity-evaluate",
        help="Measure initial LLM entity/title coverage before downstream stages.",
    )
    initial_entity_parser.add_argument("--annotations", type=Path, required=True)
    initial_entity_parser.add_argument("--results", type=Path, required=True)
    initial_entity_parser.add_argument("--output", type=Path, required=True)
    initial_entity_parser.add_argument("--report", type=Path)
    initial_entity_parser.set_defaults(handler=_run_initial_entity_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
