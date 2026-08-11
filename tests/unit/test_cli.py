from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_core.cli import build_parser, main
from ai_core.errors import CoreError, ErrorCode, ExitCode
from ai_core.extraction import CompatibilityFullDocumentProvider
from ai_core.pipeline import process_document
from ai_core.schemas import ProcessingResult


def write_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    return path


def test_process_valid_file_writes_schema_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_pdf(tmp_path / "synthetic.pdf")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "ai_core.cli.process_document",
        lambda path, **kwargs: process_document(
            path,
            extraction_provider=CompatibilityFullDocumentProvider(),
            **kwargs,
        ),
    )

    exit_code = main(["process", str(source), "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == ExitCode.SUCCESS
    assert payload["schemaVersion"] == "1.0.0"
    assert payload["status"] == "succeeded"
    assert payload["metadata"]["sourceName"] == "synthetic.pdf"
    assert payload["parsedDocument"]["metadata"]["pageCount"] == 0
    assert payload["warnings"][0]["code"] == "NO_TEXT_LAYER"
    assert str(output) in capsys.readouterr().out


def test_process_invalid_file_writes_detailed_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "missing.pdf"
    report = tmp_path / "error.json"

    exit_code = main(["process", str(source), "--report", str(report)])

    payload = json.loads(report.read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    assert exit_code == ExitCode.INPUT_ERROR
    assert payload["status"] == "failed"
    assert payload["errors"][0]["code"] == "FILE_NOT_FOUND"
    assert "FILE_NOT_FOUND" in captured.err
    assert str(report) in captured.err


def test_search_writes_ranked_report_without_loading_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jd = tmp_path / "job.txt"
    jd.write_text("Python data engineering", encoding="utf-8")
    result = ProcessingResult(
        status="succeeded",
        source_id="candidate",
        profile={
            "candidateName": "Alex",
            "skills": [{"name": "Python", "canonicalName": "Python"}],
        },
        embedding={
            "model": "Alibaba-NLP/gte-multilingual-base",
            "modelRevision": "main",
            "dimension": 768,
            "normalized": True,
            "vector": [1.0] + [0.0] * 767,
            "sourceHash": "a" * 64,
            "templateVersion": "gte-profile-v1",
            "durationMs": 1,
        },
    )
    output = tmp_path / "candidates"
    output.mkdir()
    (output / "alex.json").write_text(result.model_dump_json(by_alias=True), encoding="utf-8")
    report = tmp_path / "search.md"

    class FakeEmbedder:
        def embed_text(self, text: str) -> object:
            del text
            return result.embedding

    monkeypatch.setattr("ai_core.search.GteMultilingualEmbedder", FakeEmbedder)
    assert main(["search", "--jd", str(jd), "--cv-dir", str(output), "--report", str(report)]) == 0
    assert "Alex" in report.read_text(encoding="utf-8")
    assert "file:///" in report.read_text(encoding="utf-8")
    assert "Alex" in capsys.readouterr().out


def test_batch_writes_per_file_error_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "cvs"
    source_dir.mkdir()
    write_pdf(source_dir / "broken.pdf")
    output = tmp_path / "batch-json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "ai_core.cli.process_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CoreError(
                ErrorCode.LLM_TIMEOUT,
                "fixture timeout",
                stage="llm_provider",
            )
        ),
    )

    exit_code = main(["batch", str(source_dir), "--output", str(output), "--workers", "1"])

    artifact = output / "broken.error.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert exit_code == ExitCode.PROCESSING_ERROR
    assert payload["errors"][0]["code"] == "LLM_TIMEOUT"


def test_batch_refuses_to_mix_existing_json_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "cvs"
    source_dir.mkdir()
    write_pdf(source_dir / "candidate.pdf")
    output = tmp_path / "batch-json"
    output.mkdir()
    (output / "old.json").write_text("{}", encoding="utf-8")

    assert main(["batch", str(source_dir), "--output", str(output)]) == ExitCode.INPUT_ERROR


def test_benchmark_accepts_parallel_workers_and_writes_each_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "cvs"
    source_dir.mkdir()
    write_pdf(source_dir / "one.pdf")
    write_pdf(source_dir / "two.pdf")
    output = tmp_path / "benchmark"

    monkeypatch.setattr("ai_core.cli.BeeknoeeStructuredExtractionProvider", lambda **_: object())
    monkeypatch.setattr(
        "ai_core.cli.process_document",
        lambda source, **_: ProcessingResult(status="succeeded", source_id=source.name),
    )

    exit_code = main(
        [
            "benchmark",
            str(source_dir),
            "--output",
            str(output),
            "--raw-capture-dir",
            str(tmp_path / "capture"),
            "--synthetic-benchmark",
            "--workers",
            "2",
        ]
    )

    assert exit_code == ExitCode.SUCCESS
    assert (output / "run-01" / "one.json").exists()
    assert (output / "run-01" / "two.json").exists()


def test_inventory_ab_interleaves_arms_and_captures_initial_entities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "cvs"
    source_dir.mkdir()
    write_pdf(source_dir / "candidate.pdf")
    output = tmp_path / "inventory-ab"
    processed_scopes: list[str] = []

    def provider_factory(**kwargs: object) -> object:
        processed_scopes.append(str(kwargs["output_scope"]))
        return object()

    monkeypatch.setattr("ai_core.cli.BeeknoeeStructuredExtractionProvider", provider_factory)
    monkeypatch.setattr(
        "ai_core.cli.process_document",
        lambda source, **_: ProcessingResult(status="succeeded", source_id=source.name),
    )

    exit_code = main(
        [
            "benchmark-inventory-ab",
            str(source_dir),
            "--output",
            str(output),
            "--raw-capture-dir",
            str(tmp_path / "capture"),
            "--synthetic-benchmark",
            "--runs",
            "2",
        ]
    )

    assert exit_code == ExitCode.SUCCESS
    assert processed_scopes == [
        "entities-only",
        "entity-inventory",
        "entity-inventory",
        "entities-only",
    ]
    assert (output / "entities-only" / "run-01" / "candidate.json").exists()
    assert (output / "entity-inventory" / "run-02" / "candidate.json").exists()
    manifest = json.loads((output / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["workerCount"] == 1
    assert manifest["captureInitialEntities"] is True
    assert [job["arm"] for job in manifest["jobs"]] == [
        "entities-only",
        "entity-inventory",
        "entity-inventory",
        "entities-only",
    ]


def test_all_commands_are_registered() -> None:
    help_text = build_parser().format_help()
    for command in (
        "process",
        "batch",
        "summarize",
        "search",
        "evaluate",
        "benchmark-inventory-ab",
        "benchmark-header-mask-ab",
        "annotation-intake",
        "annotation-evaluate",
        "binding-review",
        "binding-review-evaluate",
        "initial-entity-evaluate",
    ):
        assert command in help_text


def test_annotation_intake_writes_local_generic_intermediate_record(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "case.md").write_text(
        "## case.pdf\n\n* Layout observations: single-column layout.\n\n"
        "### Employment — record every entry present\n\n"
        "1. **Employer/Organisation:** Example Corp\n\n* **Title:** Engineer\n"
        "* **Page:** Page 1\n\n"
        "### Skills — record every skill as written\n\n* Python\n",
        encoding="utf-8",
    )
    output = tmp_path / "intake.json"

    assert (
        main(
            ["annotation-intake", str(notes), "--output", str(output), "--synthetic-annotations"]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    record = payload["records"][0]
    assert record["layoutStratum"] == "single_column"
    assert record["facts"][0]["rawFields"]["Employer/Organisation"] == "Example Corp"
    assert record["facts"][0]["evidence"] == [{"pageNumber": 1, "reviewState": "verified"}]


def test_annotation_intake_keeps_project_with_verified_name_and_ambiguous_role(
    tmp_path: Path,
) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "case.md").write_text(
        "### Projects — record every entry present\n\n"
        "1. **Project Name:** Verified Project\n\n"
        "* **Role:** ambiguous (not explicitly labelled)\n",
        encoding="utf-8",
    )
    output = tmp_path / "intake.json"

    assert (
        main(["annotation-intake", str(notes), "--output", str(output), "--synthetic-annotations"])
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["records"][0]["facts"][0]["sourceState"] == "verified"
