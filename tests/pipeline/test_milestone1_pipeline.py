from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_core.cli import main
from ai_core.errors import ExitCode
from ai_core.extraction import CompatibilityFullDocumentProvider
from ai_core.pipeline import process_document
from ai_core.schemas import ProcessingResult, ProcessingStatus


def test_validated_document_flows_from_pipeline_to_cli_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")

    in_memory = process_document(source, extraction_provider=CompatibilityFullDocumentProvider())
    monkeypatch.setattr(
        "ai_core.cli.process_document",
        lambda path, **kwargs: process_document(
            path,
            extraction_provider=CompatibilityFullDocumentProvider(),
            **kwargs,
        ),
    )
    output = tmp_path / "processing-result.json"
    exit_code = main(["process", str(source), "--output", str(output)])
    artifact = ProcessingResult.model_validate_json(output.read_text(encoding="utf-8"))

    assert exit_code == ExitCode.SUCCESS
    assert in_memory.status == artifact.status == ProcessingStatus.SUCCEEDED
    assert in_memory.metadata == artifact.metadata
    assert artifact.parsed_document is not None
    assert artifact.parsed_document.raw_text == ""
    assert artifact.summary is not None
    assert artifact.summary.sentence_count == 2
    assert {score.name for score in artifact.scores} == {
        "profile_completeness.v2",
        "cv_quality.v2",
    }
    assert json.loads(output.read_text(encoding="utf-8"))["schemaVersion"] == "1.0.0"
