from __future__ import annotations

import json
from pathlib import Path

from ai_core.evaluation.annotation_metrics import evaluate_annotation_intake


def test_annotation_evaluation_matches_all_annotated_employment_fields(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sourceNotesFile": "case.md",
                        "layoutStratum": "single_column",
                        "facts": [
                            {
                                "entityKind": "employment",
                                "sourceState": "verified",
                                "rawFields": {
                                    "Employer/Organisation": "Example Corp",
                                    "Title": "Data Engineer",
                                    "Date Text": "2020 - 2022",
                                },
                                "evidence": [{"pageNumber": 1}],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "case.json").write_text(
        json.dumps(
            {
                "metadata": {"sourceName": "case.pdf"},
                "profile": {
                    "experiences": [
                        {
                            "company": "Example Corp",
                            "jobTitle": "Data Engineer",
                            "startDate": "2020-01",
                            "endDate": "2022-01",
                            "evidence": [{"pageNumber": 1}],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    payload = evaluate_annotation_intake(annotations, results, scope="entities-only")

    metrics = payload["cases"][0]["metrics"]
    assert metrics["entityRecall"] == 1.0
    assert metrics["fieldCoverage"] == 1.0
    assert metrics["evidencePageOverlap"] == 1.0
    assert payload["cases"][0]["mismatchLedger"]["missingGold"] == []
    assert payload["cases"][0]["mismatchLedger"]["unmatchedExtracted"] == []


def test_annotation_evaluation_reads_project_title_from_profile_contract(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sourceNotesFile": "case.md",
                        "facts": [
                            {
                                "entityKind": "project",
                                "sourceState": "verified",
                                "rawFields": {"Project Name": "Warehouse Pipeline"},
                                "evidence": [{"pageNumber": 1}],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "case.json").write_text(
        json.dumps(
            {
                "metadata": {"sourceName": "case.pdf"},
                "profile": {
                    "projects": [{"title": "Warehouse Pipeline", "evidence": [{"pageNumber": 1}]}]
                },
            }
        ),
        encoding="utf-8",
    )

    payload = evaluate_annotation_intake(annotations, results, scope="entities-only")

    assert payload["cases"][0]["metrics"]["entityRecall"] == 1.0


def test_annotation_evaluation_ignores_non_extraction_json(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {"records": [{"sourceNotesFile": "case.md", "facts": []}]},
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "case.json").write_text(
        json.dumps({"metadata": {"sourceName": "case.pdf"}, "profile": {}}),
        encoding="utf-8",
    )
    (results / "benchmark_manifest.json").write_text("{}", encoding="utf-8")
    raw_capture = results / "raw-capture-ttl-7d"
    raw_capture.mkdir()
    (raw_capture / "capture.json").write_text("{}", encoding="utf-8")

    payload = evaluate_annotation_intake(annotations, results, scope="entities-only")

    assert len(payload["cases"]) == 1
    assert payload["unmatchedOutputs"] == []


def test_annotation_evaluation_does_not_match_title_substrings(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sourceNotesFile": "case.md",
                        "facts": [
                            {
                                "entityKind": "employment",
                                "sourceState": "verified",
                                "rawFields": {
                                    "Employer/Organisation": "Example Corp",
                                    "Title": "Data Engineer",
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "case.json").write_text(
        json.dumps(
            {
                "metadata": {"sourceName": "case.pdf"},
                "profile": {
                    "experiences": [{"company": "Example Corp", "jobTitle": "Senior Data Engineer"}]
                },
            }
        ),
        encoding="utf-8",
    )

    payload = evaluate_annotation_intake(annotations, results, scope="entities-only")

    assert payload["cases"][0]["metrics"]["entityRecall"] == 0.0
