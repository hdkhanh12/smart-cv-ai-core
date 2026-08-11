from __future__ import annotations

import json
from pathlib import Path

from ai_core.evaluation.binding_review import build_binding_review, evaluate_binding_review
from ai_core.evaluation.initial_entity_metrics import evaluate_initial_entities
from ai_core.extraction.beeknoee import _initial_entity_snapshots


def _annotations(path: Path) -> Path:
    annotation = path / "annotations.json"
    annotation.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sourceNotesFile": "case.md",
                        "layoutStratum": "two_column",
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
    return annotation


def _artifact(path: Path, *, snapshot: bool = True) -> Path:
    result = path / "run-01" / "case.json"
    result.parent.mkdir()
    initial: dict[str, object] = {
        "dtoValidated": {"experiences": {"entities": 1}, "projects": {"entities": 0}}
    }
    if snapshot:
        initial["entitySnapshots"] = {
            "experiences": [{"jobTitle": "Data Engineer", "company": "Example Corp"}],
            "projects": [],
        }
    result.write_text(
        json.dumps(
            {
                "metadata": {"sourceName": "case.pdf"},
                "profile": {
                    "experiences": [
                        {
                            "jobTitle": "Data Engineer",
                            "company": "Example Corp",
                            "evidence": [{"pageNumber": 1, "blockId": "p1-b2"}],
                        }
                    ],
                    "projects": [],
                },
                "unifiedDocument": {
                    "pages": [
                        {
                            "pageNumber": 1,
                            "blocks": [
                                {
                                    "id": "p1-b2",
                                    "text": "Example Corp — Data Engineer",
                                    "readingOrder": 2,
                                    "bbox": [0, 0, 1, 1],
                                }
                            ],
                        }
                    ]
                },
                "audit": {
                    "localEvidenceBinding": {
                        "entities": [
                            {
                                "entityKind": "experience",
                                "entryIndex": 0,
                                "decision": "bound",
                                "reasonCode": "BOUND_EXACT_CONTEXT",
                                "evidenceReferences": [{"pageNumber": 1, "blockId": "p1-b2"}],
                            }
                        ]
                    },
                    "extractionFunnel": {
                        "provider": {"initial": initial},
                        "afterReconciliation": {
                            "experiences": {"entities": 1},
                            "projects": {"entities": 0},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return result.parent.parent


def test_binding_review_keeps_pending_block_decision_for_human_review(tmp_path: Path) -> None:
    annotations = _annotations(tmp_path)
    results = _artifact(tmp_path)

    payload = build_binding_review(annotations, results)

    record = payload["records"][0]
    assert record["status"] == "pending"
    assert record["boundBlocks"][0]["blockId"] == "p1-b2"
    assert record["review"]["verdict"] == "pending"


def test_initial_entity_evaluation_identifies_initial_title_match(tmp_path: Path) -> None:
    annotations = _annotations(tmp_path)
    results = _artifact(tmp_path)

    payload = evaluate_initial_entities(annotations, results)

    assert payload["aggregate"] == {
        "goldEntities": 1,
        "initialEntities": 1,
        "entitiesWithPrimaryIdentity": 1,
        "initialMatched": 1,
        "initialDtoEntities": 1,
        "correctionDtoEntities": 0,
        "finalEntities": 1,
    }


def test_binding_review_evaluation_uses_human_verdict_and_block_ids(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "boundBlocks": [{"blockId": "p1-b2"}, {"blockId": "p1-b3"}],
                        "review": {
                            "verdict": "correct",
                            "correctBlockIds": ["p1-b2"],
                        },
                    },
                    {"boundBlocks": [], "review": {"verdict": "incorrect", "correctBlockIds": []}},
                    {"boundBlocks": [], "review": {"verdict": "pending", "correctBlockIds": []}},
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = evaluate_binding_review(review_path)

    assert payload["metrics"] == {
        "reviewedEntities": 2,
        "correctEntities": 1,
        "incorrectEntities": 1,
        "uncertainEntities": 0,
        "pendingEntities": 1,
        "entityBindingPrecision": 0.5,
        "selectedBlockPrecision": 0.5,
        "selectedBlockRecall": 1.0,
    }


def test_initial_snapshot_excludes_descriptions_evidence_and_urls() -> None:
    snapshots = _initial_entity_snapshots(
        {
            "experiences": [
                {
                    "jobTitle": "Data Engineer",
                    "company": "Example Corp",
                    "description": ["Do not retain"],
                    "evidence": [{"text": "Do not retain"}],
                }
            ],
            "projects": [{"title": "Example Project", "url": "https://example.test"}],
        }
    )

    assert snapshots == {
        "experiences": [
            {
                "jobTitle": "Data Engineer",
                "company": "Example Corp",
                "sourceAliases": {"jobTitle": "jobTitle", "company": "company"},
            }
        ],
        "projects": [{"title": "Example Project", "sourceAliases": {"title": "title"}}],
    }


def test_initial_snapshot_canonicalizes_supported_identity_aliases() -> None:
    snapshots = _initial_entity_snapshots(
        {
            "experiences": [{"role": "Data Engineer", "employer": "Example Corp"}],
            "projects": [{"projectName": "Warehouse Pipeline"}],
        }
    )

    assert snapshots == {
        "experiences": [
            {
                "jobTitle": "Data Engineer",
                "company": "Example Corp",
                "sourceAliases": {"jobTitle": "role", "company": "employer"},
            }
        ],
        "projects": [{"title": "Warehouse Pipeline", "sourceAliases": {"title": "projectName"}}],
    }
