from __future__ import annotations

from datetime import date
from pathlib import Path

from ai_core.parsers.unified import unified_from_lines
from ai_core.reconciliation import bind_exact_entity_evidence, reconcile_profile
from ai_core.schemas import CVProfile, Experience
from ai_core.validation import validate_input


def _document(tmp_path: Path, lines: list[str]):
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    return unified_from_lines(validate_input(source), "fixture", [lines])


def test_exact_binder_adds_evidence_only_with_nearby_company_context(tmp_path: Path) -> None:
    document = _document(tmp_path, ["Northstar Labs", "Platform Engineer | 2022 - Present"])
    profile = CVProfile(
        experiences=[
            Experience(
                job_title="Platform Engineer",
                company="Northstar Labs",
                start_date=date(2022, 1, 1),
            )
        ]
    )

    bound, audit = bind_exact_entity_evidence(profile, document)
    reconciled = reconcile_profile(bound, document, provider_name="fixture-llm")

    assert len(bound.experiences[0].evidence) == 2
    assert audit["entities"][0]["decision"] == "bound"  # type: ignore[index]
    assert len(reconciled.experiences) == 1


def test_exact_binder_refuses_title_without_company_or_date_context(tmp_path: Path) -> None:
    document = _document(tmp_path, ["Platform Engineer"])
    profile = CVProfile(experiences=[Experience(job_title="Platform Engineer")])

    bound, audit = bind_exact_entity_evidence(profile, document)

    assert bound.experiences[0].evidence == []
    assert audit["entities"][0]["reasonCode"] == "TITLE_WITHOUT_STRONG_CONTEXT"  # type: ignore[index]


def test_exact_binder_uses_only_the_title_anchor_with_matching_date_context(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        [
            "Version 1 | Senior Data Engineer | 2024 - Present",
            "Version 1 | Data Engineer | 2020 - 2023",
        ],
    )
    profile = CVProfile(
        experiences=[
            Experience(
                job_title="Data Engineer",
                company="Version 1",
                start_date=date(2020, 1, 1),
            )
        ]
    )

    bound, audit = bind_exact_entity_evidence(profile, document)

    assert [item.block_id for item in bound.experiences[0].evidence] == ["p1-b2"]
    assert audit["entities"][0]["reasonCode"] == "BOUND_EXACT_CONTEXT"  # type: ignore[index]


def test_exact_binder_refuses_multiple_equally_supported_title_anchors(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        [
            "Northstar Labs | Platform Engineer | 2022 - Present",
            "Northstar Labs | Platform Engineer | 2022 - Present",
        ],
    )
    profile = CVProfile(
        experiences=[
            Experience(
                job_title="Platform Engineer",
                company="Northstar Labs",
                start_date=date(2022, 1, 1),
            )
        ]
    )

    bound, audit = bind_exact_entity_evidence(profile, document)

    assert bound.experiences[0].evidence == []
    assert audit["entities"][0]["reasonCode"] == "TITLE_AMBIGUOUS"  # type: ignore[index]
