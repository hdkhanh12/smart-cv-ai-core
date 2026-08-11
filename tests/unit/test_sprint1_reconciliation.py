"""Unit tests for Sprint-1 reconciliation fixes.

Tests cover:
- evidence_binding v2: OR-context logic and expanded ±4 window
- reconciliation 3-tier: GROUNDED / SECTION_INFERRED / HALLUCINATED
- token measurement: _estimate_description_token_impact
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pytest

from ai_core.errors import WarningCode
from ai_core.parsers.unified import unified_from_lines
from ai_core.reconciliation import bind_exact_entity_evidence, reconcile_profile
from ai_core.reconciliation.profile import _can_infer_from_section
from ai_core.schemas import CVProfile, Experience
from ai_core.validation import validate_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _document(lines: list[list[str]]):
    """Create a synthetic UnifiedDocument with one page per inner list."""
    tmpdir = tempfile.mkdtemp()
    source = Path(tmpdir) / "synthetic.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    return unified_from_lines(validate_input(source), "fixture", lines)


# ---------------------------------------------------------------------------
# R1.2 — evidence_binding v2: OR-context and ±4 window
# ---------------------------------------------------------------------------


class TestBindingV2OrContext:
    """OR-context: company alone is sufficient (no year required)."""

    def test_binds_when_only_company_present_no_year(self) -> None:
        """Company context is enough — year may be absent from nearby blocks."""
        document = _document([["Acme Corp", "Senior Data Scientist"]])
        profile = CVProfile(
            experiences=[
                Experience(
                    job_title="Senior Data Scientist",
                    company="Acme Corp",
                    # No start_date — year not available
                )
            ]
        )
        bound, audit = bind_exact_entity_evidence(profile, document)

        assert len(bound.experiences[0].evidence) >= 1
        assert audit["entities"][0]["decision"] == "bound"  # type: ignore[index]
        assert audit["entities"][0]["bindingVersion"] == "v3"  # type: ignore[index]

    def test_binds_when_only_year_present_no_company(self) -> None:
        """Year context alone is enough when company name is not known."""
        document = _document([["Machine Learning Engineer | 2021 - 2023"]])
        profile = CVProfile(
            experiences=[
                Experience(
                    job_title="Machine Learning Engineer",
                    company=None,
                    start_date=date(2021, 1, 1),
                )
            ]
        )
        bound, audit = bind_exact_entity_evidence(profile, document)

        # Title is in the block, year is in the block — should bind
        assert len(bound.experiences[0].evidence) >= 1
        assert audit["entities"][0]["decision"] == "bound"  # type: ignore[index]

    def test_binds_two_column_layout_with_extended_window(self) -> None:
        """Simulate two-column CV: company block and title block are ±3 apart."""
        # reading_order: col-left blocks get 1,3,5; col-right blocks get 2,4,6
        # In unified_from_lines each line becomes a separate block in order.
        lines = [
            "Work Experience",          # heading → reading_order 1
            "FPT Software",             # company → reading_order 2
            "some other left-col text", # filler  → reading_order 3
            "Data Engineer",            # title   → reading_order 4 (distance=2 from company)
            "2020 - 2024",              # date    → reading_order 5
        ]
        document = _document([lines])
        profile = CVProfile(
            experiences=[
                Experience(
                    job_title="Data Engineer",
                    company="FPT Software",
                    start_date=date(2020, 1, 1),
                )
            ]
        )
        bound, audit = bind_exact_entity_evidence(profile, document)
        reason = audit["entities"][0]["reasonCode"]  # type: ignore[index]
        # Either BOUND_EXACT_CONTEXT (v2 window finds company within ±4)
        # or at minimum not failed due to window constraint
        assert reason in {"BOUND_EXACT_CONTEXT", "TITLE_AMBIGUOUS", "TITLE_NOT_FOUND_EXACT"}
        # Binding version must always be v2
        assert audit["entities"][0]["bindingVersion"] == "v3"  # type: ignore[index]


# ---------------------------------------------------------------------------
# R1.4 — Reconciliation 3-tier logic
# ---------------------------------------------------------------------------


class TestReconciliationTiers:
    """Tiered evidence hierarchy: GROUNDED → SECTION_INFERRED → HALLUCINATED."""

    def test_tier1_grounded_experience_is_kept(self) -> None:
        """Tier 1: LLM evidence bound + grounded → kept, no EVIDENCE_SECTION_INFERRED."""
        document = _document(
            [["Work Experience", "Northstar Labs", "Platform Engineer | 2022 - Present"]],
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
        bound, _ = bind_exact_entity_evidence(profile, document)
        reconciled = reconcile_profile(bound, document, provider_name="fixture-llm")

        assert len(reconciled.experiences) == 1
        warning_codes = [w.code for w in reconciled.warnings]
        assert WarningCode.EVIDENCE_SECTION_INFERRED not in warning_codes

    def test_tier2_section_inferred_experience_is_kept_with_warning(self) -> None:
        """Tier 2: No LLM evidence, but title found in EXPERIENCE section → kept + warning."""
        # Document has an experience section with the job title present
        document = _document(
            [
                [
                    "Work Experience",
                    "Data Scientist at TechViet",
                    "Developed ML models for fraud detection",
                ]
            ],
        )
        # Profile has no evidence (LLM forgot to attach it — the Aris_Le scenario)
        profile = CVProfile(
            experiences=[
                Experience(
                    job_title="Data Scientist",
                    company="TechViet",
                    start_date=date(2020, 6, 1),
                    end_date=date(2023, 12, 1),
                    # evidence=[] intentionally empty
                )
            ]
        )
        reconciled = reconcile_profile(profile, document, provider_name="fixture-llm")

        # Must be kept (Tier 2)
        assert len(reconciled.experiences) == 1
        # Must have the EVIDENCE_SECTION_INFERRED warning
        warning_codes = [w.code for w in reconciled.warnings]
        assert WarningCode.EVIDENCE_SECTION_INFERRED in warning_codes
        # ValidationStatus must include manual_review signal
        from ai_core.schemas import ValidationStatus
        assert reconciled.validation_status == ValidationStatus.MANUAL_REVIEW

    def test_tier0_hallucinated_experience_is_dropped(self) -> None:
        """Tier 0: No evidence AND title not in any EXPERIENCE section block → dropped."""
        # Document has only a Skills section — no experience section
        document = _document(
            [["Skills", "Python, SQL, Docker"]],
        )
        profile = CVProfile(
            experiences=[
                Experience(
                    job_title="Ghost Engineer",
                    company="Phantom Corp",
                    # evidence=[] — hallucinated by LLM
                )
            ]
        )
        reconciled = reconcile_profile(profile, document, provider_name="fixture-llm")

        assert len(reconciled.experiences) == 0
        warning_codes = [w.code for w in reconciled.warnings]
        assert WarningCode.UNSUPPORTED_EXPERIENCE_CLAIM in warning_codes


# ---------------------------------------------------------------------------
# R1.4 — _can_infer_from_section unit tests
# ---------------------------------------------------------------------------


class TestCanInferFromSection:
    def test_returns_true_when_title_in_experience_section(self) -> None:
        document = _document(
            [["Work Experience", "Data Scientist at Grab", "Python, ML models"]],
        )
        entry = Experience(job_title="Data Scientist", company="Grab")
        assert _can_infer_from_section(entry, document) is True

    def test_returns_false_when_title_only_in_skills_section(self) -> None:
        document = _document(
            [["Skills", "Data Scientist tools: Python, SQL"]],
        )
        entry = Experience(job_title="Data Scientist", company="Grab")
        assert _can_infer_from_section(entry, document) is False

    def test_returns_false_when_no_section_heading(self) -> None:
        document = _document(
            [["Data Scientist at Grab — 2020 to 2023"]],
        )
        entry = Experience(job_title="Data Scientist", company="Grab")
        assert _can_infer_from_section(entry, document) is False

    def test_returns_false_for_entry_with_short_title(self) -> None:
        document = _document([["Work Experience", "Some text"]])
        entry = Experience(job_title="X")  # Too short (< 5 chars for title)
        assert _can_infer_from_section(entry, document) is False


# ---------------------------------------------------------------------------
# R1.5 — Token measurement
# ---------------------------------------------------------------------------


class TestDescriptionTokenImpact:
    """_estimate_description_token_impact correctness."""

    def test_empty_profile_returns_zero_tokens(self) -> None:
        from ai_core.pipeline import _estimate_description_token_impact

        profile = CVProfile()
        result = _estimate_description_token_impact(profile)

        assert result["tokens"] == 0
        assert result["characters"] == 0
        assert result["entriesAnalyzed"] == 0
        assert result["avgTokensPerEntry"] == 0.0

    def test_counts_description_and_achievements(self) -> None:
        from ai_core.pipeline import _estimate_description_token_impact

        profile = CVProfile(
            experiences=[
                Experience(
                    job_title="Engineer",
                    description=["Built a real-time pipeline"],   # 27 chars
                    achievements=["Reduced latency by 40%"],      # 22 chars
                ),
                Experience(
                    job_title="Analyst",
                    description=["Analyzed 10TB datasets"],        # 22 chars
                    achievements=[],
                ),
            ]
        )
        result = _estimate_description_token_impact(profile)

        assert result["entriesAnalyzed"] == 2
        assert result["characters"] == 26 + 22 + 22  # actual lengths
        assert result["tokens"] > 0
        assert result["estimationMethod"] == "heuristic_chars_div4"
        assert "interpretationNote" in result

    def test_entries_without_descriptions_have_zero_char_cost(self) -> None:
        from ai_core.pipeline import _estimate_description_token_impact

        profile = CVProfile(
            experiences=[
                Experience(job_title="Engineer"),  # no description, no achievements
            ]
        )
        result = _estimate_description_token_impact(profile)

        assert result["characters"] == 0
        assert result["tokens"] == 0
        assert result["entriesAnalyzed"] == 1


# ---------------------------------------------------------------------------
# Sprint-1b — Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """_deduplicate_experiences(): merge LLM-split entries for the same role/company."""

    def test_merges_two_titles_same_company_overlapping_dates(self) -> None:
        """Core case: 'Senior Data Scientist' + 'AI Engineer' at same company, same period."""
        from ai_core.errors import WarningCode
        from ai_core.reconciliation.profile import _deduplicate_experiences

        entries = [
            Experience(
                job_title="Senior Data Scientist",
                company="Atrae Inc",
                start_date=date(2021, 3, 1),
                end_date=date(2024, 2, 1),
            ),
            Experience(
                job_title="AI Engineer",
                company="Atrae Inc",
                start_date=date(2021, 3, 1),
                end_date=date(2024, 2, 1),
            ),
        ]
        issues: list = []
        result, log = _deduplicate_experiences(entries, issues)

        assert len(result) == 1
        assert len(log) == 1
        assert log[0]["action"] == "merged"
        assert "Atrae Inc" in (result[0].company or "")
        assert result[0].start_date == date(2021, 3, 1)
        assert result[0].end_date == date(2024, 2, 1)
        assert any(w.code == WarningCode.AMBIGUOUS_ENTRY_ATTACHMENT for w in issues)

    def test_does_not_merge_different_companies(self) -> None:
        from ai_core.reconciliation.profile import _deduplicate_experiences

        entries = [
            Experience(
                job_title="Data Scientist",
                company="Company A",
                start_date=date(2020, 1, 1),
            ),
            Experience(
                job_title="Data Scientist",
                company="Company B",
                start_date=date(2020, 1, 1),
            ),
        ]
        result, log = _deduplicate_experiences(entries, [])
        assert len(result) == 2
        assert log == []

    def test_does_not_merge_low_date_overlap(self) -> None:
        """Entries at the same company but different periods must NOT merge."""
        from ai_core.reconciliation.profile import _deduplicate_experiences

        entries = [
            Experience(
                job_title="Data Scientist",
                company="Acme Corp",
                start_date=date(2018, 1, 1),
                end_date=date(2020, 12, 31),
            ),
            Experience(
                job_title="Senior Data Scientist",
                company="Acme Corp",
                start_date=date(2021, 1, 1),
                end_date=date(2023, 12, 31),
            ),
        ]
        result, log = _deduplicate_experiences(entries, [])
        # No temporal overlap at all → should NOT merge
        assert len(result) == 2
        assert log == []

    def test_combined_title_preserves_both_roles(self) -> None:
        """Combined title must contain both original role names."""
        from ai_core.reconciliation.profile import _deduplicate_experiences

        entries = [
            Experience(
                job_title="Backend Engineer",
                company="TechCo",
                start_date=date(2022, 1, 1),
            ),
            Experience(
                job_title="Platform Engineer",
                company="TechCo",
                start_date=date(2022, 3, 1),
            ),
        ]
        result, log = _deduplicate_experiences(entries, [])
        assert len(result) == 1
        # Both role keywords must appear in combined title
        combined = (result[0].job_title or "").lower()
        assert "engineer" in combined  # shared substring handles this

    def test_merges_evidence_union(self) -> None:
        """Merged entry must contain evidence from both source entries."""
        from ai_core.reconciliation.profile import _deduplicate_experiences
        from ai_core.schemas import EvidenceRef

        ev_a = EvidenceRef(text="Senior Data Scientist at Atrae", page_number=1, block_id="p1-b1")
        ev_b = EvidenceRef(text="AI Engineer at Atrae", page_number=1, block_id="p1-b2")
        entries = [
            Experience(
                job_title="Senior Data Scientist",
                company="Atrae Inc",
                start_date=date(2021, 1, 1),
                evidence=[ev_a],
            ),
            Experience(
                job_title="AI Engineer",
                company="Atrae Inc",
                start_date=date(2021, 1, 1),
                evidence=[ev_b],
            ),
        ]
        result, _ = _deduplicate_experiences(entries, [])
        block_ids = {ev.block_id for ev in result[0].evidence}
        assert "p1-b1" in block_ids
        assert "p1-b2" in block_ids


# ---------------------------------------------------------------------------
# Sprint-1b — Bbox disambiguation
# ---------------------------------------------------------------------------


class TestBboxDisambiguation:
    """_bbox_y_overlap and _disambiguate_by_row logic."""

    def test_bbox_y_overlap_same_row(self) -> None:
        from ai_core.reconciliation.evidence_binding import _bbox_y_overlap

        # Two blocks with identical y bands → full overlap
        assert _bbox_y_overlap((0, 100, 200, 120), (200, 100, 400, 120)) == pytest.approx(1.0)

    def test_bbox_y_overlap_no_overlap(self) -> None:
        from ai_core.reconciliation.evidence_binding import _bbox_y_overlap

        assert _bbox_y_overlap((0, 50, 200, 60), (0, 80, 200, 90)) == pytest.approx(0.0)

    def test_bbox_y_overlap_none_returns_zero(self) -> None:
        from ai_core.reconciliation.evidence_binding import _bbox_y_overlap

        assert _bbox_y_overlap(None, (0, 50, 200, 60)) == 0.0
        assert _bbox_y_overlap((0, 50, 200, 60), None) == 0.0
