"""Conservative local binding of extracted entity values to UnifiedDocument blocks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date

from ai_core.schemas import (
    CVProfile,
    EvidenceRef,
    Experience,
    Project,
    UnifiedBlock,
    UnifiedDocument,
)

_NORMALIZE = re.compile(r"\W+", re.UNICODE)


@dataclass(frozen=True)
class _BlockCandidate:
    page_number: int
    block: UnifiedBlock
    normalized_text: str


def bind_exact_entity_evidence(
    profile: CVProfile,
    document: UnifiedDocument,
) -> tuple[CVProfile, dict[str, object]]:
    """Add only strong local entity citations; leave every weak match unbound."""
    blocks = _blocks(document)
    audit_entities: list[dict[str, object]] = []
    audit: dict[str, object] = {"mode": "exact-entity", "entities": audit_entities}

    experiences: list[Experience] = []
    for index, entry in enumerate(profile.experiences):
        evidence, reason = _experience_evidence(
            entry.job_title, entry.company, entry.start_date, blocks
        )
        bound_entry = entry
        if not entry.evidence and evidence:
            bound_entry = entry.model_copy(update={"evidence": evidence})
        experiences.append(bound_entry)
        audit_entities.append(
            _audit_entry("experience", index, entry.job_title, entry.company, evidence, reason)
        )

    projects: list[Project] = []
    for index, project_entry in enumerate(profile.projects):
        evidence, reason = _project_evidence(project_entry.title, project_entry.start_date, blocks)
        bound_project = project_entry
        if not project_entry.evidence and evidence:
            bound_project = project_entry.model_copy(update={"evidence": evidence})
        projects.append(bound_project)
        audit_entities.append(
            _audit_entry(
                "project", index, project_entry.title, project_entry.role, evidence, reason
            )
        )

    return profile.model_copy(update={"experiences": experiences, "projects": projects}), audit


def _blocks(document: UnifiedDocument) -> list[_BlockCandidate]:
    return [
        _BlockCandidate(page.page_number, block, _normalized(block.text))
        for page in document.pages
        for block in page.blocks
        if _normalized(block.text)
    ]


def _experience_evidence(
    title: str | None,
    company: str | None,
    start_date: date | None,
    blocks: list[_BlockCandidate],
) -> tuple[list[EvidenceRef], str]:
    title_matches = _experience_title_matches(title, company, blocks)
    if not title_matches:
        return [], "TITLE_NOT_FOUND_EXACT"
    candidates = [
        evidence
        for title_block in title_matches
        if (evidence := _context_evidence(title_block, company, start_date, blocks)) is not None
    ]
    if not candidates:
        return [], "TITLE_WITHOUT_STRONG_CONTEXT"
    if len(candidates) > 1:
        # Sprint-1b: Try bbox y-overlap disambiguation before giving up.
        # On two-column CVs the same role title may appear once per job entry;
        # only the one whose company/date block shares the same horizontal band
        # is the correct anchor. Filter candidates by same-row context evidence.
        disambiguated = _disambiguate_by_row(candidates)
        if len(disambiguated) == 1:
            return _evidence_from_blocks(disambiguated[0]), "BOUND_EXACT_CONTEXT_ROW"
        return [], "TITLE_AMBIGUOUS"
    return _evidence_from_blocks(candidates[0]), "BOUND_EXACT_CONTEXT"


def _project_evidence(
    title: str | None,
    start_date: date | None,
    blocks: list[_BlockCandidate],
) -> tuple[list[EvidenceRef], str]:
    matches = _phrase_matches(title, blocks)
    if not matches:
        return [], "TITLE_NOT_FOUND_EXACT"
    if len(matches) == 1:
        return _evidence_from_blocks(matches), "BOUND_EXACT_UNIQUE"
    candidates = [
        evidence
        for title_block in matches
        if (evidence := _context_evidence(title_block, None, start_date, blocks)) is not None
    ]
    if len(candidates) != 1:
        return [], "TITLE_AMBIGUOUS"
    return _evidence_from_blocks(candidates[0]), "BOUND_EXACT_CONTEXT"


def _phrase_matches(value: str | None, blocks: list[_BlockCandidate]) -> list[_BlockCandidate]:
    needle = _normalized(value)
    if len(needle) < 5:
        return []
    return [candidate for candidate in blocks if needle in candidate.normalized_text]


def _experience_title_matches(
    title: str | None,
    company: str | None,
    blocks: list[_BlockCandidate],
) -> list[_BlockCandidate]:
    """Avoid treating a shorter title as a substring of a longer role title."""
    title_key = _normalized(title)
    company_key = _normalized(company)
    if len(title_key) < 5:
        return []
    accepted_prefixes = [title_key]
    if company_key:
        accepted_prefixes.extend((company_key + title_key, title_key + company_key))
    return [
        candidate
        for candidate in blocks
        if _is_title_anchor(candidate.normalized_text, title_key)
        or any(candidate.normalized_text.startswith(prefix) for prefix in accepted_prefixes[1:])
    ]


def _is_title_anchor(block_text: str, title_key: str) -> bool:
    if block_text == title_key:
        return True
    suffix = block_text.removeprefix(title_key)
    return bool(suffix and suffix[0].isdigit())


# Sprint-1 (2026-08-10): Context window expanded from ±2 → ±4 to handle two-column
# CVs where company/date block and role/title block may have reading_order gap > 2.
# Matching logic changed from strict AND to OR: either a company match or a year match
# is sufficient context to anchor an entity, preventing false drops on CVs where one
# context signal is present in the same row but the other is on a separate line.
# Sprint-1b (2026-08-10): Added bbox y-overlap disambiguation to resolve TITLE_AMBIGUOUS
# on two-column CVs. When multiple title anchors are found, only those whose context
# block (company or date) shares the same horizontal band (y-overlap) are kept.
_CONTEXT_WINDOW = 4  # reading_order distance threshold (inclusive)

# Minimum fractional overlap required for two blocks to be considered on the same row.
# 0.3 = their vertical bands must overlap by at least 30% of the shorter block height.
_BBOX_Y_OVERLAP_MIN = 0.3


def _bbox_y_overlap(bbox_a: tuple[float, float, float, float] | None,
                    bbox_b: tuple[float, float, float, float] | None) -> float:
    """Return the fractional vertical overlap of two bboxes (x0, y0, x1, y1).

    Returns a value in [0, 1]: 0 = no vertical overlap, 1 = one fully contains
    the other.  Returns 0 if either bbox is None (no geometry available).
    """
    if bbox_a is None or bbox_b is None:
        return 0.0
    _, y0_a, _, y1_a = bbox_a
    _, y0_b, _, y1_b = bbox_b
    overlap = max(0.0, min(y1_a, y1_b) - max(y0_a, y0_b))
    if overlap == 0.0:
        return 0.0
    shorter = min(y1_a - y0_a, y1_b - y0_b)
    return overlap / shorter if shorter > 0 else 0.0


def _context_evidence(
    title_block: _BlockCandidate,
    company: str | None,
    start_date: date | None,
    blocks: list[_BlockCandidate],
) -> list[_BlockCandidate] | None:
    """Return only a unique title anchor and its local context blocks.

    A context block must appear on the same page within _CONTEXT_WINDOW reading
    positions of the title anchor.  Either a company match OR a year match is
    sufficient — both are not required — so that two-column layouts where the
    company sits in a separate column are still correctly anchored.

    Returns None when no context signal is found at all (both company key and
    year are provided but neither appears in the local window).
    """
    company_key = _normalized(company)
    year = str(start_date.year) if start_date else None
    nearby = [
        candidate
        for candidate in blocks
        if candidate.page_number == title_block.page_number
        and abs(candidate.block.reading_order - title_block.block.reading_order) <= _CONTEXT_WINDOW
    ]
    company_blocks = (
        [title_block]
        if company_key and company_key in title_block.normalized_text
        else [
            candidate
            for candidate in nearby
            if company_key and company_key in candidate.normalized_text
        ]
    )
    year_blocks = (
        [title_block]
        if year and year in title_block.normalized_text
        else [candidate for candidate in nearby if year and year in candidate.normalized_text]
    )
    # If neither company nor year is provided, we cannot anchor uniquely.
    if not company_key and not year:
        return None
    # OR logic: at least one context signal must be satisfied.
    has_company = bool(company_blocks) or not company_key
    has_year = bool(year_blocks) or not year
    if not has_company and not has_year:
        return None
    # Build evidence from whichever signals were found.
    context: list[_BlockCandidate] = [title_block]
    if company_blocks and company_blocks != [title_block]:
        context.extend(company_blocks)
    if year_blocks and year_blocks != [title_block]:
        context.extend(year_blocks)
    return context


def _disambiguate_by_row(
    candidates: list[list[_BlockCandidate]],
) -> list[list[_BlockCandidate]]:
    """Keep only those candidate evidence sets whose context block is on the same
    horizontal row as the title anchor block (bbox y-overlap >= _BBOX_Y_OVERLAP_MIN).

    Sprint-1b (2026-08-10): Resolves TITLE_AMBIGUOUS on two-column CVs where the
    same job title text appears multiple times.  The company or date block in the
    correct entry shares the same horizontal band; entries from other jobs do not.
    Falls back to the full candidate list if no bbox geometry is available.
    """
    filtered: list[list[_BlockCandidate]] = []
    for evidence_set in candidates:
        if len(evidence_set) < 2:
            # Title-only anchor — no context block to compare row with.
            filtered.append(evidence_set)
            continue
        title_block = evidence_set[0]
        context_blocks = evidence_set[1:]
        # Accept this candidate if ANY context block shares the title's horizontal row.
        same_row = any(
            _bbox_y_overlap(title_block.block.bbox, ctx.block.bbox) >= _BBOX_Y_OVERLAP_MIN
            for ctx in context_blocks
        )
        # If no bbox data at all, treat as row-sharing (degrade gracefully).
        no_geometry = all(ctx.block.bbox is None for ctx in context_blocks)
        if same_row or no_geometry:
            filtered.append(evidence_set)
    return filtered if filtered else candidates


def _evidence_from_blocks(blocks: list[_BlockCandidate]) -> list[EvidenceRef]:
    seen: set[tuple[int, str]] = set()
    result: list[EvidenceRef] = []
    for candidate in blocks:
        key = (candidate.page_number, candidate.block.id)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            EvidenceRef(
                text=candidate.block.text,
                page_number=candidate.page_number,
                block_id=candidate.block.id,
            )
        )
    return result


def _audit_entry(
    kind: str,
    index: int,
    primary: str | None,
    secondary: str | None,
    evidence: list[EvidenceRef],
    reason: str,
) -> dict[str, object]:
    raw = "\x1f".join((kind, primary or "", secondary or ""))
    return {
        "entityKind": kind,
        "entryIndex": index,
        "identityFingerprint": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
        "decision": "bound" if evidence else "unbound",
        "reasonCode": reason,
        "bindingVersion": "v3",  # Sprint-1b (2026-08-10): +bbox y-overlap disambiguation
        "evidenceReferences": [
            {"pageNumber": item.page_number, "blockId": item.block_id} for item in evidence
        ],
    }


def _normalized(value: str | None) -> str:
    return _NORMALIZE.sub("", value.casefold()) if value else ""
