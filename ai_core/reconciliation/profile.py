"""Evidence grounding, normalization and cross-field profile validation."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from hashlib import sha256

from ai_core.errors import Issue, WarningCode
from ai_core.extraction.profile import canonical_skill_mentions, canonical_skill_name
from ai_core.schemas import (
    BlockType,
    CVProfile,
    DiagnosticStatus,
    Education,
    EvidenceRef,
    EvidenceType,
    Experience,
    Project,
    SectionKind,
    Skill,
    UnifiedBlock,
    UnifiedDocument,
    ValidationStatus,
)
from ai_core.scoring.features import document_evidence_types_for_skill, document_score_features
from ai_core.sections import detect_sections, heading_kind

_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LOCATION_WORDS = {
    "sydney",
    "australia",
    "singapore",
    "vietnam",
    "việt nam",
    "ho chi minh city",
    "hanoi",
    "hà nội",
    "da nang",
    "đà nẵng",
    "dublin",
    "ireland",
}


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFC", _CONTROL.sub("", _INVISIBLE.sub("", value)))
    return re.sub(r"\s+", " ", normalized).strip() or None


def _looks_like_location(value: str) -> bool:
    key = value.casefold().strip()
    parts = {part.strip() for part in re.split(r"[,|/]", key)}
    return key in _LOCATION_WORDS or (
        len(parts) >= 2 and sum(part in _LOCATION_WORDS for part in parts) >= 2
    )


def _normalized_identity(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


_INVALID_NAME_PATTERN = re.compile(
    r"\b(graduated|bachelor|master|degree|engineer|developer|architect|analyst|intern|fresher|scientist)\b",
    re.I,
)


def _is_invalid_name(name: str) -> bool:
    clean = clean_text(name)
    if not clean:
        return True
    if _looks_like_location(clean):
        return True
    return bool(_INVALID_NAME_PATTERN.search(clean))


def _name_with_evidence_casing(profile: CVProfile, document: UnifiedDocument) -> str | None:
    """Keep the CV's original name spelling when the model returns an equivalent name."""

    raw_name = clean_text(profile.candidate_name)
    invalid_tokens = {"Candidate Name", "Alex Morgan", "UNKNOWN CANDIDATE"}

    if raw_name and raw_name not in invalid_tokens:
        if _is_invalid_name(raw_name):
            return None
        candidate_name: str | None = raw_name
    else:
        candidate_name = None
        for page in document.pages:
            for block in page.blocks:
                txt = clean_text(block.text)
                if not txt or txt.startswith("##") or "@" in txt or "+" in txt:
                    continue
                words = txt.split()
                if (
                    2 <= len(words) <= 5
                    and txt.replace(" ", "").replace(",", "").isalpha()
                    and not _is_invalid_name(txt)
                ):
                    if "," in txt:
                        parts = [p.strip() for p in txt.split(",") if p.strip()]
                        txt = " ".join(reversed(parts))
                    candidate_name = txt
                    break
            if candidate_name:
                break

    if not candidate_name:
        return None
    for evidence in profile.field_evidence.get("candidateName", []):
        evidence_name = clean_text(evidence.text)
        if (
            evidence_name
            and _normalized_identity(evidence_name) == _normalized_identity(candidate_name)
            and _grounded(evidence, document)
        ):
            return evidence_name
    return candidate_name


def _block_evidence_types(document: UnifiedDocument) -> list[tuple[str, EvidenceType]]:
    """Classify explicit skill mentions by the nearest recovered document section."""

    result: list[tuple[str, EvidenceType]] = []
    current: EvidenceType | None = None
    for page in document.pages:
        for block in page.blocks:
            text = clean_text(block.text) or ""
            key = text.casefold()
            if any(marker in key for marker in ("research", "publication", "paper", "journal")):
                current = EvidenceType.RESEARCH
            elif any(
                marker in key
                for marker in ("work experience", "professional experience", "employment")
            ):
                current = EvidenceType.WORK_EXPERIENCE
            if current is not None:
                result.append((text, current))
    return result


def _grounded(evidence: EvidenceRef, document: UnifiedDocument) -> bool:
    needle = clean_text(evidence.text)
    for page in document.pages:
        if page.page_number != evidence.page_number:
            continue
        for block in page.blocks:
            if block.id == evidence.block_id:
                return bool(needle and needle in (clean_text(block.text) or ""))
    return False


def _evidence_contains(evidence: list[EvidenceRef], value: str | None) -> bool:
    """Require an extracted entity value to be explicitly supported by its evidence."""

    target = clean_text(value)
    if not target:
        return True
    return any(target.casefold() in (clean_text(item.text) or "").casefold() for item in evidence)


def _evidence_section_kinds(
    evidence: list[EvidenceRef],
    document: UnifiedDocument,
) -> set[SectionKind]:
    """Return recovered section kinds for the blocks cited by one entity."""

    result: set[SectionKind] = set()
    evidence_ids = {(item.page_number, item.block_id) for item in evidence}
    for page in document.pages:
        current = None
        for block in page.blocks:
            heading = heading_kind(block.text)
            if heading is not None:
                current = heading
            if (page.page_number, block.id) in evidence_ids and current is not None:
                result.add(current)
    return result


_SECONDARY_ROLE_PATTERN = re.compile(
    r"\b(instructor|mentor|teaching assistant|tutor|curriculum designer)\b",
    re.I,
)


def _experience_years(profile: CVProfile) -> float | None:
    primary_entries = [
        entry
        for entry in profile.experiences
        if not (
            _SECONDARY_ROLE_PATTERN.search(entry.job_title or "")
            or _SECONDARY_ROLE_PATTERN.search(entry.company or "")
        )
    ]
    target_entries = primary_entries if primary_entries else profile.experiences

    intervals = sorted(
        (entry.start_date, entry.end_date or date.today())
        for entry in target_entries
        if entry.start_date is not None
    )
    if not intervals:
        return None
    merged: list[tuple[date, date]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return round(sum((end - start).days for start, end in merged) / 365.25, 1)


# ---------------------------------------------------------------------------
# Sprint-1b (2026-08-10): Experience deduplication
# ---------------------------------------------------------------------------

_TITLE_SEPARATORS = re.compile(r"\s*[-/|&,]\s*|\s+(?:and|&|và)\s+", re.I)

_TITLE_NOISE = re.compile(
    r"\b(senior|junior|lead|principal|staff|associate|assistant|"
    r"intern|manager|director|head\s+of)\b",
    re.I,
)


def _normalize_title(title: str | None) -> str:
    """Lower-case, strip noise words, then normalize whitespace."""
    if not title:
        return ""
    t = unicodedata.normalize("NFC", title).casefold()
    t = _TITLE_NOISE.sub("", t)
    return " ".join(t.split())


def _normalize_company(company: str | None) -> str:
    return unicodedata.normalize("NFC", (company or "")).casefold().strip()


def _date_overlap_fraction(
    start_a: date | None, end_a: date | None,
    start_b: date | None, end_b: date | None,
) -> float:
    """Fraction of the shorter interval that overlaps the longer one.

    Returns 0 if either interval has no start_date (cannot measure).
    """
    today = date.today()
    if start_a is None or start_b is None:
        return 0.0
    a0, a1 = start_a, end_a or today
    b0, b1 = start_b, end_b or today
    overlap_days = max(0, (min(a1, b1) - max(a0, b0)).days)
    if overlap_days == 0:
        return 0.0
    shorter = min((a1 - a0).days, (b1 - b0).days)
    return overlap_days / shorter if shorter > 0 else 0.0


def _title_overlap(title_a: str | None, title_b: str | None) -> bool:
    """True if one normalized title contains the other (substring match)."""
    a = _normalize_title(title_a)
    b = _normalize_title(title_b)
    if not a or not b:
        return False
    return a in b or b in a


# Professional domain words used to detect cross-title keyword overlap.
# When both titles share a domain word (e.g. "engineer", "data"), entries at
# the same company in the same period are likely an LLM-split of one role.
_DOMAIN_WORDS = re.compile(
    r"\b(data|engineer|scientist|analyst|developer|architect|researcher|"
    r"ml|ai|nlp|backend|frontend|fullstack|platform|cloud|devops|sre|"
    r"k\u1ef9|s\u01b0|nh\u00e0|l\u1eadp|tr\u00ecnh|ph\u00e2n|t\u00edch)\b",
    re.I | re.UNICODE,
)


def _title_has_shared_words(title_a: str | None, title_b: str | None) -> bool:
    """True if both titles share at least one professional domain keyword."""
    if not title_a or not title_b:
        return False
    words_a = set(_DOMAIN_WORDS.findall(title_a.casefold()))
    words_b = set(_DOMAIN_WORDS.findall(title_b.casefold()))
    return bool(words_a & words_b)


def _merge_two_experiences(primary: Experience, duplicate: Experience) -> Experience:
    """Merge `duplicate` into `primary`, keeping the longer/combined job_title,
    earliest start_date, latest end_date, and union of evidence + descriptions.
    """
    # Combined title: "Senior Data Scientist / AI Engineer" style
    t_a = (primary.job_title or "").strip()
    t_b = (duplicate.job_title or "").strip()
    if _normalize_title(t_a) in _normalize_title(t_b):
        combined_title = t_b  # b is more specific or equal
    elif _normalize_title(t_b) in _normalize_title(t_a):
        combined_title = t_a  # a is more specific
    else:
        combined_title = f"{t_a} / {t_b}" if t_a and t_b else (t_a or t_b)

    start_date = min(
        (d for d in (primary.start_date, duplicate.start_date) if d is not None),
        default=None,
    )
    today = date.today()
    end_a = primary.end_date or (today if primary.is_current else None)
    end_b = duplicate.end_date or (today if duplicate.is_current else None)
    end_dates = [d for d in (end_a, end_b) if d is not None]
    end_date = max(end_dates) if end_dates else None
    is_current = primary.is_current or duplicate.is_current
    if is_current:
        end_date = None

    # Merge evidence: deduplicate by block_id
    seen_block_ids: set[str] = set()
    merged_evidence = []
    for ev in list(primary.evidence) + list(duplicate.evidence):
        if ev.block_id not in seen_block_ids:
            seen_block_ids.add(ev.block_id)
            merged_evidence.append(ev)

    merged_desc = list(primary.description) + [
        d for d in duplicate.description if d not in primary.description
    ]
    merged_ach = list(primary.achievements) + [
        a for a in duplicate.achievements if a not in primary.achievements
    ]
    merged_skills = list(primary.skills) + [
        s for s in duplicate.skills if s not in primary.skills
    ]

    return primary.model_copy(
        update={
            "job_title": combined_title,
            "start_date": start_date,
            "end_date": end_date,
            "is_current": is_current,
            "evidence": merged_evidence,
            "description": merged_desc,
            "achievements": merged_ach,
            "skills": merged_skills,
            "confidence": min(primary.confidence, duplicate.confidence),
        }
    )


def _deduplicate_experiences(
    experiences: list[Experience],
    issues: list[Issue],
    *,
    date_overlap_threshold: float = 0.8,
) -> tuple[list[Experience], list[dict[str, object]]]:
    """Merge duplicate experience entries produced by LLM over-splitting.

    A pair (A, B) is considered a duplicate when:
      - They have the **same company** (normalized, non-empty), AND
      - Their date intervals overlap by at least `date_overlap_threshold`
        fraction of the shorter interval, AND
      - Their job titles overlap (one is a substring of the other after
        stripping noise words like "Senior", "Lead", etc.)

    OR:
      - Company is the same AND both have no start_date AND title overlaps.

    The primary entry (earlier start_date, or first in list) absorbs the
    duplicate.  The merged entry gets a combined job_title
    ("Senior Data Scientist / AI Engineer") to preserve both roles.

    Sprint-1b (2026-08-10): Fixes 9-entry count on Aris_Le CV caused by
    LLM splitting "Senior Data Scientist - AI Engineer at Atrae Inc" into
    two separate entries for the same company/period.
    """
    if len(experiences) <= 1:
        return experiences, []

    merge_log: list[dict[str, object]] = []
    remaining = list(experiences)
    result: list[Experience] = []

    while remaining:
        current = remaining.pop(0)
        company_key = _normalize_company(current.company)
        duplicates_found: list[int] = []

        for i, candidate in enumerate(remaining):
            cand_company = _normalize_company(candidate.company)

            # Both must have the same non-empty company to merge.
            if not company_key or company_key != cand_company:
                continue

            has_start = current.start_date is not None and candidate.start_date is not None
            if has_start:
                overlap = _date_overlap_fraction(
                    current.start_date, current.end_date,
                    candidate.start_date, candidate.end_date,
                )
                if overlap < date_overlap_threshold:
                    continue
                if overlap >= 1.0:
                    # Exact same period at same company: always merge regardless of title.
                    # Handles the Aris_Le case where LLM splits
                    # "Senior Data Scientist - AI Engineer" into two separate entries.
                    duplicates_found.append(i)
                    continue
            # Partial overlap or no dates: require title or shared-domain signal.
            title_signal = (
                _title_overlap(current.job_title, candidate.job_title)
                or _title_has_shared_words(current.job_title, candidate.job_title)
            )
            if not title_signal:
                continue

            duplicates_found.append(i)

        if duplicates_found:
            # Merge in reverse index order so pop() indices stay valid
            for i in sorted(duplicates_found, reverse=True):
                dup = remaining.pop(i)
                merge_log.append(
                    {
                        "action": "merged",
                        "primaryTitle": current.job_title,
                        "duplicateTitle": dup.job_title,
                        "company": current.company,
                        "dateOverlap": round(
                            _date_overlap_fraction(
                                current.start_date, current.end_date,
                                dup.start_date, dup.end_date,
                            ),
                            3,
                        ),
                        "mergedTitle": None,  # filled after merge
                    }
                )
                current = _merge_two_experiences(current, dup)
                merge_log[-1]["mergedTitle"] = current.job_title

            issues.append(
                Issue(
                    code=WarningCode.AMBIGUOUS_ENTRY_ATTACHMENT,
                    message=(
                        f"Merged {len(duplicates_found)} duplicate experience "
                        f"entry/entries for '{current.company}' into '{current.job_title}'."
                    ),
                    stage="deduplication",
                    details={
                        "company": current.company or "",
                        "mergedCount": len(duplicates_found),
                        "mergedTitle": current.job_title,
                    },
                )
            )

        result.append(current)

    return result, merge_log

# ---------------------------------------------------------------------------
# Sprint-1 (2026-08-10): Section-inferred fallback
# ---------------------------------------------------------------------------
_EXPERIENCE_SECTION_STOP_KINDS = {
    SectionKind.EDUCATION,
    SectionKind.PROJECTS,
    SectionKind.SKILLS,
    SectionKind.CERTIFICATIONS,
    SectionKind.LANGUAGES,
    SectionKind.SUMMARY,
}


def _can_infer_from_section(entry: Experience, document: UnifiedDocument) -> bool:
    """Tier-2 soft fallback: title or company text appears inside an EXPERIENCE section.

    This is a conservative scan over the raw document blocks.  It only returns
    True when an EXPERIENCE (or equivalent) section heading has been seen AND the
    entry's job_title or company text is literally present in a subsequent block
    before the next major section boundary.  It never fabricates evidence.
    """
    title_key = (entry.job_title or "").casefold().strip()
    company_key = (entry.company or "").casefold().strip()
    if not title_key and not company_key:
        return False

    in_experience_section = False
    for page in document.pages:
        for block in page.blocks:
            section_kind = heading_kind(block.text)
            if section_kind == SectionKind.EXPERIENCE:
                in_experience_section = True
                continue
            if section_kind in _EXPERIENCE_SECTION_STOP_KINDS:
                in_experience_section = False
                continue
            if not in_experience_section:
                continue
            block_text = (clean_text(block.text) or "").casefold()
            if (title_key and len(title_key) >= 5 and title_key in block_text) or (
                company_key and len(company_key) >= 4 and company_key in block_text
            ):
                return True
    return False


_IMPACT_PATTERN = re.compile(
    r"\b(processed|supported|managed|handled|improved|reduced|increased|grew|"
    r"saved|built|developed|optimized|tăng|giảm|cải thiện|quản lý|đạt)\b",
    re.I,
)
_QUANTITY_PATTERN = re.compile(
    r"(\b\d+(?:[.,]\d+)?\s*%|\b\d+(?:[.,]\d+)?\s*(?:[kmb]\b|\+|/|\s*(?:users|records|events|requests|queries|pages|percent|ms|seconds|minutes|hours|days|year|years|members|people|projects)))",
    re.I,
)
_EXCLUDE_PATTERN = re.compile(
    r"(\+?\d[\d\s-]{8,}|\b(19|20)\d{2}\b|\bv?\d+\.\d+(\.\d+)?\b)",
    re.I,
)


def _separate_quantified_achievements(
    descriptions: list[str],
    existing_achievements: list[str],
) -> tuple[list[str], list[str]]:
    remaining_desc: list[str] = []
    achievements: list[str] = list(existing_achievements)
    for desc in descriptions:
        text = desc.strip()
        if not text:
            continue
        if (
            _IMPACT_PATTERN.search(text)
            and _QUANTITY_PATTERN.search(text)
            and not _EXCLUDE_PATTERN.search(text)
            and text not in achievements
        ):
            achievements.append(text)
        else:
            remaining_desc.append(text)
    return remaining_desc, achievements


def _project_date_is_grounded(value: date | None, evidence: list[EvidenceRef]) -> bool:
    if value is None:
        return True
    year = str(value.year)
    return any(year in evidence_ref.text for evidence_ref in evidence)


def _recover_local_education(document: UnifiedDocument) -> list[Education]:
    """Recover education from original document blocks if LLM missed it."""

    education_blocks: list[tuple[int, UnifiedBlock]] = []
    in_education = False

    for page in document.pages:
        for block in page.blocks:
            text = clean_text(block.text) or ""
            key = text.casefold()
            if any(
                marker in key
                for marker in (
                    "education",
                    "học vấn",
                    "trình độ học vấn",
                    "bằng cấp",
                    "học tập",
                )
            ):
                in_education = True
                continue
            if in_education:
                if block.type == BlockType.SECTION_HEADER or any(
                    marker in key
                    for marker in (
                        "experience",
                        "projects",
                        "skills",
                        "work",
                        "kinh nghiệm",
                        "kỹ năng",
                        "dự án",
                        "certifications",
                    )
                ):
                    in_education = False
                    continue
                education_blocks.append((page.page_number, block))

    if not education_blocks:
        return []

    institution: str | None = None
    field_of_study: str | None = None
    degree: str | None = None
    evidence_refs: list[EvidenceRef] = []

    for page_num, block in education_blocks:
        raw_text = clean_text(block.text)
        if raw_text is None:
            continue
        clean_name = raw_text.removeprefix("##").strip()

        if not institution and any(
            kw in clean_name.casefold()
            for kw in (
                "university",
                "trường",
                "đại học",
                "học viện",
                "institute",
                "college",
                "academy",
            )
        ):
            institution = clean_name
            evidence_refs.append(
                EvidenceRef(text=clean_name, page_number=page_num, block_id=block.id)
            )
        elif any(
            kw in clean_name.casefold()
            for kw in (
                "graduated",
                "computer science",
                "công nghệ thông tin",
                "bachelor",
                "master",
                "engineer",
                "kỹ sư",
                "cử nhân",
                "chuyên ngành",
            )
        ):
            if "graduated in" in clean_name.casefold():
                field_of_study = clean_name.split("in", 1)[-1].strip()
            elif "chuyên ngành:" in clean_name.casefold():
                field_of_study = clean_name.split("chuyên ngành:", 1)[-1].strip()
            elif not field_of_study:
                field_of_study = clean_name
            evidence_refs.append(
                EvidenceRef(text=clean_name, page_number=page_num, block_id=block.id)
            )

    if institution or field_of_study:
        return [
            Education(
                institution=institution or "Unspecified Institution",
                degree=degree,
                field_of_study=field_of_study,
                confidence=0.9,
                evidence=evidence_refs,
            )
        ]
    return []


def _recover_local_experience(document: UnifiedDocument) -> list[Experience]:
    """Fallback local experience recovery when LLM extraction misses experience entries."""
    from ai_core.extraction.profile import _experience
    raw_text = document.markdown or ""
    if not raw_text and document.pages:
        raw_text = "\n".join(
            block.text for page in document.pages for block in page.blocks if block.text
        )
    if not raw_text:
        return []
    sections = detect_sections(raw_text)
    recovered, _ = _experience(sections)
    return recovered


def _validate_experiences(
    profile: CVProfile,
    document: UnifiedDocument,
    issues: list[Issue],
    dispositions: list[dict[str, object]] | None = None,
) -> tuple[list[Experience], bool]:
    """Keep employment claims using a three-tier evidence hierarchy.

    Tier 1 — GROUNDED: LLM returned evidence that is verified in the document.
    Tier 2 — SECTION_INFERRED: No LLM evidence, but entity text is found inside
             an EXPERIENCE section block.  Kept with a WARNING; manual_review set.
    Tier 0 — HALLUCINATED: Neither grounded evidence nor section inference.
             Dropped with UNSUPPORTED_EXPERIENCE_CLAIM warning.

    Sprint-1 (2026-08-10): Tier-2 added to recover entries whose evidence was
    omitted by the LLM (e.g. Flash Lite in baseline evidence_mode).
    """
    verified: list[Experience] = []
    unsupported_claim = False
    for index, entry in enumerate(profile.experiences):
        evidence = entry.evidence
        grounded = bool(evidence) and all(_grounded(item, document) for item in evidence)
        sections = _evidence_section_kinds(evidence, document)
        role_supported = _evidence_contains(evidence, entry.job_title)
        project_only = bool(sections) and all(section.value == "projects" for section in sections)
        disposition = _entity_disposition(
            "experience",
            index,
            entry.job_title,
            entry.company,
            evidence,
            grounded=grounded,
            value_supported=role_supported,
            project_only_evidence=project_only,
            section_kinds=sections,
        )

        if grounded and role_supported and not project_only:
            # --- Tier 1: GROUNDED ---
            disposition["decision"] = "kept"
            disposition["reasonCode"] = "VERIFIED"
            disposition["reconciliationTier"] = 1

        elif _can_infer_from_section(entry, document):
            # --- Tier 2: SECTION_INFERRED ---
            # Entity text found in document EXPERIENCE section but LLM did not
            # attach evidence.  We keep it but flag for manual review.
            disposition["decision"] = "kept"
            disposition["reasonCode"] = "SECTION_INFERRED"
            disposition["reconciliationTier"] = 2
            if dispositions is not None:
                dispositions.append(disposition)
            issues.append(
                Issue(
                    code=WarningCode.EVIDENCE_SECTION_INFERRED,
                    message=(
                        "Experience claim kept via section-inferred fallback; "
                        "LLM did not return evidence. Manual review recommended."
                    ),
                    stage="hybrid_validation",
                    details={
                        "entryIndex": index,
                        "jobTitle": entry.job_title,
                        "company": entry.company or "",
                    },
                )
            )
            company = entry.company if _evidence_contains(evidence, entry.company) else entry.company
            verified.append(
                entry.model_copy(
                    update={
                        "company": company,
                        "is_current": entry.is_current,
                    }
                )
            )
            continue

        else:
            # --- Tier 0: HALLUCINATED ---
            disposition["decision"] = "dropped"
            disposition["reconciliationTier"] = 0
            disposition["reasonCode"] = (
                "EVIDENCE_NOT_GROUNDED"
                if not grounded
                else "ROLE_NOT_SUPPORTED_BY_EVIDENCE"
                if not role_supported
                else "PROJECT_ONLY_EVIDENCE"
            )
            if dispositions is not None:
                dispositions.append(disposition)
            unsupported_claim = True
            issues.append(
                Issue(
                    code=WarningCode.UNSUPPORTED_EXPERIENCE_CLAIM,
                    message="Employment claim lacks direct non-project evidence and was excluded.",
                    stage="hybrid_validation",
                    details={
                        "entryIndex": index,
                        "hasEvidence": bool(evidence),
                        "roleSupported": role_supported,
                        "projectOnlyEvidence": project_only,
                        "reconciliationTier": 0,
                    },
                )
            )
            continue

        # --- Attribute cleanup for Tier 1 ---
        company = entry.company if _evidence_contains(evidence, entry.company) else None
        location = entry.location if _evidence_contains(evidence, entry.location) else None
        start_date = entry.start_date
        end_date = entry.end_date
        if start_date and not _evidence_contains(evidence, str(start_date.year)):
            start_date = None
        if end_date and not _evidence_contains(evidence, str(end_date.year)):
            end_date = None
        attributes_changed = (
            company != entry.company
            or location != entry.location
            or start_date != entry.start_date
            or end_date != entry.end_date
        )
        if attributes_changed:
            issues.append(
                Issue(
                    code=WarningCode.AMBIGUOUS_ENTRY_ATTACHMENT,
                    message="Unsupported employment attributes were cleared instead of inferred.",
                    stage="hybrid_validation",
                    details={"entryIndex": index},
                )
            )
        disposition["clearedAttributes"] = [
            name
            for name, original, retained in (
                ("company", entry.company, company),
                ("location", entry.location, location),
                ("startDate", entry.start_date, start_date),
                ("endDate", entry.end_date, end_date),
            )
            if original != retained
        ]
        if dispositions is not None:
            dispositions.append(disposition)
        verified.append(
            entry.model_copy(
                update={
                    "company": company,
                    "location": location,
                    "start_date": start_date,
                    "end_date": end_date,
                    "is_current": entry.is_current and end_date is None,
                }
            )
        )
    return verified, unsupported_claim


def _validate_projects(
    profile: CVProfile,
    document: UnifiedDocument,
    issues: list[Issue],
    dispositions: list[dict[str, object]] | None = None,
) -> tuple[list[Project], bool]:
    """Keep only project claims whose title is directly supported by evidence."""

    verified: list[Project] = []
    unsupported_claim = False
    for index, entry in enumerate(profile.projects):
        evidence = entry.evidence
        grounded = bool(evidence) and all(_grounded(item, document) for item in evidence)
        title_supported = _evidence_contains(evidence, entry.title)
        disposition = _entity_disposition(
            "project",
            index,
            entry.title,
            entry.role,
            evidence,
            grounded=grounded,
            value_supported=title_supported,
            project_only_evidence=False,
            section_kinds=_evidence_section_kinds(evidence, document),
        )
        if not grounded or not title_supported:
            disposition["decision"] = "dropped"
            disposition["reasonCode"] = (
                "EVIDENCE_NOT_GROUNDED" if not grounded else "TITLE_NOT_SUPPORTED_BY_EVIDENCE"
            )
            if dispositions is not None:
                dispositions.append(disposition)
            unsupported_claim = True
            issues.append(
                Issue(
                    code=WarningCode.UNSUPPORTED_PROJECT_CLAIM,
                    message="Project claim lacks direct grounded title evidence and was excluded.",
                    stage="hybrid_validation",
                    details={
                        "entryIndex": index,
                        "hasEvidence": bool(evidence),
                        "titleSupported": title_supported,
                    },
                )
            )
            continue
        disposition["decision"] = "kept"
        disposition["reasonCode"] = "VERIFIED"
        if dispositions is not None:
            dispositions.append(disposition)
        verified.append(entry)
    return verified, unsupported_claim


def _has_section(document: UnifiedDocument, section: SectionKind) -> bool:
    return any(
        heading_kind(block.text) == section for page in document.pages for block in page.blocks
    )


def _entity_disposition(
    kind: str,
    index: int,
    primary: str | None,
    secondary: str | None,
    evidence: list[EvidenceRef],
    *,
    grounded: bool,
    value_supported: bool,
    project_only_evidence: bool,
    section_kinds: set[SectionKind],
) -> dict[str, object]:
    """Capture local benchmark diagnostics without retaining entity text."""
    fingerprint_input = "\x1f".join((kind, primary or "", secondary or ""))
    return {
        "entityKind": kind,
        "entryIndex": index,
        "identityFingerprint": sha256(fingerprint_input.encode("utf-8")).hexdigest()[:16],
        "evidence": {
            "referenceCount": len(evidence),
            "references": [
                {"pageNumber": item.page_number, "blockId": item.block_id} for item in evidence
            ],
            "grounded": grounded,
            "valueSupported": value_supported,
            "projectOnlyEvidence": project_only_evidence,
            "sectionKinds": sorted(section.value for section in section_kinds),
        },
    }


def reconcile_profile(
    profile: CVProfile,
    document: UnifiedDocument,
    *,
    provider_name: str,
    allow_identity_fallback: bool = True,
    entity_dispositions: list[dict[str, object]] | None = None,
) -> CVProfile:
    issues = list(profile.warnings)
    raw_name = clean_text(profile.candidate_name)
    manual_review = document.diagnostics.status != DiagnosticStatus.USABLE

    if raw_name and _looks_like_location(raw_name):
        issues.append(
            Issue(
                code=WarningCode.CANDIDATE_NAME_IS_LOCATION,
                message="Candidate name resembles a geographic location.",
                stage="hybrid_validation",
            )
        )
        candidate_name = None
        manual_review = True
    elif raw_name or allow_identity_fallback:
        candidate_name = _name_with_evidence_casing(profile, document)
    else:
        candidate_name = None

    headline = clean_text(profile.headline)
    if headline and (len(headline) > 100 or "\n" in headline or headline == candidate_name):
        headline = None

    evidence_values: list[EvidenceRef] = [
        evidence for values in profile.field_evidence.values() for evidence in values
    ]
    for experience_entry in profile.experiences:
        evidence_values.extend(experience_entry.evidence)
    for education_entry in profile.education:
        evidence_values.extend(education_entry.evidence)
    for project in profile.projects:
        evidence_values.extend(project.evidence)
    for award in profile.honors_awards:
        evidence_values.extend(award.evidence)
    for language in profile.language_proficiencies:
        evidence_values.extend(language.evidence)
    ungrounded = [evidence for evidence in evidence_values if not _grounded(evidence, document)]
    name_evidence = profile.field_evidence.get("candidateName", [])
    if ungrounded:
        issues.append(
            Issue(
                code=WarningCode.EVIDENCE_NOT_GROUNDED,
                message="One or more evidence references are not grounded in the document.",
                stage="hybrid_validation",
                details={"count": len(ungrounded)},
            )
        )
        if any(not _grounded(ev, document) for ev in name_evidence):
            manual_review = True

    evidence_validated_experiences, unsupported_experience = _validate_experiences(
        profile,
        document,
        issues,
        entity_dispositions,
    )
    if unsupported_experience:
        manual_review = True
    # Sprint-1 (2026-08-10): Tier-2 section-inferred entries also trigger manual review
    # because they were kept without LLM-grounded evidence.
    if any(w.code == WarningCode.EVIDENCE_SECTION_INFERRED for w in issues):
        manual_review = True

    # Sprint-1b (2026-08-10): Deduplicate LLM over-split experience entries.
    # Must run AFTER validation so only verified/tier-2 entries are deduped.
    evidence_validated_experiences, dedup_log = _deduplicate_experiences(
        evidence_validated_experiences, issues
    )
    if dedup_log and entity_dispositions is not None:
        entity_dispositions.extend(
            {"entityKind": "experience_dedup", **entry} for entry in dedup_log
        )

    evidence_validated_projects, unsupported_project = _validate_projects(
        profile,
        document,
        issues,
        entity_dispositions,
    )
    if unsupported_project:
        manual_review = True

    experiences = []
    for entry in evidence_validated_experiences:
        cleaned_desc = [
            cleaned for value in entry.description if (cleaned := clean_text(value)) is not None
        ]
        cleaned_achievements = [
            cleaned for value in entry.achievements if (cleaned := clean_text(value)) is not None
        ]
        rem_desc, final_achievements = _separate_quantified_achievements(
            cleaned_desc, cleaned_achievements
        )
        experiences.append(
            entry.model_copy(
                update={
                    "job_title": clean_text(entry.job_title) or entry.job_title,
                    "company": clean_text(entry.company),
                    "location": clean_text(entry.location),
                    "description": rem_desc if rem_desc else cleaned_desc,
                    "achievements": final_achievements,
                }
            )
        )

    education = [
        entry.model_copy(
            update={
                "institution": clean_text(entry.institution) or entry.institution,
                "degree": clean_text(entry.degree),
                "field_of_study": clean_text(entry.field_of_study),
                "location": clean_text(entry.location),
                "coursework": [
                    cleaned
                    for value in entry.coursework
                    if (cleaned := clean_text(value)) is not None
                ],
            }
        )
        for entry in profile.education
    ]
    if not education:
        education = _recover_local_education(document)

    projects: list[Project] = []
    for project in evidence_validated_projects:
        cleaned_desc = [
            cleaned for value in project.description if (cleaned := clean_text(value)) is not None
        ]
        cleaned_achievements = [
            cleaned for value in project.achievements if (cleaned := clean_text(value)) is not None
        ]
        rem_desc, final_achievements = _separate_quantified_achievements(
            cleaned_desc, cleaned_achievements
        )
        projects.append(
            project.model_copy(
                update={
                    "title": clean_text(project.title) or project.title,
                    "role": clean_text(project.role),
                    "description": rem_desc if rem_desc else cleaned_desc,
                    "achievements": final_achievements,
                    "start_date": project.start_date
                    if _project_date_is_grounded(project.start_date, project.evidence)
                    else None,
                    "end_date": project.end_date
                    if _project_date_is_grounded(project.end_date, project.evidence)
                    else None,
                }
            )
        )

    missing_expected_entities = False
    if _has_section(document, SectionKind.EXPERIENCE) and not experiences:
        missing_expected_entities = True
        issues.append(
            Issue(
                code=WarningCode.MISSING_EXPECTED_SECTION_ENTITIES,
                message=(
                    "Work Experience section was recovered but no verified "
                    "experience was extracted."
                ),
                stage="hybrid_validation",
            )
        )
        recovered_exp = _recover_local_experience(document)
        if recovered_exp:
            experiences = recovered_exp
    if _has_section(document, SectionKind.PROJECTS) and not projects:
        missing_expected_entities = True
        issues.append(
            Issue(
                code=WarningCode.MISSING_EXPECTED_SECTION_ENTITIES,
                message="Projects section was recovered but no verified project was extracted.",
                stage="hybrid_validation",
            )
        )

    skills_by_canonical = {skill.canonical_name.casefold(): skill for skill in profile.skills}
    deterministic_skill_types = document_score_features(document).skill_types
    for canonical, source_types in deterministic_skill_types.items():
        key = canonical.casefold()
        existing = skills_by_canonical.get(key)
        if existing is None:
            skills_by_canonical[key] = Skill(
                name=canonical,
                canonical_name=canonical,
                evidence_types=list(source_types),
                confidence=1.0,
            )
        else:
            skills_by_canonical[key] = existing.model_copy(
                update={"evidence_types": list(source_types)}
            )
    for block_text, evidence_type in _block_evidence_types(document):
        for canonical, matched in canonical_skill_mentions(block_text):
            key = canonical.casefold()
            existing = skills_by_canonical.get(key)
            if existing is None:
                skills_by_canonical[key] = Skill(
                    name=matched,
                    canonical_name=canonical,
                    evidence=[block_text],
                    evidence_types=[evidence_type],
                    confidence=1.0,
                )
            else:
                evidence = [*existing.evidence]
                if block_text not in evidence:
                    evidence.append(block_text)
                skills_by_canonical[key] = existing.model_copy(
                    update={
                        "evidence": evidence,
                        "evidence_types": sorted({*existing.evidence_types, evidence_type}),
                    }
                )

    skills = []
    for skill in skills_by_canonical.values():
        canonical_name = canonical_skill_name(skill.canonical_name)
        evidence_types: set[EvidenceType] = set(
            deterministic_skill_types.get(canonical_name, skill.evidence_types)
        )
        if canonical_name not in deterministic_skill_types:
            evidence_types = document_evidence_types_for_skill(document, canonical_name)
        if not evidence_types:
            evidence_types.add(EvidenceType.LISTED)

        skills.append(
            skill.model_copy(
                update={
                    "canonical_name": canonical_name,
                    "evidence_types": sorted(evidence_types),
                }
            )
        )

    # Identity is intentionally optional: the PII-safe pipeline may leave a
    # candidate name unresolved while semantic evidence remains trustworthy.
    major_evidence = not experiences or all(entry.evidence for entry in experiences)
    if manual_review:
        status = ValidationStatus.MANUAL_REVIEW
    elif (
        provider_name == "local-compatibility"
        or not major_evidence
        or ungrounded
        or missing_expected_entities
    ):
        status = ValidationStatus.PARTIAL
    else:
        status = ValidationStatus.VALIDATED

    return profile.model_copy(
        update={
            "candidate_name": candidate_name,
            "headline": headline,
            "address": clean_text(profile.address),
            "experiences": experiences,
            "education": education,
            "projects": projects,
            "honors_awards": [
                award.model_copy(
                    update={
                        "title": clean_text(award.title) or award.title,
                        "issuer": clean_text(award.issuer),
                    }
                )
                for award in profile.honors_awards
                if award.evidence and all(_grounded(item, document) for item in award.evidence)
            ],
            "language_proficiencies": [
                language.model_copy(
                    update={
                        "language": clean_text(language.language) or language.language,
                        "proficiency": clean_text(language.proficiency),
                        "exam": clean_text(language.exam),
                        "issuer": clean_text(language.issuer),
                    }
                )
                for language in profile.language_proficiencies
                if language.evidence
                and all(_grounded(item, document) for item in language.evidence)
            ],
            "skills": skills,
            "total_experience_years": _experience_years(
                profile.model_copy(update={"experiences": experiences})
            ),
            "validation_status": status,
            "warnings": issues,
        }
    )
