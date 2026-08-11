"""Document-derived, deterministic inputs for profile scoring."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from ai_core.extraction.profile import canonical_skill_mentions
from ai_core.schemas import EvidenceType, UnifiedDocument

FEATURE_VERSION = "document-score-v1"
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\d)\+?\d[\d ().-]{7,}\d(?!\d)")
_IMPACT = re.compile(
    r"\b(processed|supported|managed|handled|improved|reduced|increased|grew|saved|"
    r"built|developed|optimized|tăng|giảm|cải thiện|quản lý|đạt)\b",
    re.I,
)
_QUANTITY = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|[kmb]\+?\b|\+|/|"
    r"(?:users|records|events|requests|queries|pages|ms|seconds|minutes|hours|days))",
    re.I,
)


@dataclass(frozen=True)
class DocumentScoreFeatures:
    skill_types: dict[str, tuple[EvidenceType, ...]]
    achievement_count: int
    description_score: float | None
    has_identity: bool
    has_contact: bool
    has_summary: bool
    has_experience: bool
    has_education: bool
    has_additional: bool
    structure_count: int
    feature_hash: str


def _section_for(text: str, current: str | None) -> str | None:
    key = text.casefold()
    heading = text.startswith("##") or (len(text) <= 60 and text == text.upper())
    if not heading:
        return current
    if any(value in key for value in ("work experience", "professional experience", "employment")):
        return "work"
    if any(value in key for value in ("project", "personal project")):
        return "project"
    if any(value in key for value in ("research", "publication", "paper", "journal")):
        return "research"
    if any(
        value in key for value in ("technical skills", "skills", "competencies", "technologies")
    ):
        return "skills"
    if "education" in key:
        return "education"
    if any(value in key for value in ("summary", "profile", "objective", "introduction")):
        return "summary"
    if any(value in key for value in ("language", "certification")):
        return "additional"
    return current


def document_score_features(document: UnifiedDocument) -> DocumentScoreFeatures:
    """Produce a stable scoring ledger using only original local document blocks."""

    skill_types: dict[str, set[EvidenceType]] = {}
    descriptions: list[str] = []
    achievement_blocks: set[str] = set()
    sections: set[str] = set()
    identity = False
    contact = False
    current: str | None = None
    for page in document.pages:
        for block in page.blocks:
            text = block.text.strip()
            if not text:
                continue
            next_section = _section_for(text, current)
            if next_section != current:
                current = next_section
                if current:
                    sections.add(current)
            phone_values = _PHONE.findall(text)
            if _EMAIL.search(text) or any(
                9 <= len(re.sub(r"\D", "", value)) <= 15 for value in phone_values
            ):
                contact = True
            words = text.removeprefix("##").strip().split()
            if (
                2 <= len(words) <= 5
                and text.removeprefix("##").replace(" ", "").isalpha()
                and "@" not in text
            ):
                identity = True
            evidence_type = {
                "work": EvidenceType.WORK_EXPERIENCE,
                "project": EvidenceType.PROJECT,
                "research": EvidenceType.RESEARCH,
                "skills": EvidenceType.LISTED,
            }.get(current or "", EvidenceType.LISTED)
            for canonical, _matched in canonical_skill_mentions(text):
                skill_types.setdefault(canonical, set()).add(evidence_type)
            if current in {"work", "project", "research"}:
                descriptions.append(text)
                if _IMPACT.search(text) and _QUANTITY.search(text):
                    achievement_blocks.add(text.casefold())
    description_score = None
    if descriptions:
        description_score = min(100.0, 40.0 + sum(map(len, descriptions)) / len(descriptions))
    normalized = {
        "skills": {
            name: sorted(item.value for item in values) for name, values in skill_types.items()
        },
        "achievementCount": len(achievement_blocks),
        "descriptionScore": round(description_score, 2) if description_score is not None else None,
        "sections": sorted(sections),
        "identity": identity,
        "contact": contact,
    }
    feature_hash = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return DocumentScoreFeatures(
        skill_types={name: tuple(sorted(values)) for name, values in skill_types.items()},
        achievement_count=len(achievement_blocks),
        description_score=description_score,
        has_identity=identity,
        has_contact=contact,
        has_summary="summary" in sections,
        has_experience=bool({"work", "project", "research"} & sections),
        has_education="education" in sections,
        has_additional="additional" in sections,
        structure_count=len({"summary", "skills", "work", "education", "project"} & sections),
        feature_hash=feature_hash,
    )


def document_evidence_types_for_skill(
    document: UnifiedDocument,
    skill_name: str,
) -> set[EvidenceType]:
    """Classify an LLM-only skill by direct mention in original document blocks."""

    current: str | None = None
    found: set[EvidenceType] = set()
    needle = skill_name.casefold().strip()
    if not needle:
        return found
    for page in document.pages:
        for block in page.blocks:
            text = block.text.strip()
            current = _section_for(text, current)
            if needle not in text.casefold():
                continue
            found.add(
                {
                    "work": EvidenceType.WORK_EXPERIENCE,
                    "project": EvidenceType.PROJECT,
                    "research": EvidenceType.RESEARCH,
                    "skills": EvidenceType.LISTED,
                }.get(current or "", EvidenceType.LISTED)
            )
    return found
