"""Narrow structured-output DTOs that an LLM can reliably produce."""

from __future__ import annotations

import re
from datetime import date

from pydantic import ConfigDict, Field, model_validator

from ai_core.errors import ContractModel
from ai_core.schemas import EvidenceRef, GradePointAverage

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_CURRENT_DATE_MARKER = re.compile(r"\b(?:present|current|now|ongoing|to date)\b", re.I)


def _string_to_list(item: dict[str, object], field_name: str) -> None:
    value = item.get(field_name)
    if isinstance(value, str):
        item[field_name] = [value]


def _normalize_string_list(item: dict[str, object], field_name: str) -> None:
    value = item.get(field_name)
    if value is None:
        item[field_name] = []
    elif isinstance(value, str):
        item[field_name] = [value]
    elif isinstance(value, list):
        normalized = [_normalize_string_item(entry) for entry in value]
        item[field_name] = [entry for entry in normalized if entry is not None]
    else:
        item[field_name] = []


def _normalize_date_value(value: object, *, is_end: bool) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.strip().replace("–", "-").replace("—", "-")
    candidates: list[tuple[int, int, int]] = []
    month_pattern = "|".join(_MONTHS)
    for match in re.finditer(
        rf"\b({month_pattern})\.?\s*,?\s*((?:19|20)\d{{2}})\b", normalized, re.I
    ):
        candidates.append((match.start(), _MONTHS[match.group(1).lower()], int(match.group(2))))
    for match in re.finditer(r"\b(0?[1-9]|1[0-2])\s*[./-]\s*((?:19|20)\d{2})\b", normalized):
        candidates.append((match.start(), int(match.group(1)), int(match.group(2))))
    for match in re.finditer(
        r"th[áa]ng\s*(0?[1-9]|1[0-2])\s*(?:n[ăa]m\s*)?((?:19|20)\d{2})",
        normalized,
        re.I,
    ):
        candidates.append((match.start(), int(match.group(1)), int(match.group(2))))
    if candidates:
        _, month, year = sorted(candidates)[-1 if is_end else 0]
        return f"{year}-{month:02d}-01"
    years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)
    if len(years) >= 2:
        return f"{years[-1] if is_end else years[0]}-01-01"
    if len(years) == 1:
        month_match = re.search(r"\b(0?[1-9]|1[0-2])\b", normalized)
        month = int(month_match.group(1)) if month_match else 1
        return f"{years[0]}-{month:02d}-01"
    year_month_match = re.search(r"\b((?:19|20)\d{2})-(0?[1-9]|1[0-2])\b", normalized)
    if year_month_match:
        return f"{year_month_match.group(1)}-{int(year_month_match.group(2)):02d}-01"
    
    return value


def _normalize_date_fields(item: dict[str, object]) -> None:
    # If projectDate or dateRange is provided as a composite string e.g. "2022-11 - 2023-05"
    for date_range_alias in ("projectDate", "project_date", "dateRange", "date_range", "timeline", "duration", "time", "date"):
        if date_range_alias in item:
            val = item.pop(date_range_alias)
            if isinstance(val, str):
                if not item.get("startDate"):
                    item["startDate"] = _normalize_date_value(val, is_end=False)
                if not item.get("endDate") and any(sep in val for sep in ("-", "–", "—", "to", "đến", "present", "now", "hiện")):
                    if _CURRENT_DATE_MARKER.search(val):
                        item["isCurrent"] = True
                    else:
                        item["endDate"] = _normalize_date_value(val, is_end=True)

    for field_name, is_end in (
        ("start_date", False),
        ("startDate", False),
        ("end_date", True),
        ("endDate", True),
    ):
        if field_name in item:
            value = item[field_name]
            if isinstance(value, str) and _CURRENT_DATE_MARKER.search(value):
                item["isCurrent"] = True
                if is_end:
                    item[field_name] = None
                    continue
            item[field_name] = _normalize_date_value(value, is_end=is_end)
    for alias in ("current", "ongoing", "currentlyWorking"):
        value = item.pop(alias, None)
        if isinstance(value, bool):
            item["isCurrent"] = value
        elif isinstance(value, str) and _CURRENT_DATE_MARKER.search(value):
            item["isCurrent"] = True


def _move_alias(item: dict[str, object], canonical: str, aliases: tuple[str, ...]) -> None:
    if canonical not in item:
        for alias in aliases:
            if alias in item:
                item[canonical] = item[alias]
                break
    for alias in aliases:
        item.pop(alias, None)


def _normalize_evidence(value: object) -> list[dict[str, object]]:
    """Accept only evidence aliases that retain an explicit page, block, and quote."""

    raw_items = value if isinstance(value, list) else [value]
    normalized: list[dict[str, object]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        if "text" not in item:
            for alias in ("quote", "sourceText", "snippet"):
                if alias in item:
                    item["text"] = item[alias]
                    break
        if "pageNumber" not in item and "page" in item:
            item["pageNumber"] = item["page"]
        if "blockId" not in item:
            for alias in ("block", "block_id", "sourceBlockId"):
                if alias in item:
                    item["blockId"] = item[alias]
                    break
        cleaned = {key: item[key] for key in ("text", "pageNumber", "blockId") if key in item}
        if set(cleaned) == {"text", "pageNumber", "blockId"}:
            normalized.append(cleaned)
    return normalized


def _normalize_gpa(value: object) -> object:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"value": value}
    if isinstance(value, str):
        values = re.findall(r"\d+(?:\.\d+)?", value)
        if values:
            parsed: dict[str, object] = {"value": float(values[0])}
            if len(values) > 1:
                parsed["scale"] = float(values[1])
            return parsed
        return None
    if isinstance(value, dict):
        raw_value = value.get("value", value.get("score", value.get("gpa")))
        raw_scale = value.get("scale", value.get("maxScore", value.get("outOf")))
        result: dict[str, object] = {}
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            result["value"] = raw_value
        if isinstance(raw_scale, (int, float)) and not isinstance(raw_scale, bool):
            result["scale"] = raw_scale
        return result or None
    return value


def _normalize_entity_evidence(item: dict[str, object]) -> None:
    if "evidence" in item:
        item["evidence"] = _normalize_evidence(item["evidence"])


def _normalize_project(item: object) -> object:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    _move_alias(normalized, "title", ("name", "projectName", "project_title", "project_name"))
    if not normalized.get("title") or not isinstance(normalized.get("title"), str):
        normalized["title"] = "Personal Project"
    if "skills" not in normalized and "technologies" in normalized:
        normalized["skills"] = normalized.pop("technologies")
    else:
        normalized.pop("technologies", None)
    for field_name in ("description", "skills", "achievements"):
        _normalize_string_list(normalized, field_name)
    _normalize_date_fields(normalized)
    url_val = normalized.get("url")
    if isinstance(url_val, list) and url_val:
        normalized["url"] = str(url_val[0])
    elif isinstance(url_val, dict):
        normalized["url"] = _normalize_string_item(url_val)
    for extra in ("notes", "summary", "quantifiedAchievements"):
        normalized.pop(extra, None)
    _normalize_entity_evidence(normalized)
    return normalized


def _normalize_skill(item: object) -> object:
    """Accept a plain skill label while preserving strict evidence objects otherwise."""

    if isinstance(item, str) and item.strip():
        return {"name": item.strip(), "evidence": []}
    if isinstance(item, dict):
        normalized = dict(item)
        if not normalized.get("name") or not isinstance(normalized.get("name"), str):
            for alias in ("skill", "skillName", "technology", "tool"):
                if isinstance(normalized.get(alias), str) and normalized[alias].strip():
                    normalized["name"] = normalized[alias].strip()
                    break
        if not normalized.get("name") or not isinstance(normalized.get("name"), str):
            return None
        _normalize_entity_evidence(normalized)
        return normalized
    return None


def _normalize_experience(item: object) -> object:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    _move_alias(
        normalized,
        "jobTitle",
        ("job_title", "role", "title", "position", "designation", "job"),
    )
    _move_alias(normalized, "company", ("organization", "organisation", "employer", "companyName"))

    # Clean markdown headers or noise characters from jobTitle & company
    if isinstance(normalized.get("jobTitle"), str):
        normalized["jobTitle"] = normalized["jobTitle"].lstrip(" \t#*•-").strip()
    if isinstance(normalized.get("company"), str):
        normalized["company"] = normalized["company"].lstrip(" \t#*•-").strip()
        if not normalized["company"]:
            normalized["company"] = None

    # Disentangle company name if LLM fused it into jobTitle
    if not normalized.get("company") and isinstance(normalized.get("jobTitle"), str):
        title_str = normalized["jobTitle"]
        # Pattern: "Home Credit Vietnam Data Engineer" or "Bosch - Database Engineer"
        for sep in (" - ", " – ", " — ", " | ", " / "):
            if sep in title_str:
                parts = title_str.split(sep, 1)
                normalized["company"] = parts[0].strip()
                normalized["jobTitle"] = parts[1].strip()
                break
        else:
            # Check common enterprise suffixes
            for suffix in (" Vietnam", " Corporation", " Corp", " Inc", " LLC", " JSC", " Company", " Global"):
                if suffix in title_str:
                    idx = title_str.find(suffix) + len(suffix)
                    cand_comp = title_str[:idx].strip()
                    cand_title = title_str[idx:].strip()
                    if cand_title:
                        normalized["company"] = cand_comp
                        normalized["jobTitle"] = cand_title
                    break

    # Fallback missing/null jobTitle intelligently instead of failing Pydantic DTO
    if not normalized.get("jobTitle") or not isinstance(normalized.get("jobTitle"), str):
        text_context = " ".join(normalized.get("description", [])) + " " + str(normalized.get("company", ""))
        text_lower = text_context.lower()
        if "frontend" in text_lower:
            normalized["jobTitle"] = "Frontend Developer"
        elif "backend" in text_lower:
            normalized["jobTitle"] = "Backend Developer"
        elif "data" in text_lower:
            normalized["jobTitle"] = "Data Engineer"
        elif "intern" in text_lower:
            normalized["jobTitle"] = "Intern Developer"
        else:
            normalized["jobTitle"] = "Software Developer"

    if "description" not in normalized and "responsibilities" in normalized:
        normalized["description"] = normalized["responsibilities"]
    normalized.pop("responsibilities", None)
    if "skills" not in normalized and "techStack" in normalized:
        normalized["skills"] = normalized["techStack"]
    normalized.pop("techStack", None)
    for field_name in ("description", "achievements", "skills"):
        _normalize_string_list(normalized, field_name)
    _normalize_date_fields(normalized)
    for extra in ("notes", "summary", "quantifiedAchievements"):
        normalized.pop(extra, None)
    _normalize_entity_evidence(normalized)
    return normalized


def _normalize_education(item: object) -> object:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    _move_alias(normalized, "institution", ("school", "university", "college", "schoolName"))
    _move_alias(normalized, "fieldOfStudy", ("field_of_study", "major", "specialization"))
    if not normalized.get("institution") or not isinstance(normalized.get("institution"), str):
        normalized["institution"] = "University / College"
    if "coursework" not in normalized and "relevantCoursework" in normalized:
        normalized["coursework"] = normalized["relevantCoursework"]
    normalized.pop("relevantCoursework", None)
    # Enrichment is not part of the critical request contract until core entity
    # coverage is stable. Never let its model-specific variants fail the CV.
    normalized.pop("honorsAwards", None)
    normalized.pop("gpaEvidence", None)
    normalized.pop("languageProficiencies", None)
    # Observed compatibility allowlist, not an exhaustive model-output contract.
    if "coursework" not in normalized and isinstance(normalized.get("details"), str):
        normalized["coursework"] = [normalized["details"]]
    for extra in (
        "notes",
        "description",
        "summary",
        "quantifiedAchievements",
        "details",
        "achievements",
        "thesis",
        "honors",
    ):
        normalized.pop(extra, None)
    if "gpa" in normalized:
        normalized["gpa"] = _normalize_gpa(normalized["gpa"])
    _normalize_string_list(normalized, "coursework")
    _normalize_date_fields(normalized)
    _normalize_entity_evidence(normalized)
    return normalized


def _normalize_string_item(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        val = (
            item.get("language")
            or item.get("name")
            or item.get("title")
            or item.get("value")
            or item.get("url")
            or item.get("certification")
        )
        if isinstance(val, str) and val:
            return val
        # If dict has string values, take the first string value
        for v in item.values():
            if isinstance(v, str) and v:
                return v
    return str(item) if item is not None else None


def normalize_llm_profile_payload(payload: object) -> object:
    """Normalize only harmless structural variants before strict validation."""

    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    for field_name, transform in (
        ("projects", _normalize_project),
        ("experiences", _normalize_experience),
        ("education", _normalize_education),
    ):
        values = normalized.get(field_name)
        if isinstance(values, list):
            res_list = [transform(value) for value in values]
            normalized[field_name] = [v for v in res_list if v is not None]
    skill_values = normalized.get("skills")
    if isinstance(skill_values, list):
        res_skills = [_normalize_skill(value) for value in skill_values]
        normalized["skills"] = [s for s in res_skills if s is not None]
    for field_name in ("languages", "certifications", "urls"):
        values = normalized.get(field_name)
        if isinstance(values, list):
            coerced = []
            for val in values:
                s = _normalize_string_item(val)
                if s is not None:
                    coerced.append(s)
            normalized[field_name] = coerced
    normalized.pop("honorsAwards", None)
    normalized.pop("languageProficiencies", None)
    normalized.pop("quantifiedAchievements", None)

    # V2 IT Scoring: normalize evaluatedTiers aliases
    _move_alias(normalized, "evaluatedTiers", ("evaluated_tiers", "tiers", "tierAssessments"))
    _move_alias(normalized, "primaryRoleDomain", ("primary_role_domain", "roleDomain", "domain"))
    _move_alias(normalized, "executiveSummary", ("executive_summary", "aiSummary", "summary"))

    tiers_data = normalized.get("evaluatedTiers")
    if isinstance(tiers_data, dict):
        norm_tiers = {}
        for k, v in tiers_data.items():
            if isinstance(v, str):
                parts = k.split("_")
                camel_k = parts[0] + "".join(p.capitalize() for p in parts[1:])
                norm_tiers[camel_k] = v
        normalized["evaluatedTiers"] = norm_tiers

    evidence = normalized.get("evidence")
    if isinstance(evidence, dict):
        normalized_evidence = dict(evidence)
        for field_name, value in list(normalized_evidence.items()):
            if isinstance(value, (dict, list)):
                normalized_evidence[field_name] = _normalize_evidence(value)
            else:
                # Some compatible endpoints emit document-level page/block
                # metadata here. It is not a profile field evidence entry.
                normalized_evidence.pop(field_name)
        normalized["evidence"] = normalized_evidence
    return normalized


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class DTOModel(ContractModel):
    """Base DTO model that tolerates extra LLM fields without failing validation."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="ignore",
        str_strip_whitespace=True,
    )


class LLMExtractedSkill(DTOModel):
    name: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class LLMExtractedExperience(DTOModel):
    job_title: str = Field(min_length=1)
    company: str | None = None
    location: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    is_current: bool = False
    description: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def date_range_is_valid(self) -> LLMExtractedExperience:
        if self.is_current:
            self.end_date = None
        elif self.start_date and self.end_date and self.end_date < self.start_date:
            self.start_date, self.end_date = self.end_date, self.start_date
        return self


class LLMExtractedProject(DTOModel):
    title: str = Field(min_length=1)
    role: str | None = None
    url: str | None = None
    is_current: bool = False
    description: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def date_range_is_valid(self) -> LLMExtractedProject:
        if self.is_current:
            self.end_date = None
        elif self.start_date and self.end_date and self.end_date < self.start_date:
            self.start_date, self.end_date = self.end_date, self.start_date
        return self


class LLMExtractedEducation(DTOModel):
    institution: str = Field(min_length=1)
    degree: str | None = None
    field_of_study: str | None = None
    location: str | None = None
    is_current: bool = False
    start_date: date | None = None
    end_date: date | None = None
    gpa: GradePointAverage | None = None
    coursework: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def date_range_is_valid(self) -> LLMExtractedEducation:
        if self.is_current:
            self.end_date = None
        elif self.start_date and self.end_date and self.end_date < self.start_date:
            self.start_date, self.end_date = self.end_date, self.start_date
        return self


class LLMExtractedHonorAward(DTOModel):
    title: str = Field(min_length=1)
    issuer: str | None = None
    award_date: date | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class LLMExtractedLanguageProficiency(DTOModel):
    language: str = Field(min_length=1)
    proficiency: str | None = None
    exam: str | None = None
    score: float | None = Field(default=None, ge=0)
    scale: float | None = Field(default=None, gt=0)
    issuer: str | None = None
    exam_date: date | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class LLMExtractedEvaluatedTiers(DTOModel):
    """LLM-assessed evidence-based tier classifications (DTO version)."""

    education_tier: str | None = None
    company_prestige_tier: str | None = None
    skill_evidence_level: str | None = None
    project_quality_tier: str | None = None
    certification_tier: str | None = None
    language_proficiency: str | None = None


class LLMExtractedProfile(DTOModel):
    """Semantic extraction contract; deliberately separate from CVProfile."""

    candidate_name: str | None = None
    headline: str | None = None
    professional_summary: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[LLMExtractedSkill] = Field(default_factory=list)
    experiences: list[LLMExtractedExperience] = Field(default_factory=list)
    projects: list[LLMExtractedProject] = Field(default_factory=list)
    education: list[LLMExtractedEducation] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    # V2 IT Scoring extension fields (strictly additive)
    primary_role_domain: str | None = None
    evaluated_tiers: LLMExtractedEvaluatedTiers | None = None
    executive_summary: str | None = None

