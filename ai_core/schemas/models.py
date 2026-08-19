"""Versioned business schemas for CV processing."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, HttpUrl, field_validator, model_validator

from ai_core.errors import ContractModel, Issue

SCHEMA_VERSION = "1.0.0"


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProcessingStatus(StrEnum):
    VALIDATED = "validated"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationStatus(StrEnum):
    VALIDATED = "validated"
    PARTIAL = "partial"
    MANUAL_REVIEW = "manual_review"


class ScoreStatus(StrEnum):
    SCORED = "scored"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_APPLICABLE = "not_applicable"


class EvidenceType(StrEnum):
    LISTED = "listed"
    PROJECT = "project"
    WORK_EXPERIENCE = "work_experience"
    RESEARCH = "research"
    EDUCATION = "education"


class DiagnosticStatus(StrEnum):
    USABLE = "usable"
    FALLBACK_REQUIRED = "fallback_required"
    UNUSABLE = "unusable"


class BlockType(StrEnum):
    TEXT = "text"
    SECTION_HEADER = "section_header"
    TABLE = "table"
    LIST_ITEM = "list_item"


class SectionKind(StrEnum):
    HEADER = "header"
    SUMMARY = "summary"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    LANGUAGES = "languages"
    OTHER = "other"


class DocumentMetadata(ContractModel):
    source_name: str = Field(min_length=1)
    extension: Literal[".pdf", ".docx"]
    media_type: Literal[
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int | None = Field(default=None, ge=0)
    author: str | None = None
    title: str | None = None
    is_encrypted: bool = False


class PageContent(ContractModel):
    page_number: int = Field(ge=1)
    text: str


class ParsedDocument(ContractModel):
    metadata: DocumentMetadata
    pages: list[PageContent] = Field(default_factory=list)
    raw_text: str = ""
    normalized_text: str = ""
    warnings: list[Issue] = Field(default_factory=list)

    @model_validator(mode="after")
    def pages_are_unique_and_ordered(self) -> ParsedDocument:
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != sorted(set(page_numbers)):
            raise ValueError("pages must have unique ascending pageNumber values")
        return self


class CVSection(ContractModel):
    kind: SectionKind
    heading: str | None = None
    text: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def line_range_is_valid(self) -> CVSection:
        if self.end_line < self.start_line:
            raise ValueError("endLine must be greater than or equal to startLine")
        return self


class Skill(ContractModel):
    name: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    years: float | None = Field(default=None, ge=0)
    level: str | None = None
    evidence: list[str] = Field(default_factory=list)
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    section: SectionKind | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class Experience(ContractModel):
    job_title: str = Field(min_length=1)
    company: str | None = None
    location: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    is_current: bool = False
    description: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def date_range_is_valid(self) -> Experience:
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("endDate must not be before startDate")
        if self.is_current and self.end_date:
            raise ValueError("current experience must not have endDate")
        return self


class GradePointAverage(ContractModel):
    value: float = Field(gt=0)
    scale: float | None = Field(default=None, gt=0)


class Education(ContractModel):
    institution: str = Field(min_length=1)
    degree: str | None = None
    field_of_study: str | None = None
    location: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    gpa: GradePointAverage | None = None
    coursework: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def date_range_is_valid(self) -> Education:
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("endDate must not be before startDate")
        return self


class EvidenceRef(ContractModel):
    text: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    block_id: str = Field(min_length=1)


class Project(ContractModel):
    title: str = Field(min_length=1)
    role: str | None = None
    description: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def date_range_is_valid(self) -> Project:
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("endDate must not be before startDate")
        return self


class HonorAward(ContractModel):
    title: str = Field(min_length=1)
    issuer: str | None = None
    award_date: date | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)


class LanguageProficiency(ContractModel):
    language: str = Field(min_length=1)
    proficiency: str | None = None
    exam: str | None = None
    score: float | None = Field(default=None, ge=0)
    scale: float | None = Field(default=None, gt=0)
    issuer: str | None = None
    exam_date: date | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)


class CriterionScore(ContractModel):
    """V3 AI-calibrated criterion score (0-100 scale)."""
    name: str | None = None
    score: float = Field(default=0.0, ge=0, le=100)
    tier: str | None = None
    explanation: str | None = None
    evidence_summary: list[str] = Field(default_factory=list)


class EvaluatedTiers(ContractModel):
    """AI-assessed evidence-based tier classifications for 6 IT scoring criteria."""

    education_tier: str | None = Field(
        default=None,
        description="TIER_1A_ELITE | TIER_1B_ACCREDITED_TECH | STANDARD_ACCREDITED | ASSOCIATE_OTHER | NON_DEGREE",
    )
    company_prestige_tier: str | None = Field(
        default=None,
        description="TIER_1_BIGTECH_ENTERPRISE | TIER_2_MID_TECH | STANDARD_SME",
    )
    skill_evidence_level: str | None = Field(
        default=None,
        description="ADVANCED_EVIDENCE_BASED | COMPETENT_PRODUCTION | BASIC_KEYWORD_ONLY",
    )
    project_quality_tier: str | None = Field(
        default=None,
        description="HIGH_IMPACT_METRICS | STANDARD_COMPLETED | ACADEMIC_ONLY",
    )
    certification_tier: str | None = Field(
        default=None,
        description="EXPERT_PRO | ASSOCIATE_PRACTITIONER | BASIC_FOUNDATIONAL | NONE",
    )
    language_proficiency: str | None = Field(
        default=None,
        description="EXPERT_FLUENT | WORKING_PROFICIENCY | BASIC_ELEMENTARY | NONE",
    )


class CVProfile(ContractModel):
    candidate_name: str | None = None
    email: str | None = None
    phone: str | None = None
    urls: list[HttpUrl] = Field(default_factory=list)
    address: str | None = None
    date_of_birth: date | None = None
    headline: str | None = None
    summary: str | None = None
    skills: list[Skill] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    language_proficiencies: list[LanguageProficiency] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    honors_awards: list[HonorAward] = Field(default_factory=list)
    sections: list[CVSection] = Field(default_factory=list)
    total_experience_years: float | None = Field(default=None, ge=0)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    field_evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    validation_status: ValidationStatus = ValidationStatus.PARTIAL
    warnings: list[Issue] = Field(default_factory=list)
    # V2 IT Scoring extension fields (strictly additive, backward-compatible defaults)
    primary_role_domain: str | None = None
    evaluated_tiers: EvaluatedTiers | None = None
    executive_summary: str | None = None
    rubric_scores: dict[str, CriterionScore] | None = None

    @field_validator("field_confidence")
    @classmethod
    def confidence_values_are_bounded(cls, value: dict[str, float]) -> dict[str, float]:
        if any(confidence < 0 or confidence > 1 for confidence in value.values()):
            raise ValueError("field confidence values must be between 0 and 1")
        return value



class EmbeddingResult(ContractModel):
    model: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    normalized: bool
    vector: list[float]
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_version: str = Field(min_length=1)
    duration_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def vector_matches_dimension(self) -> EmbeddingResult:
        if len(self.vector) != self.dimension:
            raise ValueError("vector length must equal dimension")
        return self


class ScoreResult(ContractModel):
    name: str = Field(min_length=1)
    status: ScoreStatus = ScoreStatus.SCORED
    score: float | None = Field(default=None, ge=0, le=100)
    breakdown: dict[str, float | None] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    warnings: list[Issue] = Field(default_factory=list)


class SummaryResult(ContractModel):
    text: str = Field(min_length=1)
    sentence_count: int = Field(ge=2, le=3)
    source_fields: list[str] = Field(min_length=1)
    reused_existing: bool = False


class UnifiedBlock(ContractModel):
    id: str = Field(min_length=1)
    type: BlockType = BlockType.TEXT
    text: str
    bbox: tuple[float, float, float, float] | None = None
    reading_order: int = Field(ge=1)


class UnifiedPage(ContractModel):
    page_number: int = Field(ge=1)
    blocks: list[UnifiedBlock] = Field(default_factory=list)


class ExtractionDiagnostics(ContractModel):
    status: DiagnosticStatus
    issues: list[Issue] = Field(default_factory=list)
    text_character_count: int = Field(ge=0)
    block_count: int = Field(ge=0)
    heading_count: int = Field(ge=0)


class UnifiedDocument(ContractModel):
    schema_version: Literal["2.0"] = "2.0"
    file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    extractor: str = Field(min_length=1)
    markdown: str
    pages: list[UnifiedPage] = Field(default_factory=list)
    diagnostics: ExtractionDiagnostics


class MatchResult(ContractModel):
    candidate_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    raw_similarity: float = Field(ge=-1, le=1)
    score: ScoreResult | None = None
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    warnings: list[Issue] = Field(default_factory=list)


class ProcessingResult(ContractModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: ProcessingStatus
    source_id: str = Field(min_length=1)
    metadata: DocumentMetadata | None = None
    parsed_document: ParsedDocument | None = None
    unified_document: UnifiedDocument | None = None
    profile: CVProfile | None = None
    summary: SummaryResult | None = None
    embedding: EmbeddingResult | None = None
    scores: list[ScoreResult] = Field(default_factory=list)
    warnings: list[Issue] = Field(default_factory=list)
    errors: list[Issue] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    audit: dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def status_matches_errors(self) -> ProcessingResult:
        if self.status == ProcessingStatus.FAILED and not self.errors:
            raise ValueError("failed result must contain at least one error")
        if self.status != ProcessingStatus.FAILED and self.errors:
            raise ValueError("only failed result may contain errors")
        return self
