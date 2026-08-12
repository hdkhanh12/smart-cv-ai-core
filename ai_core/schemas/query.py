"""Pydantic schemas for Search Query & Job Description Analysis."""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from ai_core.errors import ContractModel

EMBEDDING_DIMENSION = 1024



class SearchQueryFilters(ContractModel):
    """Structured Metadata Filters extracted from Query or JD for PostgreSQL pgvector filtering."""

    min_experience_years: float | None = Field(default=None, alias="MinYearsOfExperience", description="Số năm kinh nghiệm tối thiểu")
    max_experience_years: float | None = Field(default=None, alias="MaxYearsOfExperience", description="Số năm kinh nghiệm tối đa")
    skills: list[str] = Field(default_factory=list, alias="Skills", description="Kỹ năng bắt buộc (Must-have skills)")
    job_titles: list[str] = Field(default_factory=list, alias="JobTitles", description="Chức danh công việc tìm kiếm")
    companies: list[str] = Field(default_factory=list, alias="Companies", description="Công ty mục tiêu hoặc công ty đăng tuyển")
    languages: list[str] = Field(default_factory=list, alias="Languages", description="Yêu cầu ngoại ngữ (Tiếng Anh, Tiếng Nhật...)")
    highest_education: str | None = Field(default=None, alias="HighestEducation", description="Trình độ học vấn cao nhất")
    work_type: Literal["Remote", "Onsite", "Hybrid"] | None = Field(default=None, alias="WorkType", description="Hình thức làm việc")
    location: str | None = Field(default=None, alias="Location", description="Địa điểm làm việc")


class SearchQueryRequest(ContractModel):
    """Request payload from C# Backend or Client."""

    text: str = Field(..., alias="Text", min_length=1, description="Câu search query ngắn hoặc toàn bộ nội dung JD")


class SearchQueryEmbeddingInfo(ContractModel):
    """Metadata thông tin Vector Embedding BGE-M3."""

    vector: list[float] = Field(..., min_length=EMBEDDING_DIMENSION, max_length=EMBEDDING_DIMENSION)
    dimension: int = Field(EMBEDDING_DIMENSION)
    normalized: bool = Field(True)
    model: str
    model_revision: str
    template_version: str
    source_hash: str
    duration_ms: float


class SearchQueryAnalysisResult(ContractModel):
    """Full analysis result for internally processed JD / Query."""

    status: Literal["succeeded", "fallback_taxonomy"]
    filters: SearchQueryFilters
    embedding: SearchQueryEmbeddingInfo
    latency_ms: float
