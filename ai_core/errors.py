"""Stable error and warning contracts used across the standalone core."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ContractModel(BaseModel):
    """Base model with a stable camelCase JSON representation."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class ErrorCode(StrEnum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    NOT_A_FILE = "NOT_A_FILE"
    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CORRUPT_PDF = "CORRUPT_PDF"
    ENCRYPTED_PDF = "ENCRYPTED_PDF"
    CORRUPT_DOCX = "CORRUPT_DOCX"
    LLM_CONFIGURATION_ERROR = "LLM_CONFIGURATION_ERROR"
    LLM_AUTHENTICATION_ERROR = "LLM_AUTHENTICATION_ERROR"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_MALFORMED_RESPONSE = "LLM_MALFORMED_RESPONSE"
    LLM_DEGENERATE_RESPONSE = "LLM_DEGENERATE_RESPONSE"
    LLM_SEMANTIC_REGRESSION = "LLM_SEMANTIC_REGRESSION"
    LLM_EMPTY_RESPONSE = "LLM_EMPTY_RESPONSE"
    LLM_REQUEST_FAILED = "LLM_REQUEST_FAILED"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class WarningCode(StrEnum):
    PARSER_PENDING = "PARSER_PENDING"
    TEXT_TOO_SHORT = "TEXT_TOO_SHORT"
    NO_TEXT_LAYER = "NO_TEXT_LAYER"
    REPEATED_MARGIN_REMOVED = "REPEATED_MARGIN_REMOVED"
    INVALID_TIMELINE = "INVALID_TIMELINE"
    LOW_CONFIDENCE_NAME = "LOW_CONFIDENCE_NAME"
    MISSING_CONTACT = "MISSING_CONTACT"
    MISSING_PROFILE_FIELD = "MISSING_PROFILE_FIELD"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    EVIDENCE_NOT_GROUNDED = "EVIDENCE_NOT_GROUNDED"
    UNSUPPORTED_EXPERIENCE_CLAIM = "UNSUPPORTED_EXPERIENCE_CLAIM"
    UNSUPPORTED_PROJECT_CLAIM = "UNSUPPORTED_PROJECT_CLAIM"
    AMBIGUOUS_ENTRY_ATTACHMENT = "AMBIGUOUS_ENTRY_ATTACHMENT"
    MISSING_EXPECTED_SECTION_ENTITIES = "MISSING_EXPECTED_SECTION_ENTITIES"
    CANDIDATE_NAME_IS_LOCATION = "CANDIDATE_NAME_IS_LOCATION"
    SUSPICIOUS_READING_ORDER = "SUSPICIOUS_READING_ORDER"
    PAGE_COUNT_MISMATCH = "PAGE_COUNT_MISMATCH"
    MISSING_LAYOUT_GEOMETRY = "MISSING_LAYOUT_GEOMETRY"
    TABLE_LAYOUT_AMBIGUITY = "TABLE_LAYOUT_AMBIGUITY"
    FALLBACK_REQUIRED = "FALLBACK_REQUIRED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    # Sprint-1: Reconciliation tiered fallback (2026-08-10)
    EVIDENCE_SECTION_INFERRED = "EVIDENCE_SECTION_INFERRED"
    # Sprint-1: Token budget awareness for description pruning (2026-08-10)
    DESCRIPTION_TOKENS_ESTIMATED = "DESCRIPTION_TOKENS_ESTIMATED"


class Issue(ContractModel):
    code: ErrorCode | WarningCode
    message: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class CoreError(Exception):
    """Expected processing error which is safe to show in concise form."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.issue = Issue(
            code=code,
            message=message,
            stage=stage,
            details=details or {},
        )


class ExitCode(IntEnum):
    SUCCESS = 0
    USAGE_ERROR = 2
    INPUT_ERROR = 3
    PROCESSING_ERROR = 4
    NOT_IMPLEMENTED = 5
