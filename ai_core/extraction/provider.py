"""One-call full-document structured extraction provider contract."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from ai_core.errors import ContractModel
from ai_core.extraction.dto import LLMExtractedProfile
from ai_core.extraction.privacy import CandidateIdentity, mask_document
from ai_core.extraction.profile import extract_cv_profile
from ai_core.parsers.unified import hybrid_config
from ai_core.schemas import CVProfile, UnifiedDocument


class ExtractionRequest(ContractModel):
    document: UnifiedDocument
    json_schema: dict[str, object]
    prompt_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)


class ExtractionProvider(Protocol):
    name: str
    model: str
    revision: str

    def extract(self, request: ExtractionRequest) -> CVProfile:
        """Return the complete profile in one provider call."""
        ...


class CompatibilityFullDocumentProvider:
    """Rollback-safe local provider until an external LLM is configured.

    It consumes the complete document once and is deliberately marked as
    compatibility output by reconciliation; it never masquerades as an LLM.
    """

    name = "local-compatibility"
    model = "deterministic-full-document-baseline"
    revision = "1"

    def extract(self, request: ExtractionRequest) -> CVProfile:
        return extract_cv_profile(request.document.markdown)


def configured_provider() -> ExtractionProvider:
    """Build the configured external provider without changing the protocol."""

    provider_name = str(hybrid_config()["provider"])
    if provider_name == "beeknoee-openai-compatible":
        from ai_core.extraction.beeknoee import BeeknoeeStructuredExtractionProvider

        return BeeknoeeStructuredExtractionProvider()
    if provider_name == CompatibilityFullDocumentProvider.name:
        return CompatibilityFullDocumentProvider()
    raise ValueError(f"Unsupported extraction provider: {provider_name}")


def build_request(
    document: UnifiedDocument,
    *,
    local_identity: CandidateIdentity | None = None,
    header_masking_mode: str = "selective",
) -> ExtractionRequest:
    """Build a request only from a locally redacted semantic document.

    This second, idempotent boundary protects direct provider use as well as
    the normal pipeline boundary.
    """

    config = hybrid_config()
    return ExtractionRequest(
        document=mask_document(
            document,
            local_identity=local_identity,
            header_masking_mode=header_masking_mode,
        ).document,
        json_schema=LLMExtractedProfile.model_json_schema(by_alias=True),
        prompt_version=str(config["promptVersion"]),
        schema_version=str(config["profileSchemaVersion"]),
    )
