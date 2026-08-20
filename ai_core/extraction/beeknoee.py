"""Beeknoee OpenAI-compatible full-document structured extraction provider."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from pydantic import ValidationError

from ai_core.errors import CoreError, ErrorCode
from ai_core.extraction.dto import LLMExtractedProfile, normalize_llm_profile_payload
from ai_core.extraction.layout_prompt import serialize_layout_transcript
from ai_core.extraction.mapping import map_llm_profile
from ai_core.extraction.provider import ExtractionRequest
from ai_core.extraction.pruning import estimate_tokens
from ai_core.parsers.unified import hybrid_config
from ai_core.schemas import CVProfile

DEFAULT_BASE_URL = "https://platform.beeknoee.com/v1"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
MISSING_API_KEY_MESSAGE = (
    "Missing BEE_API_KEY. Copy .env.example to .env and configure the API key."
)
_ENTITY_COLLECTIONS = ("experiences", "projects", "education")
_INPUT_REPRESENTATIONS = {"layout-transcript", "layout-native", "canonical-text"}
_EVIDENCE_MODES = {"baseline", "entity-bound"}
_EVIDENCE_CONTRACTS = {"optional", "required-initial"}
_OUTPUT_SCOPES = {"full", "entities-only", "entity-inventory"}
_REPEATED_NULL = re.compile(r'(?:"null"\s*,?\s*){8,}')


def _build_http_session() -> requests.Session:
    """Build a reusable HTTP session with connection pooling and keep-alive."""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            raise_on_status=False,
        ),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Connection": "keep-alive"})
    return session



def _entity_signature(field_name: str, entry: object) -> str | None:
    """Create a non-reversible, local-only identity token for an LLM entity."""

    if not isinstance(entry, dict):
        return None
    aliases = {
        "experiences": (
            ("jobTitle", "job_title", "role", "title", "position", "designation", "job"),
            ("company", "companyName", "employer", "organization", "organisation"),
        ),
        "projects": (("title", "name", "projectName", "project_title", "project_name"), ()),
        "education": (("institution", "school", "university", "college", "schoolName"), ()),
    }
    primary_aliases, secondary_aliases = aliases[field_name]

    def value_for(names: tuple[str, ...]) -> str:
        for name in names:
            value = entry.get(name)
            if isinstance(value, str) and value.strip():
                return " ".join(value.casefold().split())
        return ""

    primary = value_for(primary_aliases)
    if not primary:
        return None
    secondary = value_for(secondary_aliases)
    start = value_for(("startDate", "start_date", "startYear", "start_year"))
    end = value_for(("endDate", "end_date", "endYear", "end_year"))
    signature_input = "|".join((field_name, primary, secondary, start, end))
    return hashlib.sha256(signature_input.encode("utf-8")).hexdigest()[:16]


def _payload_entity_counts(payload: object) -> dict[str, dict[str, object]]:
    """Return shape-only entity counts; never retain response text or PII."""

    source = payload if isinstance(payload, dict) else {}
    result: dict[str, dict[str, object]] = {}
    for field_name in _ENTITY_COLLECTIONS:
        values = source.get(field_name)
        entries = values if isinstance(values, list) else []
        evidence_values = [entry.get("evidence") for entry in entries if isinstance(entry, dict)]
        evidence_references: dict[str, list[str]] = {}
        for entry in entries:
            signature = _entity_signature(field_name, entry)
            if signature is None or not isinstance(entry, dict):
                continue
            references = {
                f"{evidence.get('pageNumber')}|{evidence.get('blockId')}"
                for evidence in entry.get("evidence", [])
                if isinstance(evidence, dict)
                and evidence.get("pageNumber") is not None
                and isinstance(evidence.get("blockId"), str)
            }
            evidence_references[signature] = sorted(references)
        result[field_name] = {
            "entities": len(entries),
            "entitiesWithEvidenceField": sum(bool(value) for value in evidence_values),
            "evidenceReferences": sum(
                len(value) if isinstance(value, list) else int(isinstance(value, dict))
                for value in evidence_values
            ),
            "entitySignatures": [
                signature
                for entry in entries
                if (signature := _entity_signature(field_name, entry)) is not None
            ],
            "entityEvidenceReferences": evidence_references,
        }
    return result


def _initial_entity_snapshots(payload: object) -> dict[str, list[dict[str, object]]]:
    """Benchmark-only identity snapshot before aliases/DTO/reconciliation.

    Deliberately excludes descriptions, evidence text, contacts and URLs. It is
    opt-in under the synthetic benchmark gate and exists only to diagnose where
    an entity loses its primary identity.
    """
    source = payload if isinstance(payload, dict) else {}
    result: dict[str, list[dict[str, object]]] = {}
    fields = {
        "experiences": {
            "jobTitle": ("jobTitle", "job_title", "role", "title", "position", "designation"),
            "company": ("company", "companyName", "employer", "organization", "organisation"),
            "startDate": ("startDate", "start_date", "startYear", "start_year"),
            "endDate": ("endDate", "end_date", "endYear", "end_year"),
        },
        "projects": {
            "title": ("title", "name", "projectName", "project_title", "project_name"),
            "role": ("role", "jobTitle", "position"),
            "startDate": ("startDate", "start_date", "startYear", "start_year"),
            "endDate": ("endDate", "end_date", "endYear", "end_year"),
        },
    }
    for collection, canonical_fields in fields.items():
        values = source.get(collection)
        entries: list[dict[str, object]] = []
        for entry in values if isinstance(values, list) else []:
            if isinstance(entry, dict):
                snapshot: dict[str, object] = {}
                aliases_used: dict[str, str] = {}
                for canonical, aliases in canonical_fields.items():
                    for alias in aliases:
                        value = entry.get(alias)
                        if value is not None:
                            snapshot[canonical] = value
                            aliases_used[canonical] = alias
                            break
                if aliases_used:
                    snapshot["sourceAliases"] = aliases_used
                entries.append(snapshot)
        result[collection] = entries
    return result


def _dto_entity_counts(profile: LLMExtractedProfile) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for field_name in _ENTITY_COLLECTIONS:
        entries = getattr(profile, field_name)
        result[field_name] = {
            "entities": len(entries),
            "entitiesWithEvidenceField": sum(bool(entry.evidence) for entry in entries),
            "evidenceReferences": sum(len(entry.evidence) for entry in entries),
        }
    return result


def _semantic_regression(initial: object, correction: object) -> dict[str, int]:
    """Compare only opaque entity signatures; no model text enters diagnostics."""

    regression: dict[str, int] = {}
    initial_source = initial if isinstance(initial, dict) else {}
    correction_source = correction if isinstance(correction, dict) else {}
    for field_name in _ENTITY_COLLECTIONS:
        initial_field = initial_source.get(field_name)
        correction_field = correction_source.get(field_name)
        initial_signatures = (
            set(initial_field.get("entitySignatures", []))
            if isinstance(initial_field, dict)
            else set()
        )
        correction_signatures = (
            set(correction_field.get("entitySignatures", []))
            if isinstance(correction_field, dict)
            else set()
        )
        regression[field_name] = len(initial_signatures - correction_signatures)
    return regression


def _evidence_reference_regression(initial: object, correction: object) -> dict[str, int]:
    """Count lost page/block evidence references for entities that survive correction."""

    result: dict[str, int] = {}
    initial_source = initial if isinstance(initial, dict) else {}
    correction_source = correction if isinstance(correction, dict) else {}
    for field_name in _ENTITY_COLLECTIONS:
        initial_field = initial_source.get(field_name)
        correction_field = correction_source.get(field_name)
        initial_refs = (
            initial_field.get("entityEvidenceReferences", {})
            if isinstance(initial_field, dict)
            else {}
        )
        correction_refs = (
            correction_field.get("entityEvidenceReferences", {})
            if isinstance(correction_field, dict)
            else {}
        )
        lost = 0
        if isinstance(initial_refs, dict) and isinstance(correction_refs, dict):
            for signature, references in initial_refs.items():
                if signature not in correction_refs or not isinstance(references, list):
                    continue
                corrected = correction_refs.get(signature)
                if isinstance(corrected, list):
                    lost += len(set(references) - set(corrected))
        result[field_name] = lost
    return result


@dataclass(frozen=True)
class BeeknoeeSettings:
    """Runtime settings. The key is deliberately omitted from repr output."""

    base_url: str
    api_key: str = field(repr=False)
    model: str


class ResponseFormatUnsupported(Exception):
    """The compatible endpoint rejected the requested structured-output mode."""


class OutputValidationError(Exception):
    """A model response needs one schema-correction retry."""

    def __init__(self, field_paths: list[str]) -> None:
        self.field_paths = field_paths
        super().__init__("; ".join(field_paths))


def load_beeknoee_settings(
    *,
    dotenv_path: Path | None = None,
    override: bool = False,
) -> BeeknoeeSettings:
    """Load local configuration without exposing any secret in diagnostics."""

    root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=dotenv_path or root / ".env", override=override)
    api_key = os.getenv("BEE_API_KEY", "").strip()
    if not api_key:
        raise CoreError(
            ErrorCode.LLM_CONFIGURATION_ERROR,
            MISSING_API_KEY_MESSAGE,
            stage="llm_provider",
        )
    configured_base_url = os.getenv("BEE_BASE_URL", DEFAULT_BASE_URL).strip()
    return BeeknoeeSettings(
        base_url=(configured_base_url or DEFAULT_BASE_URL).rstrip("/"),
        api_key=api_key,
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
    )


class BeeknoeeStructuredExtractionProvider:
    """One OpenAI-compatible chat-completions call for each UnifiedDocument."""

    name = "beeknoee-openai-compatible"

    def __init__(
        self,
        *,
        settings: BeeknoeeSettings | None = None,
        session: requests.Session | None = None,
        raw_capture_dir: Path | None = None,
        raw_capture_ttl_days: int = 7,
        input_representation: str = "layout-transcript",
        evidence_mode: str = "baseline",
        evidence_contract: str = "optional",
        output_scope: str = "full",
        capture_initial_entities: bool = False,
    ) -> None:
        self._settings = settings or load_beeknoee_settings()
        if self._settings.base_url.endswith("/"):
            self._settings = BeeknoeeSettings(
                base_url=self._settings.base_url.rstrip("/"),
                api_key=self._settings.api_key,
                model=self._settings.model,
            )
        self._session = session or _build_http_session()
        config = hybrid_config()
        self.model = self._settings.model
        self.revision = str(config["modelRevision"])
        self._timeout_seconds = float(config["requestTimeoutSeconds"])
        self._max_retries = int(config["maxRetries"])
        self.last_audit: dict[str, object] = {}
        self._last_funnel: dict[str, object] = {}
        self._raw_capture_dir = raw_capture_dir
        self._raw_capture_ttl = timedelta(days=raw_capture_ttl_days)
        if input_representation not in _INPUT_REPRESENTATIONS:
            raise ValueError(f"Unsupported input representation: {input_representation}")
        self._input_representation = input_representation
        if evidence_mode not in _EVIDENCE_MODES:
            raise ValueError(f"Unsupported evidence mode: {evidence_mode}")
        self._evidence_mode = evidence_mode
        if evidence_contract not in _EVIDENCE_CONTRACTS:
            raise ValueError(f"Unsupported evidence contract: {evidence_contract}")
        if evidence_contract == "required-initial" and output_scope != "entities-only":
            raise ValueError(
                "required-initial evidence contract is limited to entities-only benchmarks"
            )
        if output_scope == "entity-inventory" and (
            evidence_mode != "baseline" or evidence_contract != "optional"
        ):
            raise ValueError(
                "entity-inventory is a no-evidence benchmark contract; use baseline/optional"
            )
        self._evidence_contract = evidence_contract
        if output_scope not in _OUTPUT_SCOPES:
            raise ValueError(f"Unsupported output scope: {output_scope}")
        self._output_scope = output_scope
        self._capture_initial_entities = capture_initial_entities
        if self._raw_capture_dir is not None:
            self._raw_capture_dir.mkdir(parents=True, exist_ok=True)
            self._purge_expired_captures()

    def extract(self, request: ExtractionRequest) -> CVProfile:
        """Request the complete profile once; never issue per-field LLM calls."""

        config = hybrid_config()
        pref = config.get("responseFormatPreference", ["json_object", "json_schema", "text_json"])
        format_map: dict[str, dict[str, object] | None] = {
            "json_object": {"type": "json_object"},
            "json_schema": {
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_extracted_profile",
                    "strict": True,
                    "schema": request.json_schema,
                },
            },
            "text_json": None,
        }
        response_formats: list[dict[str, object] | None] = [
            format_map[p] for p in pref if p in format_map
        ] or [{"type": "json_object"}, None]

        for response_format in response_formats:
            try:
                return self._extract_with_format(request, response_format)
            except ResponseFormatUnsupported:
                continue
        raise CoreError(
            ErrorCode.LLM_REQUEST_FAILED,
            "Beeknoee did not accept any structured-output response format.",
            stage="llm_provider",
        )

    def _extract_with_format(
        self,
        request: ExtractionRequest,
        response_format: dict[str, object] | None,
    ) -> CVProfile:
        payload = self._payload(request, response_format)
        self._last_funnel = {}
        response = self._post_with_retry(payload, capture_stage="initial")
        degeneration = self._degeneration_diagnostics(response)
        if degeneration is not None:
            # Degeneration retry with slight temperature & frequency penalty nudge to break greedy repetition
            retry_payload = dict(payload)
            retry_payload["temperature"] = 0.2
            retry_payload["frequency_penalty"] = 0.2
            response_retry = self._post_with_retry(retry_payload, capture_stage="degeneration_retry")
            degeneration_retry = self._degeneration_diagnostics(response_retry)
            if degeneration_retry is None:
                response = response_retry
            else:
                self._last_funnel = {"response": degeneration}
                self.last_audit = {"funnel": {"initial": self._last_funnel}}
                raise CoreError(
                    ErrorCode.LLM_DEGENERATE_RESPONSE,
                    "Beeknoee completion entered a detected repetition loop.",
                    stage="llm_provider",
                    details={"degeneration": degeneration, "funnel": self.last_audit["funnel"]},
                )
        self.last_audit = {
            "input": estimate_tokens(json.dumps(payload, ensure_ascii=False)),
            "output": estimate_tokens(str(self._completion_content(response))),
            "inputRepresentation": self._input_representation,
            "evidenceMode": self._evidence_mode,
            "evidenceContract": self._evidence_contract,
            "outputScope": self._output_scope,
        }
        try:
            extracted = self._parse_extracted_profile(response)
            self.last_audit["funnel"] = {"initial": self._last_funnel}
            return map_llm_profile(extracted)
        except OutputValidationError as exc:
            import os
            pass_mode = os.getenv("EXTRACTION_PASS_MODE", "single").lower().strip()
            if pass_mode == "single":
                # Single-pass mode: try payload self-healing before making expensive remote calls
                try:
                    healed = self._heal_payload_and_parse(response)
                    self.last_audit["funnel"] = {"initial": self._last_funnel}
                    return map_llm_profile(healed)
                except Exception as heal_err:
                    print(f"[BEEKNOEE SINGLE-PASS HEAL FAILED] {heal_err}. Falling back to correction pass.")

            self.last_audit["funnel"] = {"initial": self._last_funnel}
            correction = self._correction_payload(payload, response, exc.field_paths, request)
            corrected_response = self._post_with_retry(correction, capture_stage="correction")
            try:
                extracted = self._parse_extracted_profile(corrected_response)
                funnel = self.last_audit.get("funnel")
                if isinstance(funnel, dict):
                    funnel["correction"] = self._last_funnel
                    initial = funnel.get("initial")
                    initial_recognized = (
                        initial.get("llmRecognized") if isinstance(initial, dict) else None
                    )
                    correction_recognized = self._last_funnel.get("llmRecognized")
                    entity_regression = _semantic_regression(
                        initial_recognized, correction_recognized
                    )
                    evidence_regression = _evidence_reference_regression(
                        initial_recognized, correction_recognized
                    )
                    regression = {
                        "lostEntities": entity_regression,
                        "lostEvidenceReferences": evidence_regression,
                    }
                    funnel["semanticRegression"] = regression
                    if any(entity_regression.values()) or any(evidence_regression.values()):
                        raise CoreError(
                            ErrorCode.LLM_SEMANTIC_REGRESSION,
                            "Schema correction removed previously recognized semantic entities.",
                            stage="llm_provider",
                            details={
                                "lostEntityCounts": entity_regression,
                                "semanticRegression": regression,
                                "funnel": funnel,
                            },
                        )
                return map_llm_profile(extracted)
            except OutputValidationError as correction_error:
                print(f"[BEEKNOEE DEBUG VALIDATION ERROR] {correction_error.field_paths} | {correction_error}")
                funnel = self.last_audit.get("funnel")
                if isinstance(funnel, dict):
                    funnel["correction"] = self._last_funnel
                raise CoreError(
                    ErrorCode.LLM_MALFORMED_RESPONSE,
                    "Beeknoee completion did not match the LLM extraction schema after correction.",
                    stage="llm_provider",
                    details={
                        "fieldPaths": correction_error.field_paths,
                        "funnel": self.last_audit.get("funnel", {}),
                    },
                ) from correction_error

    def _payload(
        self,
        request: ExtractionRequest,
        response_format: dict[str, object] | None,
    ) -> dict[str, Any]:
        evidence_instruction = (
            "5. EVIDENCE: For every experience, project and rubric criterion you return, include concise verbatim text quote strings from the CV in evidence_summary / evidenceSummary proving the claim. Do not invent text.\n"
        )
        entity_instruction = (
            "1. PAST EXPERIENCES: Extract ALL employment history entries across all years "
            "in the past (including early career entries from 2018 or earlier) with "
            "complete startDate and endDate for accurate total experience calculation.\n"
            "2. EMPLOYMENT BOUNDARY: Create an experience only when its evidence explicitly "
            "shows employment. Personal, academic, portfolio, and project-experience sections "
            "are projects, not employment. Never invent an employer, location, date, or role.\n"
            "3. CORE COVERAGE: Before final JSON, scan every recovered Work Experience and "
            "Projects section (including section headers: Projects, Personal Projects, Pet Projects, "
            "Key Projects, Academic Projects, Dự án, Dự án cá nhân, Dự án thực tế, Dự án nổi bật, "
            "Dự án học tập, Đề tài). When an explicit entity exists in either section, return it "
            "with evidence; do not return an empty list merely because details are incomplete.\n"
        )
        privacy_and_layout_instruction = (
            "4. PRIVACY BOUNDARY: candidateName, email, phone, address and personal URLs "
            "are resolved locally and are not extraction targets. Always return null/empty "
            "values for them; never infer, repeat, or decode placeholders.\n"
            "5. COMPLETENESS: Process every explicit employment/project entity; do not omit an entity "
            "solely because one optional field is absent.\n"
            "Return null or empty lists when uncertain; never infer ungrounded info."
        )
        full_scope_instruction = (
            "1. SKILLS: Extract technical, domain, algorithmic, and tool skills as concise "
            "terms (1-4 words max). Do NOT output long sentence descriptions as skills.\n"
            "2. WORK EXPERIENCES & PROJECTS: For EVERY job experience and project in the CV, extract:\n"
            "   - 'description': 2-5 concise bullet points describing key responsibilities, system architectures, data pipelines, ML models, or core technologies used. NEVER leave description empty if the CV describes duties.\n"
            "   - 'achievements': Every bullet point containing numeric metrics, scale (e.g. 100GB/day, millions of rows), percentages (e.g. 40% faster), cost savings, throughput, or business impacts.\n"
            "   - 'skills': List of relevant tools/technologies used in that specific role.\n"
            "3. PROJECT DATES: Extract project dates only from explicit date evidence; otherwise null.\n"
            "4. HEADLINE: Extract only a clear single-line title or objective.\n"
            "5. QUALITATIVE TIERS & BILINGUAL SUMMARY: Classify candidate with exact enum values:\n"
            "   - primaryRoleDomain: 'BACKEND_CLOUD' | 'FRONTEND_WEB' | 'MOBILE' | 'DATA_AI' | 'QA_TESTING' | 'DEVOPS_SRE' | 'FULLSTACK'\n"
            "   - evaluatedTiers.educationTier: 'TIER_1A_ELITE' | 'TIER_1B_ACCREDITED_TECH' | 'STANDARD_ACCREDITED' | 'ASSOCIATE_OTHER' | 'NON_DEGREE'\n"
            "   - evaluatedTiers.companyPrestigeTier: 'TIER_1_BIGTECH_ENTERPRISE' | 'TIER_2_MID_TECH' | 'STANDARD_SME'\n"
            "   - evaluatedTiers.skillEvidenceLevel: 'ADVANCED_EVIDENCE_BASED' | 'COMPETENT_PRODUCTION' | 'BASIC_KEYWORD_ONLY'\n"
            "   - evaluatedTiers.projectQualityTier: 'HIGH_IMPACT_METRICS' | 'STANDARD_COMPLETED' | 'ACADEMIC_ONLY'\n"
            "   - evaluatedTiers.certificationTier: 'EXPERT_PRO' | 'ASSOCIATE_PRACTITIONER' | 'BASIC_FOUNDATIONAL' | 'NONE'\n"
            "   - evaluatedTiers.languageProficiency: 'EXPERT_FLUENT' | 'WORKING_PROFICIENCY' | 'BASIC_ELEMENTARY' | 'NONE'\n"
            "   - executiveSummary: Tóm tắt tổng quan về năng lực, chuyên môn nổi bật và số năm kinh nghiệm của ứng viên bằng TIẾNG VIỆT (2-3 câu súc tích), 100% PII-free.\n"
            "7. RUBRIC SCORING V3 (rubricScores): Score each criterion INDEPENDENTLY on 0-100 scale.\n"
            "   For EACH criterion below, return: {name, score (float 0-100), tier (string), explanation (1-2 Vietnamese sentences citing evidence), evidenceSummary (list of key evidence strings)}.\n"
            "   ANTI-INFLATION RULES: (a) Score MUST stay within the Ceiling Gate range. (b) Each score MUST have at least 1 evidence item. (c) Vague words like 'significant', 'impressive' do NOT count as evidence. (d) When information is missing, use the FLOOR of the matching tier.\n"
            "   rubricScores.technicalDepth (Kỹ năng & Kiến trúc):\n"
            "     [93-100] PRINCIPAL_ARCHITECT: REQUIRES evidence of DESIGNING complex production architecture (Distributed Systems, Data Lakehouse, MLOps, Microservices) AND solving large-scale problems (millions req/day, TBs data).\n"
            "     [84-92] SENIOR_ARCHITECT: Evidence of designing subsystems or deep performance optimization.\n"
            "     [72-83] COMPETENT_PRODUCTION: Proficient with tech stack in real production environment.\n"
            "     [50-71] COMPETENT_BASIC: Common technologies listed but no depth evidence.\n"
            "     [20-49] BASIC_KEYWORD_ONLY: Only keyword listing, no practical evidence.\n"
            "     CEILING GATE: No architecture design evidence -> score MUST be <= 83.\n"
            "   rubricScores.impactMetrics (Dự án & Thành tựu):\n"
            "     [93-100] HIGH_IMPACT_ELITE: REQUIRES >= 3 EXPLICIT quantified metrics with units (% growth, latency ms, cost $, million users).\n"
            "     [80-92] HIGH_IMPACT_METRICS: 1-2 explicit quantified metrics in real projects.\n"
            "     [65-79] STANDARD_COMPLETED_DESCRIBED: Projects described in prose without quantified metrics.\n"
            "     [45-64] STANDARD_COMPLETED_SPARSE: Only project names and technologies listed.\n"
            "     [20-44] ACADEMIC_ONLY: Only graduation projects or coursework.\n"
            "     CEILING GATE: No explicit quantified metrics -> score MUST be <= 79.\n"
            "   rubricScores.enterpriseScale (Quy mô Doanh nghiệp):\n"
            "     [93-100] TIER_1_BIGTECH_ELITE: >= 2 years at Big Tech (Google, Amazon, Meta, Intel, Samsung), Unicorn (Grab, Shopee, MoMo, TikTok), or Top VN Corp (Viettel, Vingroup, VNPT, VNPay).\n"
            "     [80-92] TIER_1_BIGTECH_ENTERPRISE: Experience at Big Tech (<2yr) or >=2yr at Tier 1 Bank (VPBank, Techcombank, MBBank) or Global Corp (Bosch, Accenture).\n"
            "     [65-79] TIER_2_MID_TECH: Reputable Product/Software House (KMS, TMA, NashTech, VNG, Sun*, FPT).\n"
            "     [45-64] STANDARD_SME: Small outsourcing, seed startup, freelance.\n"
            "     CEILING GATE: Only freelance/small startup -> score MUST be <= 64.\n"
            "   rubricScores.education (Học vấn & CS Foundation):\n"
            "     [93-100] TIER_1A_ELITE_PLUS: Master/PhD at QS Top 500 or Honors/Valedictorian at BK, KHTN, VNU, RMIT.\n"
            "     [83-92] TIER_1A_ELITE: Bachelor in CS/IT at top VN tech university (Bach Khoa, KHTN, UET, FPT, RMIT).\n"
            "     [72-82] TIER_1B_ACCREDITED: Bachelor in CS/IT at reputable university (PTIT, SPKT, UEL, HOU, TDT, CTU).\n"
            "     [55-71] STANDARD_ACCREDITED: Bachelor at other universities or non-CS major.\n"
            "     [30-54] NON_DEGREE: College, self-taught, online certificates only.\n"
            "     CEILING GATE: No university degree -> score MUST be <= 54.\n"
            "   rubricScores.certifications (Chứng chỉ Chuyên môn):\n"
            "     [90-100] EXPERT_PRO: Professional-level cert (AWS SA Pro, CKA/CKS, GCP Pro DE, TOGAF, PMP, CISSP).\n"
            "     [75-89] ASSOCIATE_PRACTITIONER: Associate-level cert (AWS SA Associate, Azure Admin, CCNA, Fabric DE).\n"
            "     [40-74] FOUNDATIONAL_ONLINE: Foundational cert (AWS Cloud Practitioner, AZ-900) or Coursera specialization.\n"
            "     [15-39] NONE: No IT certifications mentioned.\n"
            "     CEILING GATE: No certifications -> score MUST be <= 39.\n"
            "   rubricScores.languageProficiency (Năng lực Ngoại ngữ):\n"
            "     [90-100] EXPERT_FLUENT: IELTS>=7.5/TOEIC>=850/C1+ OR worked directly in foreign market (US, EU, SG, JP).\n"
            "     [75-89] WORKING_PROFICIENCY: IELTS 6.0-7.0/TOEIC 650-800 OR regular work with international clients/partners. NOTE: Working at international company automatically qualifies >= 80.\n"
            "     [50-74] BASIC_READING: Basic technical English reading, limited communication.\n"
            "     [20-49] NONE_OR_MINIMAL: No language information or only basic level.\n"
            "6. COMPANY AND JOB TITLE: Always extract 'company' as the employer name (e.g. 'Home Credit Vietnam', 'Bosch Global Software Technologies', 'VPBank') and 'jobTitle' as the job position (e.g. 'Data Engineer', 'Database Developer'). NEVER merge the company name into jobTitle, and NEVER include markdown symbols like '##' in any field.\n"
        )
        if self._output_scope == "entity-inventory":
            system = (
                "Inventory explicit CV employment and project entities using only the supplied "
                "UnifiedDocument. This is a narrow benchmark contract, not a full profile.\n"
                "1. Return every explicit employment and project entity found in the relevant "
                "sections (including English and Vietnamese section headers such as Projects, Personal Projects, "
                "Pet Projects, Key Projects, Academic Projects, Dự án, Dự án cá nhân, Dự án thực tế, "
                "Dự án nổi bật, Dự án học tập, Đề tài), including entries with incomplete optional details.\n"
                "2. For employment return only jobTitle, company, startDate, endDate, and "
                "isCurrent. For projects return only title, role, startDate, endDate, and "
                "isCurrent.\n"
                "3. Do not emit evidence, descriptions, achievements, skills, URLs, contact, "
                "education, languages, certifications, summaries, or any other fields.\n"
                "4. Do not infer absent values. Use null for an unknown scalar and false for "
                "isCurrent unless the CV explicitly indicates a current entry.\n"
                "5. On TWO-COLUMN pages, associate title/company/date only when row alignment "
                "or adjacent labelled blocks supports the association.\n"
            )
        else:
            system = "Extract a CV using only explicit evidence in the supplied UnifiedDocument.\n"
        if self._output_scope == "full":
            system += full_scope_instruction
        if self._output_scope != "entity-inventory":
            system += evidence_instruction + entity_instruction + privacy_and_layout_instruction
        representation_label, document_content = self._document_content(request)
        output_instruction, template = self._output_instruction_and_template()
        user = (
            "Produce one JSON object that validates against the supplied LLM extraction schema. "
            f"Prompt version: {request.prompt_version}. Schema version: {request.schema_version}.\n"
            + output_instruction
            + template
            + "The JSON shape above is a key-only template, not CV content: do not copy or infer "
            "fictional entities.\n" + representation_label + document_content
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "top_p": 0.95,
            "frequency_penalty": 0.1,
            "max_tokens": 8192,
            "n": 1,
        }

        if response_format is not None:
            json_schema = response_format.get("json_schema")
            if (
                self._output_scope in {"entities-only", "entity-inventory"}
                and response_format.get("type") == "json_schema"
                and isinstance(json_schema, dict)
            ):
                response_format = dict(response_format)
                response_format["json_schema"] = {
                    **json_schema,
                    "schema": self._initial_entity_schema(request.json_schema),
                }
            payload["response_format"] = response_format
        return payload

    def _output_instruction_and_template(self) -> tuple[str, str]:
        if self._output_scope == "entity-inventory":
            return (
                "This diagnostic run measures only the employment/project identity inventory. "
                "Return only these exact camelCase keys; do not return evidence or any profile "
                "enrichment:\n",
                '{"experiences":[],"projects":[]}\n',
            )
        if self._output_scope == "entities-only":
            return (
                "This diagnostic run measures only employment and project entities. "
                "Return only these exact camelCase keys; do not return contact, skills, education, "
                "languages, certifications, URLs, summaries, or other fields:\n",
                '{"experiences":[],"projects":[],"evidence":{}}\n',
            )
        return (
            "Use these exact camelCase keys; do not use aliases such as name, workHistory, "
            "or personalProjects:\n",
            '{"candidateName":null,"headline":null,"professionalSummary":null,"email":null,'
            '"phone":null,"skills":[],"experiences":[],"projects":[],"education":[],'
            '"languages":[],"certifications":[],"urls":[],"evidence":{},'
            '"primaryRoleDomain":null,"evaluatedTiers":null,"executiveSummary":null,'
            '"rubricScores":{"technicalDepth":null,"impactMetrics":null,"enterpriseScale":null,'
            '"education":null,"certifications":null,"languageProficiency":null}}\n',
        )

    @staticmethod
    def _entity_only_schema(schema: dict[str, object]) -> dict[str, object]:
        """Restrict experiment output without changing the production DTO contract."""

        scoped = dict(schema)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            scoped["properties"] = {
                name: value
                for name, value in properties.items()
                if name in {"experiences", "projects", "evidence"}
            }
        scoped.pop("required", None)
        return scoped

    def _initial_entity_schema(self, schema: dict[str, object]) -> dict[str, object]:
        scoped = self._entity_only_schema(schema)
        if self._output_scope == "entity-inventory":
            properties = scoped.get("properties")
            if isinstance(properties, dict):
                scoped["properties"] = {
                    name: value
                    for name, value in properties.items()
                    if name in {"experiences", "projects"}
                }
            definitions = scoped.get("$defs")
            if not isinstance(definitions, dict):
                return scoped
            inventory_schema = deepcopy(scoped)
            inventory_definitions = inventory_schema.get("$defs")
            assert isinstance(inventory_definitions, dict)
            allowed_fields = {
                "LLMExtractedExperience": {
                    "jobTitle",
                    "company",
                    "startDate",
                    "endDate",
                    "isCurrent",
                },
                "LLMExtractedProject": {"title", "role", "startDate", "endDate", "isCurrent"},
            }
            for name, allowed in allowed_fields.items():
                definition = inventory_definitions.get(name)
                if not isinstance(definition, dict):
                    continue
                definition_properties = definition.get("properties")
                if isinstance(definition_properties, dict):
                    definition["properties"] = {
                        field: value
                        for field, value in definition_properties.items()
                        if field in allowed
                    }
                required = definition.get("required")
                if isinstance(required, list):
                    definition["required"] = [field for field in required if field in allowed]
            return inventory_schema
        if self._evidence_contract != "required-initial":
            return scoped
        required_schema = deepcopy(scoped)
        definitions = required_schema.get("$defs")
        if not isinstance(definitions, dict):
            return required_schema
        for name in ("LLMExtractedExperience", "LLMExtractedProject"):
            definition = definitions.get(name)
            if not isinstance(definition, dict):
                continue
            required = definition.get("required")
            required_fields = list(required) if isinstance(required, list) else []
            if "evidence" not in required_fields:
                required_fields.append("evidence")
            definition["required"] = required_fields
            properties = definition.get("properties")
            if not isinstance(properties, dict):
                continue
            evidence = properties.get("evidence")
            if isinstance(evidence, dict):
                properties["evidence"] = {**evidence, "minItems": 1}
        return required_schema

    def _document_content(self, request: ExtractionRequest) -> tuple[str, str]:
        if self._input_representation == "layout-transcript":
            return (
                "Layout transcript (complete, including block identifiers):\n",
                serialize_layout_transcript(request.document),
            )
        if self._input_representation == "canonical-text":
            return (
                "Canonical text in source order (page and block labels are evidence anchors only; "
                "there are no spatial layout cues):\n",
                self._canonical_text(request),
            )
        return (
            "Layout-native UnifiedDocument JSON (complete, including pages, blocks, and bboxes):\n",
            json.dumps(
                request.document.model_dump(mode="json", exclude_none=True),
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def _canonical_text(request: ExtractionRequest) -> str:
        lines: list[str] = []
        for page in request.document.pages:
            for block in sorted(page.blocks, key=lambda value: value.reading_order):
                if block.text.strip():
                    lines.append(f"[{page.page_number}|{block.id}] {block.text}")
        return "\n".join(lines)

    def _correction_payload(
        self,
        payload: dict[str, Any],
        response: dict[str, Any],
        field_paths: list[str],
        request: ExtractionRequest,
    ) -> dict[str, Any]:
        content = BeeknoeeStructuredExtractionProvider._completion_content(response)
        correction = dict(payload)
        messages = list(payload["messages"])
        messages.extend(
            [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Correct the previous JSON only. Validation errors by field path: "
                        + "; ".join(field_paths)
                        + ". Preserve every existing experiences, projects, and education entry "
                        "and its semantic content. Do not delete entities, reduce collection "
                        "length, or replace a populated collection with [] to fix validation. "
                        "Correct only invalid fields; use null or [] for an invalid optional "
                        "value. "
                        "Return JSON only."
                    ),
                },
            ]
        )
        correction["messages"] = messages

        if self._evidence_contract == "required-initial":
            response_format = correction.get("response_format")
            if isinstance(response_format, dict):
                json_schema = response_format.get("json_schema")
                if isinstance(json_schema, dict):
                    correction["response_format"] = {
                        **response_format,
                        "json_schema": {
                            **json_schema,
                            "schema": self._entity_only_schema(request.json_schema),
                        },
                    }
        return correction


    def _post_with_retry(
        self,
        payload: dict[str, Any],
        *,
        capture_stage: str,
    ) -> dict[str, Any]:
        last_error: CoreError | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                return self._post(payload, capture_stage=capture_stage)
            except ResponseFormatUnsupported:
                raise
            except CoreError as exc:
                last_error = exc
                if exc.issue.code in {
                    ErrorCode.LLM_AUTHENTICATION_ERROR,
                    ErrorCode.LLM_CONFIGURATION_ERROR,
                }:
                    raise
        assert last_error is not None
        raise last_error

    def _post(self, payload: dict[str, Any], *, capture_stage: str) -> dict[str, Any]:
        try:
            response = self._session.post(
                f"{self._settings.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise CoreError(
                ErrorCode.LLM_TIMEOUT,
                "Beeknoee request timed out.",
                stage="llm_provider",
            ) from exc
        except requests.RequestException as exc:
            raise CoreError(
                ErrorCode.LLM_REQUEST_FAILED,
                "Beeknoee request failed.",
                stage="llm_provider",
                details={"exceptionType": type(exc).__name__},
            ) from exc
        if response.status_code in {400, 422} and self._format_error(response):
            raise ResponseFormatUnsupported()
        if response.status_code in {401, 403}:
            raise CoreError(
                ErrorCode.LLM_AUTHENTICATION_ERROR,
                "Beeknoee authentication failed. Check BEE_API_KEY in .env.",
                stage="llm_provider",
            )
        if response.status_code == 429:
            raise CoreError(
                ErrorCode.LLM_RATE_LIMIT,
                "Beeknoee rate limit reached.",
                stage="llm_provider",
            )
        if response.status_code >= 400:
            raise CoreError(
                ErrorCode.LLM_REQUEST_FAILED,
                "Beeknoee returned an unsuccessful response.",
                stage="llm_provider",
                details={"statusCode": response.status_code},
            )
        try:
            payload_data = response.json()
        except ValueError as exc:
            raise CoreError(
                ErrorCode.LLM_MALFORMED_RESPONSE,
                "Beeknoee returned malformed JSON.",
                stage="llm_provider",
            ) from exc
        if not isinstance(payload_data, dict):
            raise CoreError(
                ErrorCode.LLM_MALFORMED_RESPONSE,
                "Beeknoee returned an unexpected response shape.",
                stage="llm_provider",
            )
        self._capture_raw_exchange(
            payload,
            payload_data,
            getattr(response, "headers", {}),
            capture_stage,
        )
        usage = payload_data.get("usage", {}) if isinstance(payload_data, dict) else {}
        self.last_completion_tokens = usage.get("completion_tokens") or usage.get("completionTokens")
        return payload_data

    def _capture_raw_exchange(
        self,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        response_headers: Any,
        stage: str,
    ) -> None:
        """Persist opt-in synthetic benchmark exchanges with an explicit expiry."""

        if self._raw_capture_dir is None:
            return
        allowed_headers = {
            key: value
            for key, value in dict(response_headers).items()
            if key.casefold()
            in {"x-request-id", "request-id", "x-model-version", "model-version", "x-backend"}
        }
        now = datetime.now(UTC)
        capture = {
            "kind": "smart_cv_raw_capture",
            "stage": stage,
            "capturedAt": now.isoformat(),
            "expiresAt": (now + self._raw_capture_ttl).isoformat(),
            "requestSha256": hashlib.sha256(
                json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "responseHeaders": allowed_headers,
            "request": request_payload,
            "response": response_payload,
        }
        destination = self._raw_capture_dir / (
            f"{now:%Y%m%dT%H%M%S%fZ}-{stage}-{uuid4().hex}.raw-capture.json"
        )
        destination.write_text(
            json.dumps(capture, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _purge_expired_captures(self) -> None:
        assert self._raw_capture_dir is not None
        now = datetime.now(UTC)
        for candidate in self._raw_capture_dir.glob("*.raw-capture.json"):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                expires_at = datetime.fromisoformat(str(data.get("expiresAt", "")))
                if data.get("kind") == "smart_cv_raw_capture" and expires_at <= now:
                    candidate.unlink()
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    @staticmethod
    def _format_error(response: requests.Response) -> bool:
        return any(
            marker in response.text.lower()
            for marker in ("response_format", "json_schema", "json_object", "structured output")
        )

    @staticmethod
    def _completion_content(payload: dict[str, Any]) -> str | dict[str, Any]:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise CoreError(
                ErrorCode.LLM_EMPTY_RESPONSE,
                "Beeknoee returned no completion content.",
                stage="llm_provider",
            ) from exc
        if isinstance(content, (dict, str)):
            return content
        raise CoreError(
            ErrorCode.LLM_EMPTY_RESPONSE,
            "Beeknoee returned empty completion content.",
            stage="llm_provider",
        )

    @classmethod
    def _degeneration_diagnostics(cls, payload: dict[str, Any]) -> dict[str, object] | None:
        content = cls._completion_content(payload)
        finish_reason = None
        with suppress(IndexError, KeyError, TypeError):
            finish_reason = payload["choices"][0].get("finish_reason")
        if (
            isinstance(content, str)
            and finish_reason == "length"
            and _REPEATED_NULL.search(content)
        ):
            return {"classification": "degenerate_repetition", "finishReason": "length"}
        return None

    def _parse_extracted_profile(self, payload: dict[str, Any]) -> LLMExtractedProfile:
        content = self._completion_content(payload)
        if isinstance(content, dict):
            parsed: Any = content
        elif isinstance(content, str) and content.strip():
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if "{" in text and "}" in text:
                first_brace = text.find("{")
                last_brace = text.rfind("}")
                text = text[first_brace:last_brace + 1].strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                self._last_funnel = {"response": {"parseStatus": "invalid_json", "rawText": text[:200]}}
                raise OutputValidationError(["$: valid JSON required"]) from exc
        else:
            self._last_funnel = {"response": {"parseStatus": "empty_json"}}
            raise OutputValidationError(["$: non-empty JSON object required"])
        try:
            normalized_payload = normalize_llm_profile_payload(parsed)
            recognized_counts = _payload_entity_counts(parsed)
            normalized_counts = _payload_entity_counts(normalized_payload)
            # Compare canonicalized identities so date/alias correction does
            # not look like an entity deletion.
            for field_name in _ENTITY_COLLECTIONS:
                recognized_counts[field_name]["entitySignatures"] = normalized_counts[field_name][
                    "entitySignatures"
                ]
                recognized_counts[field_name]["entityEvidenceReferences"] = normalized_counts[
                    field_name
                ]["entityEvidenceReferences"]
            self._last_funnel = {
                "llmRecognized": recognized_counts,
                "afterNormalization": normalized_counts,
            }
            if self._capture_initial_entities:
                self._last_funnel["entitySnapshots"] = _initial_entity_snapshots(parsed)
            normalized = LLMExtractedProfile.model_validate(normalized_payload)
            self._last_funnel["dtoValidated"] = _dto_entity_counts(normalized)
            return normalized
        except ValidationError as exc:
            print(f"[BEEKNOEE DEBUG VALIDATION ERROR] {exc.errors(include_url=False)}")
            field_paths = [
                ".".join(str(part) for part in error["loc"]) or "$"
                for error in exc.errors(include_url=False)
            ]
            self._last_funnel["validationErrors"] = field_paths
            raise OutputValidationError(field_paths) from exc

    @staticmethod
    def _repair_and_load_json(raw_text: str) -> dict[str, Any]:
        """Robust JSON loader that repairs missing commas, trailing commas, and boundary issues."""
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if "{" in text and "}" in text:
            text = text[text.find("{") : text.rfind("}") + 1].strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Fix common LLM formatting flaws
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        fixed = re.sub(r"}\s*\{", "}, {", fixed)
        fixed = re.sub(r'("|\d|true|false|null|\]|\})\s*\n\s*("|\{)', r"\1,\n\2", fixed)
        fixed = re.sub(r'"\s*\n\s*"', '",\n"', fixed)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError as exc:
            # 3. Pinpoint comma insertion at error position
            pos = exc.pos
            if 0 < pos < len(fixed):
                for candidate in (
                    fixed[:pos] + "," + fixed[pos:],
                    fixed[:pos] + '",' + fixed[pos:],
                    fixed[:pos] + "}" + fixed[pos:],
                ):
                    try:
                        return json.loads(candidate)
                    except Exception:
                        continue
            raise

    def _heal_payload_and_parse(self, payload: dict[str, Any]) -> LLMExtractedProfile:
        """Self-heal minor LLM formatting inconsistencies without an extra API call."""
        content = self._completion_content(payload)
        if isinstance(content, dict):
            parsed = dict(content)
        elif isinstance(content, str) and content.strip():
            parsed = self._repair_and_load_json(content)
        else:
            raise ValueError("Empty content cannot be healed")

        normalized = normalize_llm_profile_payload(parsed)
        if not isinstance(normalized, dict):
            raise ValueError("Normalized payload is not a dict")

        # Gracefully handle date coercion errors by setting invalid dates to None
        for collection in ("experiences", "projects", "education"):
            items = normalized.get(collection)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        for field_name in ("startDate", "endDate", "start_date", "end_date", "projectDate", "date"):
                            if field_name in item and isinstance(item[field_name], str):
                                val = str(item[field_name]).strip()
                                if not re.match(r"^\d{4}-\d{2}-\d{2}$", val):
                                    item[field_name] = None
        return LLMExtractedProfile.model_validate(normalized)
