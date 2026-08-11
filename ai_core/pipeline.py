"""Top-level standalone processing orchestration."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ai_core.embeddings import GteMultilingualEmbedder
from ai_core.extraction import (
    ExtractionProvider,
    build_request,
    configured_provider,
)
from ai_core.extraction.privacy import (
    apply_local_contact,
    apply_verified_identity,
    deanonymize_profile,
    extract_filename_identity,
    extract_pdf_header_identity,
    mask_document,
    resolve_local_contact,
    resolve_verified_identity,
)
from ai_core.extraction.pruning import prune_document
from ai_core.parsers import parse_document
from ai_core.parsers.unified import (
    ParsedDocumentAdapter,
    PyMuPdf4LlmAdapter,
    configured_fallback_extractor,
    configured_main_extractor,
    effective_document_text,
    extract_with_fallback,
)
from ai_core.reconciliation import bind_exact_entity_evidence, reconcile_profile
from ai_core.schemas import (
    CVProfile,
    DiagnosticStatus,
    ProcessingResult,
    ProcessingStatus,
    ValidationStatus,
)
from ai_core.scoring import score_cv_quality, score_profile_completeness
from ai_core.scoring.cache import cache_key, load_scores, store_scores
from ai_core.scoring.features import FEATURE_VERSION, document_score_features
from ai_core.summarization import summarize_profile
from ai_core.validation import DEFAULT_MAX_FILE_SIZE, validate_input


def _profile_entity_counts(profile: CVProfile) -> dict[str, dict[str, int]]:
    """Emit counts only, never raw profile fields or evidence text."""

    result: dict[str, dict[str, int]] = {}
    for field_name in ("experiences", "projects", "education"):
        entries = getattr(profile, field_name)
        result[field_name] = {
            "entities": len(entries),
            "entitiesWithEvidence": sum(bool(entry.evidence) for entry in entries),
            "evidenceReferences": sum(len(entry.evidence) for entry in entries),
        }
    return result


def _estimate_description_token_impact(profile: CVProfile) -> dict[str, object]:
    """Estimate how many tokens come from experience descriptions and achievements.

    Sprint-1 (2026-08-10): diagnostic helper requested by the team to evaluate
    whether removing verbose work-experience bullet-points would materially
    reduce LLM input cost.

    Returns:
        tokens: estimated token count for all description + achievement strings
        characters: raw character count for reference
        entriesAnalyzed: number of experience entries scanned
        avgTokensPerEntry: average per-entry cost
        estimationMethod: always ``heuristic_chars_div4`` (same as pruning.py)
        interpretationNote: human-readable guidance
    """
    from ai_core.extraction.pruning import estimate_tokens

    total_chars = 0
    entry_count = len(profile.experiences)
    for entry in profile.experiences:
        total_chars += sum(len(line) for line in entry.description)
        total_chars += sum(len(line) for line in entry.achievements)

    token_estimate = estimate_tokens("x" * total_chars)
    avg = round(token_estimate / entry_count, 1) if entry_count else 0.0
    return {
        "tokens": token_estimate,
        "characters": total_chars,
        "entriesAnalyzed": entry_count,
        "avgTokensPerEntry": avg,
        "estimationMethod": "heuristic_chars_div4",
        "interpretationNote": (
            f"Description+achievement content = ~{token_estimate} input tokens "
            f"across {entry_count} experience entries (avg {avg} tkn/entry). "
            "Compare with audit.tokens.documentAfter for the full document token count."
        ),
    }


def process_document(
    path: str | Path,
    *,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
    debug_dir: Path | None = None,
    extraction_provider: ExtractionProvider | None = None,
    embed: bool = False,
    embedder: GteMultilingualEmbedder | None = None,
    capture_reconciliation_dispositions: bool = False,
    # Sprint-1 (2026-08-10): Changed default from "none" → "exact-entity".
    # bind_exact_entity_evidence() is additive-only — it adds evidence to entries
    # that lack it, never removing existing evidence — so risk of regression is 0.
    # "none" caused all 5 companies to be dropped from the Aris_Le CV because the
    # reconciliation drop-rule checked bool(evidence) on an empty list.
    local_evidence_binding: str = "exact-entity",
    header_masking_mode: str = "selective",
) -> ProcessingResult:
    """Validate and parse one CV without any external integration."""

    pipeline_started = time.perf_counter()
    validation_started = time.perf_counter()
    validated = validate_input(path, max_size_bytes=max_size_bytes)
    validation_ms = (time.perf_counter() - validation_started) * 1000
    parsing_started = time.perf_counter()
    parsed = parse_document(validated)
    parsing_ms = (time.perf_counter() - parsing_started) * 1000
    unified_started = time.perf_counter()
    if validated.metadata.extension == ".pdf":
        try:
            unified = extract_with_fallback(
                validated,
                configured_main_extractor(),
                configured_fallback_extractor(),
            )
        except Exception:
            # Parser failure is a valid fallback trigger. PyMuPDF4LLM is the
            # lightweight rollback adapter; diagnostics still decide whether
            # its output may proceed or requires manual review.
            try:
                unified = extract_with_fallback(
                    validated,
                    PyMuPdf4LlmAdapter(),
                    configured_fallback_extractor(),
                )
            except Exception:
                parsed_adapter = ParsedDocumentAdapter([page.text for page in parsed.pages])
                unified = parsed_adapter.extract(validated)
    else:
        unified = ParsedDocumentAdapter([page.text for page in parsed.pages]).extract(validated)
    unified_ms = (time.perf_counter() - unified_started) * 1000
    extraction_started = time.perf_counter()
    reconciliation_ms = 0.0
    audit: dict[str, object] = {
        "sourceHash": validated.metadata.sha256,
        "timestamps": {"startedAt": pipeline_started},
        "diagnostics": unified.diagnostics.model_dump(mode="json"),
        "blockCount": sum(len(page.blocks) for page in unified.pages),
    }
    if validated.metadata.extension == ".pdf":
        source_text_characters = len(parsed.normalized_text.strip())
        audit["ocr"] = {
            "sourceIsImageOnlyPdf": source_text_characters == 0,
            "sourceTextCharacterCount": source_text_characters,
            "fallbackUsed": unified.extractor == "docling-rapidocr",
            "finalExtractor": unified.extractor,
            "rawUnifiedCharacterCount": len(unified.markdown.strip()),
            "effectiveUnifiedCharacterCount": len(effective_document_text(unified.markdown)),
            "diagnosticStatus": unified.diagnostics.status.value,
        }
    if unified.diagnostics.status != DiagnosticStatus.USABLE:
        profile = CVProfile(
            validation_status=ValidationStatus.MANUAL_REVIEW,
            warnings=list(unified.diagnostics.issues),
        )
    else:
        provider = extraction_provider or configured_provider()
        # This boundary is mandatory. The remote provider never receives the
        # original UnifiedDocument, irrespective of CLI or caller settings.
        header_identity = (
            extract_pdf_header_identity(str(validated.path))
            if validated.metadata.extension == ".pdf"
            else resolve_verified_identity(unified)
        )
        verified_identity = header_identity or extract_filename_identity(validated.path)
        masking = mask_document(
            unified,
            local_identity=verified_identity,
            header_masking_mode=header_masking_mode,
        )
        request_document = masking.document
        pruned = prune_document(request_document)
        audit["tokens"] = {
            "documentBefore": pruned.input_tokens_before,
            "documentAfter": pruned.input_tokens_after,
        }
        audit["masking"] = {
            "enabled": True,
            "policy": "mandatory_local_pii_boundary",
            "headerMaskingMode": header_masking_mode,
            "mapping": masking.mapping,
            "reconciliationInput": "original_local_document",
            "privacyRevision": "v3_single_semantic_pipeline",
            "headerBlockCount": sum(
                block.text == "[HEADER_REDACTED]"
                for page in masking.document.pages
                for block in page.blocks
            ),
            "counts": {
                "names": sum(token == "[ANON_NAME]" for token in (masking.mapping or {}).values()),
                "emails": sum("[ANON_EMAIL" in token for token in (masking.mapping or {}).values()),
                "phones": sum("[ANON_PHONE" in token for token in (masking.mapping or {}).values()),
                "urls": sum("[ANON_URL" in token for token in masking.mapping.values()),
            },
        }
        extracted_profile = provider.extract(
            build_request(
                pruned.document,
                local_identity=verified_identity,
                header_masking_mode=header_masking_mode,
            )
        )
        if hasattr(provider, "last_completion_tokens") and provider.last_completion_tokens:
            audit["tokens"]["completionTokens"] = provider.last_completion_tokens
        reconciliation_started = time.perf_counter()
        # The provider can only echo placeholders. Restore those exact local
        # values (including evidence text) before grounding against the
        # original local UnifiedDocument; this never sends PII remotely.
        extracted_profile = deanonymize_profile(extracted_profile, masking.mapping)
        local_contact = resolve_local_contact(unified)
        extracted_profile = apply_local_contact(extracted_profile, local_contact)
        llm_candidate_name = extracted_profile.candidate_name
        extracted_profile = apply_verified_identity(extracted_profile, verified_identity)
        audit["identityResolution"] = {
            "source": verified_identity.source if verified_identity else "unresolved_header",
            "blockId": verified_identity.evidence.block_id
            if verified_identity and verified_identity.evidence
            else None,
            "pageNumber": verified_identity.evidence.page_number
            if verified_identity and verified_identity.evidence
            else (1 if verified_identity else None),
            "fontSize": verified_identity.font_size if verified_identity else None,
            "bbox": verified_identity.bbox if verified_identity else None,
            "confidence": verified_identity.confidence if verified_identity else None,
            "filenameHash": hashlib.sha256(
                validated.metadata.source_name.encode("utf-8")
            ).hexdigest()
            if verified_identity and verified_identity.source == "filename_fallback"
            else None,
            "llmCandidateOverridden": bool(
                verified_identity and llm_candidate_name != verified_identity.name
            ),
        }
        binding_audit: dict[str, object] | None = None
        if local_evidence_binding == "exact-entity":
            extracted_profile, binding_audit = bind_exact_entity_evidence(
                extracted_profile, unified
            )
        elif local_evidence_binding != "none":
            raise ValueError(f"Unsupported local evidence binding: {local_evidence_binding}")
        pre_reconciliation_counts = _profile_entity_counts(extracted_profile)
        dispositions: list[dict[str, object]] | None = (
            [] if capture_reconciliation_dispositions else None
        )
        profile = reconcile_profile(
            extracted_profile,
            unified,
            provider_name=provider.name,
            allow_identity_fallback=False,
            entity_dispositions=dispositions,
        )
        audit["reconciliation"] = {
            "validationStatus": profile.validation_status.value,
            "experienceCount": len(profile.experiences),
            "projectCount": len(profile.projects),
            "warningCodes": [warning.code.value for warning in profile.warnings],
            "entityDispositions": dispositions,
        }
        if binding_audit is not None:
            audit["localEvidenceBinding"] = binding_audit
        provider_audit = getattr(provider, "last_audit", {})
        if not isinstance(provider_audit, dict):
            provider_audit = {}
        provider_funnel = provider_audit.get("funnel")
        audit["extractionFunnel"] = {
            "provider": provider_funnel if isinstance(provider_funnel, dict) else {},
            "beforeReconciliation": pre_reconciliation_counts,
            "afterReconciliation": _profile_entity_counts(profile),
        }
        reconciliation_ms = (time.perf_counter() - reconciliation_started) * 1000
        audit_tokens = cast(dict[str, Any], audit["tokens"])
        provider_tokens = {
            key: value for key, value in provider_audit.items() if key in {"input", "output"}
        }
        # Sprint-1 (2026-08-10): Token breakdown for description-pruning analysis.
        # Colleagues can compare descriptionImpact.tokens vs total to evaluate
        # whether removing work-experience descriptions would materially reduce cost.
        description_impact = _estimate_description_token_impact(extracted_profile)
        audit["tokens"] = {
            **audit_tokens,
            **provider_tokens,
            "descriptionImpact": description_impact,
        }
    extraction_ms = (time.perf_counter() - extraction_started) * 1000
    summary_started = time.perf_counter()
    summary = summarize_profile(profile)
    summary_ms = (time.perf_counter() - summary_started) * 1000
    scoring_started = time.perf_counter()
    score_features = document_score_features(unified)
    score_key = cache_key(
        validated.metadata.sha256,
        anonymize=True,
        feature_hash=score_features.feature_hash,
        validation_status=profile.validation_status,
    )
    score_cache_path = Path("outputs") / "score-cache" / f"{score_key}.json"
    scores = load_scores(score_cache_path)
    score_cache_hit = scores is not None
    if scores is None:
        scores = [
            score_profile_completeness(profile, unified),
            score_cv_quality(profile, unified),
        ]
        store_scores(score_cache_path, scores)
    scoring_ms = (time.perf_counter() - scoring_started) * 1000
    audit["scoreFeatures"] = {
        "version": FEATURE_VERSION,
        "source": "unified_document",
        "hash": score_features.feature_hash,
        "skillCount": len(score_features.skill_types),
        "achievementCount": score_features.achievement_count,
        "cacheKey": score_key,
        "cacheHit": score_cache_hit,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    embedding = None
    embedding_ms = 0.0
    if embed and profile.validation_status != ValidationStatus.MANUAL_REVIEW:
        embedding_started = time.perf_counter()
        embedding = (embedder or GteMultilingualEmbedder()).embed_profile(profile)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        audit["embedding"] = {
            "model": embedding.model,
            "dimension": embedding.dimension,
            "latencyMs": embedding.duration_ms,
            "sourceHash": embedding.source_hash,
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": "profile_embedding",
        }
    elif embed:
        audit["embedding"] = {"status": "skipped_manual_review"}
    if debug_dir is not None:
        artifact_dir = debug_dir / validated.metadata.sha256
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "raw_text.txt").write_text(parsed.raw_text, encoding="utf-8")
        (artifact_dir / "normalized_text.txt").write_text(
            parsed.normalized_text,
            encoding="utf-8",
        )
        (artifact_dir / "profile.json").write_text(
            profile.model_dump_json(by_alias=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "unified_document.json").write_text(
            unified.model_dump_json(by_alias=True, indent=2) + "\n",
            encoding="utf-8",
        )
    total_ms = (time.perf_counter() - pipeline_started) * 1000
    return ProcessingResult(
        status=ProcessingStatus.SUCCEEDED,
        source_id=validated.metadata.sha256,
        metadata=parsed.metadata,
        parsed_document=parsed,
        unified_document=unified,
        profile=profile,
        summary=summary,
        embedding=embedding,
        scores=scores,
        warnings=[
            *parsed.warnings,
            *profile.warnings,
            *(warning for score in scores for warning in score.warnings),
        ],
        timings_ms={
            "inputValidation": round(validation_ms, 3),
            "parsingAndPreprocessing": round(parsing_ms, 3),
            "unifiedExtraction": round(unified_ms, 3),
            "profileExtraction": round(extraction_ms, 3),
            "reconciliation": round(reconciliation_ms, 3),
            "summarization": round(summary_ms, 3),
            "scoring": round(scoring_ms, 3),
            "embedding": round(embedding_ms, 3),
            "total": round(total_ms, 3),
        },
        audit=audit,
    )
