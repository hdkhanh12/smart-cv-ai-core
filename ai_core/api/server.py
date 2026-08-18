"""Internal FastAPI adapter for profile extraction and reproducible embeddings."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import suppress
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import Field

from ai_core.api.response_contract import (
    to_embedded_payload,
    to_extracted_payload,
    to_query_search_payload,
    to_scored_payload,
)
from ai_core.search import analyze_query_or_jd
from ai_core.schemas import (
    SearchQueryAnalysisResult,
    SearchQueryRequest,
)

from ai_core.embeddings import (
    EMBEDDING_DIMENSION,
    BgeM3Embedder,
    GteMultilingualEmbedder,
    semantic_profile_hash,
)
from ai_core.errors import ContractModel, CoreError
from ai_core.extraction import ExtractionProvider, configured_provider
from ai_core.parsers.unified import hybrid_config
from ai_core.pipeline import process_document

from ai_core.schemas import (
    SCHEMA_VERSION,
    CVProfile,
    EmbeddingResult,
    ProcessingResult,
    ValidationStatus,
)

ProcessDocument = Callable[..., ProcessingResult]


class PublicWarning(ContractModel):
    """Safe warning projection: no issue details or document content."""

    code: str
    message: str
    stage: str


class PublicSkill(ContractModel):
    name: str
    canonical_name: str
    years: float | None = None
    level: str | None = None
    confidence: float


class PublicExperience(ContractModel):
    job_title: str
    company: str | None = None
    location: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    is_current: bool
    description: list[str]
    achievements: list[str]
    skills: list[str]
    confidence: float


class PublicEducation(ContractModel):
    institution: str
    degree: str | None = None
    field_of_study: str | None = None
    location: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    confidence: float


class PublicProject(ContractModel):
    title: str
    role: str | None = None
    description: list[str]
    skills: list[str]
    achievements: list[str]
    start_date: date | None = None
    end_date: date | None = None
    confidence: float


class PublicProfile(ContractModel):
    """UI/persistence profile with raw document evidence intentionally omitted."""

    candidate_name: str | None = None
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    date_of_birth: date | None = None
    headline: str | None = None
    summary: str | None = None
    skills: list[PublicSkill]
    experiences: list[PublicExperience]
    education: list[PublicEducation]
    projects: list[PublicProject]
    languages: list[str]
    certifications: list[str]
    total_experience_years: float | None = None
    validation_status: ValidationStatus
    warnings: list[PublicWarning]


class ProfileMeta(ContractModel):
    profile_schema_version: str
    semantic_profile_hash: str
    extraction_model: str
    input_token_estimate: int | None = None
    output_token_usage: int | None = None
    latency_ms: float


class ProfileFileResponse(ContractModel):
    status: Literal["succeeded", "manual_review"]
    profile: PublicProfile
    meta: ProfileMeta


class EmbeddingResponse(ContractModel):
    """Stable query embedding contract."""

    embedding: list[float] = Field(min_length=EMBEDDING_DIMENSION, max_length=EMBEDDING_DIMENSION)
    dimension: int = Field(EMBEDDING_DIMENSION)


class ProfileEmbeddingRequest(ContractModel):
    profile: CVProfile


class ProfileEmbeddingResponse(EmbeddingResponse):
    semantic_profile_hash: str
    model: str
    model_revision: str
    template_version: str


class ProcessFileResponse(ProfileFileResponse):
    """Rich profile and its initial vector from one pipeline execution."""

    embedding: ProfileEmbeddingResponse


class SubmitCvRequest(ContractModel):
    cvId: int
    fileUrl: str


def _safe_client_error() -> HTTPException:
    """Avoid returning filenames, parser diagnostics, or document contents."""

    return HTTPException(status_code=422, detail={"code": "PROCESSING_UNAVAILABLE"})


def _embedding_response(embedding: EmbeddingResult) -> EmbeddingResponse:
    if (
        embedding.dimension != EMBEDDING_DIMENSION
        or not embedding.normalized
        or len(embedding.vector) != EMBEDDING_DIMENSION
    ):
        raise _safe_client_error()
    return EmbeddingResponse(embedding=embedding.vector, dimension=embedding.dimension)


def _public_profile(profile: CVProfile) -> PublicProfile:
    return PublicProfile(
        candidate_name=profile.candidate_name,
        email=profile.email,
        phone=profile.phone,
        address=profile.address,
        date_of_birth=profile.date_of_birth,
        headline=profile.headline,
        summary=profile.summary,
        skills=[
            PublicSkill(
                name=skill.name,
                canonical_name=skill.canonical_name,
                years=skill.years,
                level=skill.level,
                confidence=skill.confidence,
            )
            for skill in profile.skills
        ],
        experiences=[
            PublicExperience(
                job_title=item.job_title,
                company=item.company,
                location=item.location,
                start_date=item.start_date,
                end_date=item.end_date,
                is_current=item.is_current,
                description=item.description,
                achievements=item.achievements,
                skills=item.skills,
                confidence=item.confidence,
            )
            for item in profile.experiences
        ],
        education=[
            PublicEducation(
                institution=item.institution,
                degree=item.degree,
                field_of_study=item.field_of_study,
                location=item.location,
                start_date=item.start_date,
                end_date=item.end_date,
                confidence=item.confidence,
            )
            for item in profile.education
        ],
        projects=[
            PublicProject(
                title=item.title,
                role=item.role,
                description=item.description,
                skills=item.skills,
                achievements=item.achievements,
                start_date=item.start_date,
                end_date=item.end_date,
                confidence=item.confidence,
            )
            for item in profile.projects
        ],
        languages=profile.languages,
        certifications=profile.certifications,
        total_experience_years=profile.total_experience_years,
        validation_status=profile.validation_status,
        warnings=[
            PublicWarning(code=str(item.code), message=item.message, stage=item.stage)
            for item in profile.warnings
        ],
    )


def _profile_response(result: ProcessingResult) -> ProfileFileResponse:
    if result.profile is None:
        raise _safe_client_error()
    tokens = result.audit.get("tokens", {})
    token_data = tokens if isinstance(tokens, dict) else {}
    input_estimate = token_data.get("documentAfter")
    output_usage = token_data.get("completionTokens")
    return ProfileFileResponse(
        status=(
            "manual_review"
            if result.profile.validation_status == ValidationStatus.MANUAL_REVIEW
            else "succeeded"
        ),
        profile=_public_profile(result.profile),
        meta=ProfileMeta(
            profile_schema_version=SCHEMA_VERSION,
            semantic_profile_hash=semantic_profile_hash(result.profile),
            extraction_model=str(hybrid_config()["model"]),
            input_token_estimate=input_estimate if isinstance(input_estimate, int) else None,
            output_token_usage=output_usage if isinstance(output_usage, int) else None,
            latency_ms=result.timings_ms.get("total", 0.0),
        ),
    )


def _profile_embedding_response(
    profile: CVProfile, embedding: EmbeddingResult
) -> ProfileEmbeddingResponse:
    response = _embedding_response(embedding)
    return ProfileEmbeddingResponse(
        embedding=response.embedding,
        dimension=response.dimension,
        semantic_profile_hash=semantic_profile_hash(profile),
        model=embedding.model,
        model_revision=embedding.model_revision,
        template_version=embedding.template_version,
    )


async def _save_upload(input: UploadFile) -> Path:
    suffix = Path(input.filename or "").suffix.lower()
    with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        while chunk := await input.read(1024 * 1024):
            temp_file.write(chunk)
    return temp_path


async def _report_failed(http: httpx.AsyncClient, cv_id: int, step: str, reason: str):
    backend_base_url = os.getenv("BACKEND_BASE_URL", "http://10.10.32.48").rstrip("/")
    url = (
        f"{backend_base_url}/api/v1/cvs/{cv_id}/extracted"
        if step == "Extract"
        else f"{backend_base_url}/api/v1/cvs/{cv_id}/scored"
    )
    status = "Extract-Failed" if step == "Extract" else "Score-Failed"
    payload = {"status": status, "reason": reason}
    try:
        r = await http.put(url, json=payload)
        print(f"[PIPELINE] 🔴 CV #{cv_id} status={status} -> {r.status_code}")
    except Exception as e:
        print(f"[PIPELINE] ⚠️ Failed to report error for CV #{cv_id}: {e}")


async def _process_backend_cv_pipeline(
    cv_id: int,
    file_url: str | None,
    file_bytes: bytes | None,
    filename: str | None,
    shared_embedder: BgeM3Embedder | None,
    process_fn: ProcessDocument,
):
    backend_base_url = os.getenv("BACKEND_BASE_URL", "http://10.10.32.48").rstrip("/")
    extracted_url = f"{backend_base_url}/api/v1/cvs/{cv_id}/extracted"
    scored_url = f"{backend_base_url}/api/v1/cvs/{cv_id}/scored"
    embedded_url = f"{backend_base_url}/api/v1/cvs/{cv_id}/embedded"

    temp_path: Path | None = None
    async with httpx.AsyncClient(timeout=60) as http:
        try:
            if file_url:
                r = await http.get(file_url)
                if r.status_code != 200:
                    raise ValueError(f"Failed to download file from {file_url}, status code {r.status_code}")
                content = r.content
                suffix = Path(file_url.split("?")[0]).suffix.lower() or ".pdf"
            elif file_bytes:
                content = file_bytes
                suffix = Path(filename or "").suffix.lower() or ".pdf"
            else:
                raise ValueError("Neither fileUrl nor file_bytes provided")

            with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)

            result = await run_in_threadpool(
                process_fn,
                temp_path,
                embed=True,
                embedder=shared_embedder,
            )

            if result.status.value == "failed" or result.profile is None:
                err_msg = "; ".join(e.message for e in result.errors) if result.errors else "Extraction failed"
                await _report_failed(http, cv_id, "Extract", err_msg)
                return

            extracted_payload = to_extracted_payload(result)
            scored_payload = to_scored_payload(result)
            embedded_payload = to_embedded_payload(result)

            try:
                r_ext = await http.put(extracted_url, json=extracted_payload)
                print(f"[PIPELINE] ✅ PUT /extracted CV #{cv_id} -> {r_ext.status_code}")
            except Exception as e:
                print(f"[PIPELINE] ⚠️ Error sending PUT /extracted for CV #{cv_id}: {e}")

            try:
                r_score = await http.put(scored_url, json=scored_payload)
                print(f"[PIPELINE] 🏆 PUT /scored CV #{cv_id} -> {r_score.status_code}")
            except Exception as e:
                print(f"[PIPELINE] ⚠️ Error sending PUT /scored for CV #{cv_id}: {e}")

            try:
                r_embed = await http.put(embedded_url, json=embedded_payload)
                print(f"[PIPELINE] 🧬 PUT /embedded CV #{cv_id} -> {r_embed.status_code}")
            except Exception as e:
                print(f"[PIPELINE] ⚠️ Error sending PUT /embedded for CV #{cv_id}: {e}")

        except Exception as e:
            print(f"[PIPELINE] ❌ Pipeline error for CV #{cv_id}: {e}")
            await _report_failed(http, cv_id, "Extract", str(e))
        finally:
            if temp_path is not None:
                with suppress(FileNotFoundError):
                    temp_path.unlink()


def create_app(
    *,
    embedder: BgeM3Embedder | None = None,
    process: ProcessDocument = process_document,
    provider: ExtractionProvider | None = None,
) -> FastAPI:
    """Create the internal service with BGE-M3 (1024d) embedding model."""

    app = FastAPI(
        title="Smart CV Embedding & AI Handle API",
        version="2.3.0",
        description="Internal bridge for CV processing, BGE-M3 1024d embeddings, and Search-AI backend pipeline.",
    )
    shared_embedder = embedder or BgeM3Embedder()
    try:
        shared_provider = provider or configured_provider()
    except Exception:
        shared_provider = None


    @app.on_event("startup")
    async def warmup_bge_m3_embedder():
        """Pre-load BGE-M3 (1024d) model into RAM on server startup to eliminate 15s cold-start latency."""
        print("[STARTUP WARMUP] 🧠 Pre-loading BGE-M3 (1024d) embedding model into RAM...")
        try:
            res = await run_in_threadpool(shared_embedder.embed_text, "warmup bge-m3 model initialization")
            print(f"[STARTUP WARMUP] ✅ BGE-M3 model ready in RAM! (warmup latency: {res.duration_ms:.2f}ms)")
        except Exception as e:
            print(f"[STARTUP WARMUP] ⚠️ BGE-M3 warmup warning: {e}")

    @app.post("/api/v1/cv/ai-handle")
    @app.post("/ai/handle")
    async def submit_cv_backend(
        request: Request,
        background_tasks: BackgroundTasks,
    ):
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            body = await request.json()
            cv_id = int(body.get("cvId") or body.get("id"))
            file_url = body.get("fileUrl")
            background_tasks.add_task(
                _process_backend_cv_pipeline,
                cv_id,
                file_url,
                None,
                None,
                shared_embedder,
                process,
            )
            return {
                "success": True,
                "statusCode": 200,
                "message": f"CV #{cv_id} đã được đưa vào hàng xử lý.",
                "data": None,
            }
        else:
            form = await request.form()
            cv_id = int(form.get("id") or form.get("cvId"))
            upload_file: UploadFile = form.get("file")
            file_bytes = await upload_file.read()
            background_tasks.add_task(
                _process_backend_cv_pipeline,
                cv_id,
                None,
                file_bytes,
                upload_file.filename,
                shared_embedder,
                process,
            )
            return {
                "success": True,
                "statusCode": 200,
                "message": f"CV #{cv_id} đã được đưa vào hàng xử lý.",
                "data": None,
            }

    @app.post("/v1/cv/process", response_model=ProcessFileResponse)
    async def process_cv(input: Annotated[UploadFile, File(...)]) -> ProcessFileResponse:
        temp_path: Path | None = None
        try:
            temp_path = await _save_upload(input)
            result = await run_in_threadpool(
                process,
                temp_path,
                embed=True,
                embedder=shared_embedder,
            )
            if result.profile is None or result.embedding is None:
                raise _safe_client_error()
            if result.profile.validation_status == ValidationStatus.MANUAL_REVIEW:
                raise _safe_client_error()
            profile_response = _profile_response(result)
            return ProcessFileResponse(
                status=profile_response.status,
                profile=profile_response.profile,
                meta=profile_response.meta,
                embedding=_profile_embedding_response(result.profile, result.embedding),
            )
        except CoreError as exc:
            raise _safe_client_error() from exc
        except ValueError as exc:
            raise _safe_client_error() from exc
        finally:
            await input.close()
            if temp_path is not None:
                with suppress(FileNotFoundError):
                    temp_path.unlink()

    @app.post("/v1/embeddings/profile", response_model=ProfileEmbeddingResponse)
    async def embed_profile(request: ProfileEmbeddingRequest) -> ProfileEmbeddingResponse:
        try:
            result = await run_in_threadpool(shared_embedder.embed_profile, request.profile)
        except ValueError as exc:
            raise _safe_client_error() from exc
        return _profile_embedding_response(request.profile, result)

    @app.post("/v1/embeddings/text", response_model=EmbeddingResponse)
    async def embed_text(input: Annotated[str, Form(min_length=1)]) -> EmbeddingResponse:
        try:
            result = await run_in_threadpool(shared_embedder.embed_text, input)
        except ValueError as exc:
            raise _safe_client_error() from exc
        return _embedding_response(result)

    @app.post(
        "/v1/embeddings/file",
        response_model=EmbeddingResponse,
        summary="Demo endpoint: embed one CV without returning its profile.",
    )
    async def embed_file(input: Annotated[UploadFile, File(...)]) -> EmbeddingResponse:
        temp_path: Path | None = None
        try:
            temp_path = await _save_upload(input)
            result = await run_in_threadpool(
                process,
                temp_path,
                embed=True,
                embedder=shared_embedder,
            )
            if result.embedding is None or result.profile is None:
                raise _safe_client_error()
            if result.profile.validation_status == ValidationStatus.MANUAL_REVIEW:
                raise _safe_client_error()
            return _embedding_response(result.embedding)
        except CoreError as exc:
            raise _safe_client_error() from exc
        except ValueError as exc:
            raise _safe_client_error() from exc
        finally:
            await input.close()
            if temp_path is not None:
                with suppress(FileNotFoundError):
                    temp_path.unlink()

    @app.post("/api/v1/embeddings/search")
    @app.post(
        "/v1/embeddings/search",
        summary="C# Backend Search API: extract filters and 1024d BGE-M3 vector for Search Query or JD.",
    )
    async def search_query_backend(payload: SearchQueryRequest):
        try:
            result = await analyze_query_or_jd(payload.text, shared_embedder, provider=shared_provider)
            return to_query_search_payload(result)
        except Exception as exc:
            raise _safe_client_error() from exc

    @app.post(
        "/v1/search/analyze",
        response_model=SearchQueryAnalysisResult,
        summary="Internal rich Search API: extract filters, canonical representation, and 1024d BGE-M3 vector with latency metrics.",
    )
    async def analyze_search_query(payload: SearchQueryRequest) -> SearchQueryAnalysisResult:
        try:
            return await analyze_query_or_jd(payload.text, shared_embedder, provider=shared_provider)
        except Exception as exc:
            raise _safe_client_error() from exc


    return app



app = create_app()
