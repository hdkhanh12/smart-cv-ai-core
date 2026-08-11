"""Response formatter matching Search-AI (Codebase B) backend & FE contracts."""

from __future__ import annotations

from typing import Any

from ai_core.api.scoring_rubric import compute_rubric_scoring
from ai_core.schemas import CVProfile, ProcessingResult
from ai_core.taxonomy import extract_job_titles, infer_location, normalize_location


def _highest_education(profile: CVProfile) -> str | None:
    if not profile.education:
        return None
    return profile.education[0].institution or profile.education[0].degree or None


def _estimate_birth_year(profile: CVProfile) -> int | None:
    if profile.date_of_birth:
        return profile.date_of_birth.year
    for edu in profile.education:
        if edu.start_date:
            return edu.start_date.year - 18
        if edu.end_date:
            return edu.end_date.year - 22
    return None


def to_extracted_payload(result: ProcessingResult) -> dict[str, Any]:
    """Format ProcessingResult into the exact JSON schema expected by PUT /api/v1/cvs/{id}/extracted."""
    profile = result.profile
    if profile is None:
        return {
            "status": "Extract-Failed",
            "reason": "Profile extraction yielded null profile",
        }

    job_titles = extract_job_titles(profile)
    normalized_address = infer_location(profile) or normalize_location(profile.address) or profile.address

    companies = []
    for exp in profile.experiences:
        if not exp.company:
            continue
        cleaned_company = exp.company.strip(" \t#*•-").strip()
        if cleaned_company and len(cleaned_company) <= 60 and not any(loc in cleaned_company.lower() for loc in ("vietnam", "hồ chí minh", "hà nội", "singapore")):
            companies.append(cleaned_company)
    unique_companies = list(dict.fromkeys(companies))

    formatted_skills = [
        {"skillName": skill.canonical_name, "years": None, "level": None}
        for skill in profile.skills
    ]

    has_pii = bool(profile.candidate_name or profile.email or profile.phone or profile.address)

    raw_text = ""
    if result.parsed_document and result.parsed_document.raw_text:
        raw_text = result.parsed_document.raw_text
    elif result.unified_document and result.unified_document.markdown:
        raw_text = result.unified_document.markdown

    return {
        "status": "Extracted",
        "candidateName": profile.candidate_name or "",
        "email": profile.email,
        "phone": profile.phone,
        "address": normalized_address,
        "dateOfBirth": None,
        "estimatedBirthYear": _estimate_birth_year(profile),
        "highestEducation": _highest_education(profile),
        "yearsOfExperience": profile.total_experience_years or 0.0,
        "companies": unique_companies,
        "jobTitles": job_titles,
        "languages": profile.languages,
        "skills": formatted_skills,
        "workType": None,
        "summary": result.summary.text if result.summary else "",
        "parsedText": raw_text,
        "profileJson": profile.model_dump_json(by_alias=True),
        "piiFields": {
            "containsPii": has_pii,
            "fields": {
                "candidateName": profile.candidate_name,
                "email": profile.email,
                "phone": profile.phone,
                "address": profile.address,
            },
        },
    }


def to_scored_payload(result: ProcessingResult) -> dict[str, Any]:
    """Format ProcessingResult into the exact JSON schema expected by PUT /api/v1/cvs/{id}/scored."""
    score, scoring_details = compute_rubric_scoring(result)

    a_scores = [s.model_dump(mode="json") for s in result.scores]

    return {
        "status": "Scored",
        "Score": score,
        "ScoringDetails": scoring_details,
        "reason": None,
        "_aScores": a_scores,
    }


def to_embedded_payload(result: ProcessingResult) -> dict[str, Any]:
    """Format ProcessingResult into the exact JSON schema expected by PUT /api/v1/cvs/{id}/embedded."""
    if result.embedding is None or not result.embedding.vector:
        return {
            "status": "Embed-Failed",
            "reason": "Embedding generation yielded null vector",
            "embedding": None,
            "dimension": None,
            "model": "BAAI/bge-m3",
            "version": "v2.3.0-bgem3",
        }
    return {
        "status": "Embedded",
        "reason": None,
        "embedding": result.embedding.vector,
        "dimension": result.embedding.dimension,
        "model": result.embedding.model,
        "version": "v2.3.0-bgem3",
    }
