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

    _STANDALONE_LOCATIONS = {
        "vietnam", "viet nam", "việt nam", "hồ chí minh", "ho chi minh", "hcm", "hcmc",
        "tp.hcm", "tp hcm", "tp hồ chí minh", "hà nội", "ha noi", "hn", "da nang", "đà nẵng",
        "singapore", "remote", "hybrid", "on-site", "onsite", "vietnam remote", "hcm, vietnam",
        "ho chi minh city, vietnam", "hanoi, vietnam", "da nang, vietnam"
    }

    companies = []
    for exp in profile.experiences:
        company_val = exp.company
        # If company is missing, check if it was embedded in job_title or description
        if not company_val:
            title_clean = (exp.job_title or "").lstrip("#*•- ").strip()
            if " - " in title_clean:
                comp_cand = title_clean.split(" - ", 1)[0].strip()
                if comp_cand and len(comp_cand) <= 60 and comp_cand.lower() not in _STANDALONE_LOCATIONS:
                    companies.append(comp_cand)
            continue

        cleaned_company = company_val.lstrip(" \t#*•-").strip()
        if (
            cleaned_company
            and len(cleaned_company) <= 80
            and cleaned_company.lower() not in _STANDALONE_LOCATIONS
        ):
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
    """Format ProcessingResult into clean V3 IT scoring JSON schema.

    Emits only the streamlined V3 multi-criteria scoring payload with
    Vietnamese-only summary, removing legacy and redundant fields.
    """
    from ai_core.api.scoring_v2 import compute_it_rubric_scoring

    v3_scoring: dict[str, Any] = {}
    if result.profile is not None:
        v3_scoring = compute_it_rubric_scoring(result.profile)

    # Clean Vietnamese summary (strip any leftover [VI] / [EN] tags)
    summary_text = v3_scoring.get("summary", "")
    if "[VI]" in summary_text:
        summary_text = summary_text.split("[VI]", 1)[-1].strip()
    elif "[EN]" in summary_text:
        summary_text = summary_text.split("[EN]", 1)[-1].split("[VI]")[0].strip()

    final_score = v3_scoring.get("Score", 0.0)

    return {
        "status": "Scored",
        "Score": final_score,
        "candidateName": result.profile.candidate_name if result.profile else None,
        "summary": summary_text,
        "criteriaScores": v3_scoring.get("criteriaScores", {}),
        "scoringRationale": v3_scoring.get("scoringRationale", ""),
        "seniorityCalibratedLevel": v3_scoring.get("seniorityCalibratedLevel", ""),
        "scoringVersion": v3_scoring.get("scoringVersion", "v3_rubric"),
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


def to_query_search_payload(result: Any) -> dict[str, Any]:
    """Format exact JSON schema expected by Backend C# for POST /v1/embeddings/search."""
    filters = result.filters
    embedding = result.embedding
    return {
        "MinYearsOfExperience": filters.min_experience_years,
        "MaxYearsOfExperience": filters.max_experience_years,
        "Skills": filters.skills if filters.skills else None,
        "JobTitles": filters.job_titles if filters.job_titles else None,
        "Companies": filters.companies if filters.companies else None,
        "Languages": filters.languages if filters.languages else None,
        "HighestEducation": filters.highest_education,
        "WorkType": filters.work_type,
        "Location": filters.location,
        "Embedding": embedding.vector if embedding else None,
        "Dimension": embedding.dimension if embedding else 1024,
        "Version": "embedding-v1",
    }

