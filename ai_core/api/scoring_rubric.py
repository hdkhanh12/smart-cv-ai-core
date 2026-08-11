"""Deterministic 4-criteria scoring rubric adapter compatible with Search-AI (Codebase B)."""

from __future__ import annotations

import re
from typing import Any

from ai_core.schemas import CVProfile, ProcessingResult

MAX_SCORE = {
    "Học vấn": 15.0,
    "Kinh nghiệm": 40.0,
    "Kỹ năng": 30.0,
    "Chứng chỉ": 15.0,
}

_TOP_UNIVERSITIES = re.compile(
    r"\b(bách khoa|tự nhiên|quốc gia|bưu chính|công nghệ|fpt|rmit|khởi nghiệp|polytechnic|hust|vnu|hcmut|uit)\b",
    re.IGNORECASE,
)

_ADVANCED_SKILL_KEYWORDS = re.compile(
    r"\b(aws|gcp|azure|kubernetes|docker|microservices|system design|ci/cd|llm|rag|pyspark|kafka|bigdata|hadoop|spark)\b",
    re.IGNORECASE,
)

_INTL_CERTS = re.compile(
    r"\b(aws|gcp|azure|databricks|ckad|certified|ielts|toeic|toefl|jlpt|topik|hsk)\b",
    re.IGNORECASE,
)


def _score_education(profile: CVProfile) -> tuple[float, str]:
    if not profile.education:
        return 5.0, "Chưa có thông tin học vấn chi tiết."

    edu = profile.education[0]
    inst = edu.institution or ""
    degree = (edu.degree or "").lower()
    field = (edu.field_of_study or "").lower()

    is_top = bool(_TOP_UNIVERSITIES.search(inst))
    is_stem = any(kw in field or kw in degree for kw in ["công nghệ", "kỹ thuật", "computer", "data", "software", "it"])

    if "thạc sĩ" in degree or "tiến sĩ" in degree or "master" in degree or "phd" in degree or (is_top and is_stem):
        score = 14.0
        reason = f"Tốt nghiệp/Đang học chuyên ngành tại {inst}."
    elif is_top or is_stem or edu.institution:
        score = 11.5
        reason = f"Đào tạo chuyên ngành tại {inst}."
    else:
        score = 8.0
        reason = f"Học vấn tại {inst}."

    return min(15.0, max(0.0, round(score, 1))), reason


def _score_experience(profile: CVProfile) -> tuple[float, str]:
    yoe = profile.total_experience_years or 0.0
    exp_count = len(profile.experiences)

    if yoe >= 4.0:
        score = min(40.0, 32.0 + (yoe - 4.0) * 1.5)
        reason = f"Hơn {yoe:.1f} năm kinh nghiệm làm việc thực chiến."
    elif yoe >= 1.0:
        score = 20.0 + (yoe - 1.0) * 3.8
        reason = f"{yoe:.1f} năm kinh nghiệm làm việc tích lũy qua {exp_count} vị trí."
    elif yoe > 0.0 or exp_count > 0:
        score = 14.0
        reason = f"Dưới 1 năm kinh nghiệm hoặc kinh nghiệm thực tập ({exp_count} vị trí)."
    else:
        score = 5.0
        reason = "Chưa có thông tin kinh nghiệm làm việc chính thức."

    return min(40.0, max(0.0, round(score, 1))), reason


def _score_skills(profile: CVProfile) -> tuple[float, str]:
    skill_names = [s.canonical_name for s in profile.skills]
    skill_count = len(skill_names)

    if skill_count == 0:
        return 5.0, "Liệt kê ít kỹ năng chuyên môn."

    has_advanced = any(_ADVANCED_SKILL_KEYWORDS.search(s) for s in skill_names)

    if skill_count >= 8 and has_advanced:
        score = 27.5
        reason = f"Bộ kỹ năng phong phú ({skill_count} kỹ năng) bao gồm các công nghệ nâng cao."
    elif skill_count >= 4:
        score = 20.0
        reason = f"Kỹ năng tiêu chuẩn đáp ứng tốt yêu cầu công việc ({skill_count} kỹ năng)."
    else:
        score = 12.0
        reason = f"Kỹ năng cơ bản ({skill_count} kỹ năng)."

    return min(30.0, max(0.0, round(score, 1))), reason


def _score_certifications(profile: CVProfile) -> tuple[float, str]:
    certs = profile.certifications
    langs = profile.language_proficiencies
    has_cert = len(certs) > 0 or len(langs) > 0

    if not has_cert:
        return 0.0, "Không có chứng chỉ chuyên môn hay ngoại ngữ nổi bật."

    cert_text = " ".join(certs) + " " + " ".join(l.language + " " + (l.exam or "") for l in langs)
    has_intl = bool(_INTL_CERTS.search(cert_text))

    if has_intl:
        score = 13.0
        reason = "Có chứng chỉ quốc tế hoặc chứng chỉ ngoại ngữ uy tín."
    else:
        score = 8.0
        reason = f"Có {len(certs) + len(langs)} chứng chỉ/ngoại ngữ liệt kê."

    return min(15.0, max(0.0, round(score, 1))), reason


def compute_rubric_scoring(result: ProcessingResult) -> tuple[float, list[dict[str, Any]]]:
    """Compute 4-criteria rubric score matching Codebase B's UI output contract.

    Returns:
        (total_score, scoring_details_list)
    """
    profile = result.profile
    if profile is None:
        return 0.0, [
            {"criterionName": k, "score": 0.0, "maxScore": MAX_SCORE[k], "reason": "Không có thông tin profile."}
            for k in ["Học vấn", "Kinh nghiệm", "Kỹ năng", "Chứng chỉ"]
        ]

    edu_score, edu_reason = _score_education(profile)
    exp_score, exp_reason = _score_experience(profile)
    skl_score, skl_reason = _score_skills(profile)
    crt_score, crt_reason = _score_certifications(profile)

    scoring_details = [
        {"criterionName": "Học vấn", "score": edu_score, "maxScore": MAX_SCORE["Học vấn"], "reason": edu_reason},
        {"criterionName": "Kinh nghiệm", "score": exp_score, "maxScore": MAX_SCORE["Kinh nghiệm"], "reason": exp_reason},
        {"criterionName": "Kỹ năng", "score": skl_score, "maxScore": MAX_SCORE["Kỹ năng"], "reason": skl_reason},
        {"criterionName": "Chứng chỉ", "score": crt_score, "maxScore": MAX_SCORE["Chứng chỉ"], "reason": crt_reason},
    ]

    total_score = round(edu_score + exp_score + skl_score + crt_score, 1)
    return min(100.0, max(0.0, total_score)), scoring_details
