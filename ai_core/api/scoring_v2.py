"""V2/V3 IT Evidence-Based Scoring Rubric: 6-criteria x 100-point scale.

V2: Maps AI-assessed tier labels (from evaluated_tiers) into deterministic
numeric scores on CPU in < 0.2ms with zero API cost.

V3: Passthrough AI-calibrated rubric scores (from rubric_scores) with
Ceiling Gate validation guardrail. Falls back to V2 tier mapping when
rubric_scores is unavailable.
"""

from __future__ import annotations

import re
from typing import Any

from ai_core.schemas import CriterionScore, CVProfile, EvaluatedTiers, ProcessingResult

# ---------------------------------------------------------------------------
# Criterion display names (Vietnamese, for HR tooltip)
# ---------------------------------------------------------------------------
CRITERIA_NAMES: dict[str, str] = {
    "EDUCATION": "Học vấn & CS Foundation",
    "EXPERIENCE_YEARS": "Thâm niên Kinh nghiệm",
    "ENTERPRISE_SCALE": "Quy mô Doanh nghiệp",
    "TECHNICAL_SKILLS": "Kỹ năng & Công nghệ",
    "PROJECTS_PORTFOLIO": "Dự án & Thành tựu",
    "CERTIFICATIONS_LANGUAGE": "Chứng chỉ & Ngoại ngữ",
}

# ---------------------------------------------------------------------------
# Tier → Score mapping tables (Section 2 of the V2 specification)
# ---------------------------------------------------------------------------
_EDUCATION_SCORE: dict[str, float] = {
    "TIER_1A_ELITE": 100.0,
    "TIER_1B_ACCREDITED_TECH": 90.0,
    "STANDARD_ACCREDITED": 75.0,
    "ASSOCIATE_OTHER": 60.0,
    "NON_DEGREE": 40.0,
}

_ENTERPRISE_SCORE: dict[str, float] = {
    "TIER_1_BIGTECH_ENTERPRISE": 100.0,
    "TIER_2_MID_TECH": 80.0,
    "STANDARD_SME": 60.0,
}

_SKILL_EVIDENCE_SCORE: dict[str, float] = {
    "ADVANCED_EVIDENCE_BASED": 95.0,
    "COMPETENT_PRODUCTION": 85.0,
    "BASIC_KEYWORD_ONLY": 65.0,
}

_PROJECT_QUALITY_SCORE: dict[str, float] = {
    "HIGH_IMPACT_METRICS": 95.0,
    "STANDARD_COMPLETED": 80.0,
    "ACADEMIC_ONLY": 50.0,
}

_CERT_SCORE: dict[str, float] = {
    "EXPERT_PRO": 95.0,
    "ASSOCIATE_PRACTITIONER": 85.0,
    "BASIC_FOUNDATIONAL": 50.0,
    "NONE": 20.0,
}

_LANG_SCORE: dict[str, float] = {
    "EXPERT_FLUENT": 95.0,
    "WORKING_PROFICIENCY": 85.0,
    "BASIC_ELEMENTARY": 50.0,
    "NONE": 20.0,
}

# ---------------------------------------------------------------------------
# Heuristic fallback patterns (used when evaluated_tiers is None)
# ---------------------------------------------------------------------------
_TOP_UNIVERSITIES = re.compile(
    r"\b(bách khoa|bach khoa|tự nhiên|quốc gia|bưu chính|công nghệ thông tin|fpt|rmit|"
    r"sư phạm kỹ thuật|tôn đức thắng|cần thơ|đà nẵng|uit|uet|hcmut|hust|hus|hcmus|ptit|spkt|"
    r"công nghiệp hà nội|hanoi university of science|vietnam national university|"
    r"university of science|university of technology|technology and education|"
    r"stanford|mit|nus|monash|cmu|harvard|oxford|cambridge|berkeley|epfl|eth zurich)\b",
    re.IGNORECASE,
)

_TIER_1B_UNIVERSITIES = re.compile(
    r"\b(fpt|rmit|spkt|sư phạm kỹ thuật|bưu chính|ptit|tôn đức thắng|"
    r"công nghiệp hà nội|cần thơ|bách khoa đà nẵng|vku|danang university of technology)\b",
    re.IGNORECASE,
)

_HONORS_DEGREE = re.compile(
    r"\b(xuất sắc|giỏi|honors?|high distinction|magna cum laude|summa cum laude|"
    r"thạc sĩ|tiến sĩ|master|phd|msc|mba|gpa\s*[:\s]*3\.[2-9]|gpa\s*[:\s]*[89]\.)\b",
    re.IGNORECASE,
)

_BIGTECH_COMPANIES = re.compile(
    r"\b(viettel|vingroup|vinai|vinbigdata|vin|fpt|vng|momo|shopee|vnpay|grab|topcv|synopsys|marvell|"
    r"samsung|techcombank|vpbank|mbbank|accenture|deloitte|kpmg|ey|pwc|microsoft|"
    r"google|amazon|meta|apple|netflix|tiktok|bytedance|paypal|oracle|sea group|atrae)\b",
    re.IGNORECASE,
)

_MIDTECH_COMPANIES = re.compile(
    r"\b(kms|tma|nashtech|axon|sun\*|orient|elca|paradox|dek|nfq|fossil|"
    r"rikkeisoft|smartosc|sutrix|hybrid technologies)\b",
    re.IGNORECASE,
)

_ADVANCED_SKILLS = re.compile(
    r"\b(microservices?|kubernetes|kafka|system design|distributed|sharding|"
    r"high availability|architecture|micro.?frontends?|web performance|ssr|ssg|"
    r"clean architecture|viper|module federation|automation framework|playwright|"
    r"appium|load testing|k6|jmeter|data lakehouse|spark|flink|llmops|fine.?tuning|"
    r"rag|distributed training|deep learning|nlp|recommendation systems?|llm|langchain)\b",
    re.IGNORECASE,
)

_INTL_CERTS_EXPERT = re.compile(
    r"\b(aws\s*(?:solutions?\s*architect|saa|sap|devops)|gcp\s*pro|cka|cks|pmp|cissp|"
    r"databricks\s*pro|ielts\s*[789]|toeic\s*(?:8[5-9]\d|9\d{2})|ef\s*set|c1\s*advanced|"
    r"mlops\s*specialization|machine\s*learning\s*specialization)\b",
    re.IGNORECASE,
)

_INTL_CERTS_ASSOCIATE = re.compile(
    r"\b(azure\s*(associate|data\s*engineer)|databricks\s*associate|terraform\s*associate|"
    r"ccna|istqb|ielts\s*[67]|toeic\s*(6[5-9]\d|[78]\d{2})|specialization|professional certificate)\b",
    re.IGNORECASE,
)

_INTL_CERTS_BASIC = re.compile(
    r"\b(cloud\s*practitioner|az.?900|coursera|udemy|toeic\s*[56]\d{2})\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Tier aliases & normalizers (maps LLM output variations to canonical tiers)
# ---------------------------------------------------------------------------
_EDU_ALIASES: dict[str, str] = {
    "TIER_1A_ELITE": "TIER_1A_ELITE", "TIER_1A": "TIER_1A_ELITE", "ELITE": "TIER_1A_ELITE", "TOP": "TIER_1A_ELITE", "MASTER": "TIER_1A_ELITE", "PHD": "TIER_1A_ELITE",
    "TIER_1B_ACCREDITED_TECH": "TIER_1B_ACCREDITED_TECH", "TIER_1B": "TIER_1B_ACCREDITED_TECH", "ACCREDITED_TECH": "TIER_1B_ACCREDITED_TECH",
    "STANDARD_ACCREDITED": "STANDARD_ACCREDITED", "STANDARD": "STANDARD_ACCREDITED", "BACHELOR": "STANDARD_ACCREDITED", "BACHELORS": "STANDARD_ACCREDITED",
    "ASSOCIATE_OTHER": "ASSOCIATE_OTHER", "ASSOCIATE": "ASSOCIATE_OTHER", "COLLEGE": "ASSOCIATE_OTHER",
    "NON_DEGREE": "NON_DEGREE", "SELF_TAUGHT": "NON_DEGREE", "NONE": "NON_DEGREE",
}

_CORP_ALIASES: dict[str, str] = {
    "TIER_1_BIGTECH_ENTERPRISE": "TIER_1_BIGTECH_ENTERPRISE", "TIER_1": "TIER_1_BIGTECH_ENTERPRISE", "BIGTECH": "TIER_1_BIGTECH_ENTERPRISE", "GLOBAL_ENTERPRISE": "TIER_1_BIGTECH_ENTERPRISE", "ENTERPRISE": "TIER_1_BIGTECH_ENTERPRISE", "UNICORN": "TIER_1_BIGTECH_ENTERPRISE", "GLOBAL": "TIER_1_BIGTECH_ENTERPRISE",
    "TIER_2_MID_TECH": "TIER_2_MID_TECH", "TIER_2": "TIER_2_MID_TECH", "MID_TECH": "TIER_2_MID_TECH", "MID": "TIER_2_MID_TECH", "PRODUCT": "TIER_2_MID_TECH",
    "STANDARD_SME": "STANDARD_SME", "SME": "STANDARD_SME", "STARTUP": "STANDARD_SME", "SMALL": "STANDARD_SME",
}

_SKILL_ALIASES: dict[str, str] = {
    "ADVANCED_EVIDENCE_BASED": "ADVANCED_EVIDENCE_BASED", "ADVANCED": "ADVANCED_EVIDENCE_BASED", "HIGH": "ADVANCED_EVIDENCE_BASED", "EXPERT": "ADVANCED_EVIDENCE_BASED",
    "COMPETENT_PRODUCTION": "COMPETENT_PRODUCTION", "COMPETENT": "COMPETENT_PRODUCTION", "PRODUCTION": "COMPETENT_PRODUCTION", "MEDIUM": "COMPETENT_PRODUCTION", "MID": "COMPETENT_PRODUCTION",
    "BASIC_KEYWORD_ONLY": "BASIC_KEYWORD_ONLY", "BASIC": "BASIC_KEYWORD_ONLY", "LOW": "BASIC_KEYWORD_ONLY", "KEYWORD": "BASIC_KEYWORD_ONLY",
}

_PROJ_ALIASES: dict[str, str] = {
    "HIGH_IMPACT_METRICS": "HIGH_IMPACT_METRICS", "HIGH_IMPACT": "HIGH_IMPACT_METRICS", "HIGH": "HIGH_IMPACT_METRICS", "PROFESSIONAL": "HIGH_IMPACT_METRICS", "METRICS": "HIGH_IMPACT_METRICS",
    "STANDARD_COMPLETED": "STANDARD_COMPLETED", "STANDARD": "STANDARD_COMPLETED", "COMPLETED": "STANDARD_COMPLETED",
    "ACADEMIC_ONLY": "ACADEMIC_ONLY", "ACADEMIC": "ACADEMIC_ONLY",
}

_CERT_ALIASES: dict[str, str] = {
    "EXPERT_PRO": "EXPERT_PRO", "EXPERT": "EXPERT_PRO", "PRO": "EXPERT_PRO", "PROFESSIONAL": "EXPERT_PRO", "HIGH": "EXPERT_PRO",
    "ASSOCIATE_PRACTITIONER": "ASSOCIATE_PRACTITIONER", "ASSOCIATE": "ASSOCIATE_PRACTITIONER", "PRACTITIONER": "ASSOCIATE_PRACTITIONER", "MID": "ASSOCIATE_PRACTITIONER",
    "BASIC_FOUNDATIONAL": "BASIC_FOUNDATIONAL", "FOUNDATIONAL": "BASIC_FOUNDATIONAL", "BASIC": "BASIC_FOUNDATIONAL", "COURSE": "BASIC_FOUNDATIONAL",
    "NONE": "NONE",
}

_LANG_ALIASES: dict[str, str] = {
    "EXPERT_FLUENT": "EXPERT_FLUENT", "FLUENT": "EXPERT_FLUENT", "EXPERT": "EXPERT_FLUENT", "NATIVE": "EXPERT_FLUENT", "PROFESSIONAL": "EXPERT_FLUENT", "C1": "EXPERT_FLUENT", "C2": "EXPERT_FLUENT",
    "WORKING_PROFICIENCY": "WORKING_PROFICIENCY", "WORKING": "WORKING_PROFICIENCY", "INTERMEDIATE": "WORKING_PROFICIENCY", "B2": "WORKING_PROFICIENCY",
    "BASIC_ELEMENTARY": "BASIC_ELEMENTARY", "BASIC": "BASIC_ELEMENTARY", "ELEMENTARY": "BASIC_ELEMENTARY", "A1": "BASIC_ELEMENTARY", "A2": "BASIC_ELEMENTARY",
    "NONE": "NONE",
}


def _resolve_tier(val: str | None, mapping: dict[str, str]) -> str | None:
    if not val:
        return None
    val_clean = val.strip().upper()
    if val_clean in mapping:
        return mapping[val_clean]
    for key, canonical in mapping.items():
        if key in val_clean:
            return canonical
    return None


# Per-criterion scoring functions
# ---------------------------------------------------------------------------

def _score_education(profile: CVProfile, tiers: EvaluatedTiers | None) -> tuple[float, str, str]:
    """Score education tier on 100-point scale."""
    tier = _resolve_tier(tiers.education_tier, _EDU_ALIASES) if tiers else None
    if tier and tier in _EDUCATION_SCORE:
        score = _EDUCATION_SCORE[tier]
        inst = profile.education[0].institution if profile.education else "N/A"
        degree = (profile.education[0].degree or "") if profile.education else ""
        reason_parts = [f"Tốt nghiệp {inst}"]
        if degree:
            reason_parts.append(f"({degree})")
        return score, " ".join(reason_parts) + ".", "llm_tier"

    # Heuristic fallback
    if not profile.education:
        return 40.0, "Không có thông tin học vấn chính quy.", "heuristic_fallback"

    edu = profile.education[0]
    inst = edu.institution or ""
    degree = (edu.degree or "").lower()

    is_honors = bool(_HONORS_DEGREE.search(degree) or _HONORS_DEGREE.search(inst))
    is_top = bool(_TOP_UNIVERSITIES.search(inst))
    is_tier_1b = bool(_TIER_1B_UNIVERSITIES.search(inst))

    if is_top and is_honors:
        return 100.0, f"Tốt nghiệp {inst} (Bằng Giỏi/Thạc sĩ+).", "heuristic_fallback"
    if is_top:
        return 90.0, f"Tốt nghiệp {inst} - trường kỹ thuật uy tín.", "heuristic_fallback"
    if is_tier_1b:
        return 90.0, f"Tốt nghiệp {inst} - trường đào tạo CNTT chất lượng.", "heuristic_fallback"
    if inst:
        return 75.0, f"Tốt nghiệp đại học chính quy tại {inst}.", "heuristic_fallback"
    return 60.0, "Học vấn đào tạo khác.", "heuristic_fallback"


def _score_experience_years(profile: CVProfile) -> tuple[float, str, str]:
    """Score experience years on 100-point scale."""
    yoe = profile.total_experience_years or 0.0

    if yoe >= 5.0:
        return 100.0, f"{yoe:.1f} năm kinh nghiệm thực tế cấp bậc Senior/Lead.", "computed_timeline"
    if yoe >= 3.0:
        return 85.0, f"{yoe:.1f} năm kinh nghiệm thực tế cấp bậc Mid-level.", "computed_timeline"
    if yoe >= 1.5:
        return 70.0, f"{yoe:.1f} năm kinh nghiệm thực tế cấp bậc Junior.", "computed_timeline"
    if yoe >= 0.5:
        return 55.0, f"{yoe:.1f} năm kinh nghiệm (Fresher/Junior mới).", "computed_timeline"
    if yoe > 0 or len(profile.experiences) > 0:
        return 35.0, f"Dưới 0.5 năm kinh nghiệm / Thực tập sinh.", "computed_timeline"
    return 35.0, "Chưa có thông tin kinh nghiệm làm việc.", "computed_timeline"


def _score_enterprise_scale(profile: CVProfile, tiers: EvaluatedTiers | None) -> tuple[float, str, str]:
    """Score company prestige on 100-point scale."""
    tier = _resolve_tier(tiers.company_prestige_tier, _CORP_ALIASES) if tiers else None
    if tier and tier in _ENTERPRISE_SCORE:
        score = _ENTERPRISE_SCORE[tier]
        companies = [exp.company for exp in profile.experiences if exp.company]
        top = companies[0] if companies else "doanh nghiệp lớn"
        return score, f"Từng làm việc tại {top}.", "llm_tier"

    # Heuristic fallback
    companies_text = " ".join(exp.company or "" for exp in profile.experiences)
    if _BIGTECH_COMPANIES.search(companies_text):
        matched = _BIGTECH_COMPANIES.search(companies_text)
        return 100.0, f"Từng làm việc tại tập đoàn lớn: {matched.group(0)}.", "heuristic_fallback"
    if _MIDTECH_COMPANIES.search(companies_text):
        return 80.0, "Từng làm tại công ty Product/Outsourcing uy tín.", "heuristic_fallback"
    return 60.0, "Công ty quy mô vừa và nhỏ hoặc startup.", "heuristic_fallback"


def _score_technical_skills(profile: CVProfile, tiers: EvaluatedTiers | None) -> tuple[float, str, str]:
    """Score technical skills on 100-point scale with evidence weighting."""
    tier = _resolve_tier(tiers.skill_evidence_level, _SKILL_ALIASES) if tiers else None
    if tier and tier in _SKILL_EVIDENCE_SCORE:
        score = _SKILL_EVIDENCE_SCORE[tier]
        skill_names = [s.canonical_name for s in profile.skills[:6]]
        return score, f"Kỹ năng nổi bật: {', '.join(skill_names)}." if skill_names else f"Trình độ kỹ năng: {tier}.", "llm_tier"

    # Heuristic fallback with 75/25 evidence weighting
    skill_names = [s.canonical_name for s in profile.skills]
    if not skill_names:
        return 65.0, "Không có thông tin kỹ năng chi tiết.", "heuristic_fallback"

    # Skills with evidence from experience/project sections
    evidenced_skills = [s for s in profile.skills if s.evidence]
    keyword_only_skills = [s for s in profile.skills if not s.evidence]

    has_advanced = any(
        _ADVANCED_SKILLS.search(s.canonical_name) for s in profile.skills
    )
    # Also check experience/project descriptions for advanced patterns
    all_desc = " ".join(
        " ".join(exp.description + exp.achievements) for exp in profile.experiences
    ) + " " + " ".join(
        " ".join(proj.description + proj.achievements) for proj in profile.projects
    )
    has_advanced = has_advanced or bool(_ADVANCED_SKILLS.search(all_desc))

    evidenced_ratio = len(evidenced_skills) / max(len(profile.skills), 1)
    # Weighted: 75% evidence-based + 25% keyword coverage
    weighted_evidence = evidenced_ratio * 0.75

    if has_advanced and weighted_evidence >= 0.3:
        return 95.0, f"Kỹ năng chuyên sâu có bằng chứng thực chiến ({len(evidenced_skills)} kỹ năng có evidence).", "heuristic_fallback"
    if len(skill_names) >= 4:
        return 85.0, f"Bộ kỹ năng phong phú ({len(skill_names)} kỹ năng).", "heuristic_fallback"
    return 65.0, f"Kỹ năng cơ bản ({len(skill_names)} kỹ năng).", "heuristic_fallback"


def _score_projects_portfolio(profile: CVProfile, tiers: EvaluatedTiers | None) -> tuple[float, str, str]:
    """Score projects & portfolio on 100-point scale."""
    tier = _resolve_tier(tiers.project_quality_tier, _PROJ_ALIASES) if tiers else None
    if tier and tier in _PROJECT_QUALITY_SCORE:
        score = _PROJECT_QUALITY_SCORE[tier]
        proj_count = len(profile.projects)
        return score, f"Có {proj_count} dự án thực tế đạt tiêu chuẩn cao ({tier})." if proj_count else "Dự án và thành tựu được đánh giá tích cực.", "llm_tier"

    # Heuristic fallback
    projects = profile.projects
    if not projects:
        # Check experiences for project-like achievements
        all_achievements = []
        for exp in profile.experiences:
            all_achievements.extend(exp.achievements)
        if all_achievements:
            has_metrics = any(
                re.search(r"\d+%|\d+k|\d+M|\d+\.\d+", a) for a in all_achievements
            )
            if has_metrics and len(all_achievements) >= 3:
                return 95.0, f"Có {len(all_achievements)} thành tựu định lượng trong kinh nghiệm làm việc.", "heuristic_fallback"
            if all_achievements:
                return 80.0, f"Có mô tả thành tựu trong kinh nghiệm làm việc.", "heuristic_fallback"
        return 50.0, "Không có thông tin dự án cụ thể.", "heuristic_fallback"

    has_metrics = False
    for proj in projects:
        for line in proj.achievements + proj.description:
            if re.search(r"\d+%|\d+k|\d+M|\d+\.\d+", line):
                has_metrics = True
                break

    if len(projects) >= 3 and has_metrics:
        return 95.0, f"Có {len(projects)} dự án thực tế kèm số liệu định lượng.", "heuristic_fallback"
    if projects:
        return 80.0, f"Có {len(projects)} dự án hoàn chỉnh.", "heuristic_fallback"
    return 50.0, "Chỉ có đồ án môn học hoặc bài tập lớn.", "heuristic_fallback"


def _score_certifications_language(profile: CVProfile, tiers: EvaluatedTiers | None) -> tuple[float, str, str]:
    """Score certifications & language proficiency on 100-point scale."""
    # Prefer the higher of certification_tier and language_proficiency
    cert_score: float | None = None
    lang_score: float | None = None

    if tiers:
        cert_tier = _resolve_tier(tiers.certification_tier, _CERT_ALIASES)
        if cert_tier and cert_tier in _CERT_SCORE:
            cert_score = _CERT_SCORE[cert_tier]
        lang_tier = _resolve_tier(tiers.language_proficiency, _LANG_ALIASES)
        if lang_tier and lang_tier in _LANG_SCORE:
            lang_score = _LANG_SCORE[lang_tier]

    if cert_score is not None or lang_score is not None:
        final_score = max(cert_score if cert_score is not None else 0, lang_score if lang_score is not None else 0)
        parts = []
        if profile.certifications:
            parts.append(f"Chứng chỉ: {', '.join(profile.certifications[:3])}")
        if profile.language_proficiencies:
            parts.append(f"Ngoại ngữ: {', '.join(lp.language for lp in profile.language_proficiencies[:2])}")
        elif profile.languages:
            parts.append(f"Ngoại ngữ: {', '.join(profile.languages[:2])}")
        reason = "; ".join(parts) + "." if parts else "Đạt tiêu chuẩn chứng chỉ & ngoại ngữ chuyên nghiệp."
        return final_score, reason, "llm_tier"

    # Heuristic fallback
    cert_text = " ".join(profile.certifications)
    lang_text = " ".join(profile.languages)
    lang_prof_text = " ".join(
        f"{lp.language} {lp.exam or ''} {lp.score or ''}" for lp in profile.language_proficiencies
    )
    # Also check skill evidences and experience/project achievements for embedded certs/TOEIC/IELTS
    evidence_text = " ".join(
        " ".join(s.evidence) for s in profile.skills
    ) + " " + " ".join(
        " ".join(exp.achievements + exp.description) for exp in profile.experiences
    ) + " " + " ".join(
        " ".join(edu.coursework) for edu in profile.education
    )

    combined = f"{cert_text} {lang_text} {lang_prof_text} {evidence_text}"

    if _INTL_CERTS_EXPERT.search(combined):
        matched = _INTL_CERTS_EXPERT.search(combined).group(0)[:50]
        return 95.0, f"Có chứng chỉ quốc tế cấp cao hoặc ngoại ngữ xuất sắc: {matched}.", "heuristic_fallback"
    if _INTL_CERTS_ASSOCIATE.search(combined):
        matched = _INTL_CERTS_ASSOCIATE.search(combined).group(0)[:50]
        return 85.0, f"Có chứng chỉ chuyên môn / Associate: {matched}.", "heuristic_fallback"
    if _INTL_CERTS_BASIC.search(combined):
        return 50.0, "Có chứng chỉ cơ bản hoặc khóa học online.", "heuristic_fallback"
    if profile.certifications or profile.language_proficiencies or profile.languages:
        return 50.0, "Có thông tin chứng chỉ/ngoại ngữ cơ bản.", "heuristic_fallback"
    return 20.0, "Không có chứng chỉ hay thông tin ngoại ngữ.", "heuristic_fallback"


# ---------------------------------------------------------------------------
# V3: Industry-standard baseline weights (70% practical / 30% credentials)
# ---------------------------------------------------------------------------
_V3_WEIGHTS: dict[str, float] = {
    "TECHNICAL_DEPTH": 0.30,
    "IMPACT_METRICS": 0.25,
    "ENTERPRISE_SCALE": 0.15,
    "EDUCATION": 0.15,
    "CERTIFICATIONS": 0.10,
    "LANGUAGE_PROFICIENCY": 0.05,
}

# V3 Criterion display names (Vietnamese, for HR tooltip)
_V3_CRITERIA_NAMES: dict[str, str] = {
    "TECHNICAL_DEPTH": "Kỹ năng & Kiến trúc",
    "IMPACT_METRICS": "Dự án & Thành tựu",
    "ENTERPRISE_SCALE": "Quy mô Doanh nghiệp",
    "EDUCATION": "Học vấn & CS Foundation",
    "CERTIFICATIONS": "Chứng chỉ Chuyên môn",
    "LANGUAGE_PROFICIENCY": "Năng lực Ngoại ngữ",
}

# V3 Ceiling Gate tier limits: maps (criterion_key, tier) -> max_allowed_score
# If AI assigns a tier, the score CANNOT exceed that tier's upper boundary.
_TIER_CEILING_GATES: dict[str, dict[str, float]] = {
    "TECHNICAL_DEPTH": {
        "BASIC_KEYWORD_ONLY": 49.0,
        "COMPETENT_BASIC": 71.0,
        "COMPETENT_PRODUCTION": 83.0,  # No architecture design -> cap at 83
        "SENIOR_ARCHITECT": 92.0,
        "PRINCIPAL_ARCHITECT": 100.0,
    },
    "IMPACT_METRICS": {
        "ACADEMIC_ONLY": 44.0,
        "STANDARD_COMPLETED_SPARSE": 64.0,
        "STANDARD_COMPLETED_DESCRIBED": 79.0,  # No quantified metrics -> cap at 79
        "HIGH_IMPACT_METRICS": 92.0,
        "HIGH_IMPACT_ELITE": 100.0,
    },
    "ENTERPRISE_SCALE": {
        "STANDARD_SME": 64.0,  # Small startup / freelance -> cap at 64
        "TIER_2_MID_TECH": 79.0,
        "TIER_1_BIGTECH_ENTERPRISE": 92.0,
        "TIER_1_BIGTECH_ELITE": 100.0,
    },
    "EDUCATION": {
        "NON_DEGREE": 54.0,  # No university degree -> cap at 54
        "STANDARD_ACCREDITED": 71.0,
        "TIER_1B_ACCREDITED": 82.0,
        "TIER_1B_ACCREDITED_TECH": 82.0,
        "TIER_1A_ELITE": 92.0,
        "TIER_1A_ELITE_PLUS": 100.0,
    },
    "CERTIFICATIONS": {
        "NONE": 39.0,  # No certifications -> cap at 39
        "FOUNDATIONAL_ONLINE": 74.0,
        "BASIC_FOUNDATIONAL": 74.0,
        "ASSOCIATE_PRACTITIONER": 89.0,
        "EXPERT_PRO": 100.0,
    },
    "LANGUAGE_PROFICIENCY": {
        "NONE": 49.0,
        "NONE_OR_MINIMAL": 49.0,
        "BASIC_ELEMENTARY": 49.0,
        "BASIC_READING": 74.0,
        "WORKING_PROFICIENCY": 89.0,
        "EXPERT_FLUENT": 100.0,
    },
}


def _validate_ceiling_gate(key: str, score: float, tier: str | None = None) -> float:
    """Clamp score to tier's ceiling gate maximum if it exceeds the boundary."""
    if not tier:
        return score
    tier_upper = tier.strip().upper()
    gates = _TIER_CEILING_GATES.get(key, {})
    # Check exact match or normalized alias
    max_allowed = gates.get(tier_upper)
    if max_allowed is not None and score > max_allowed:
        return max_allowed
    return score


# V3 Seniority classification based on overall score
_SENIORITY_BANDS: list[tuple[float, str, str]] = [
    (93.0, "ELITE_ARCHITECT_PRINCIPAL", "EXCEPTIONAL"),
    (85.0, "ELITE_SENIOR_LEAD", "STRONG_RECOMMEND"),
    (75.0, "COMPETENT_SENIOR", "RECOMMEND"),
    (62.0, "MID_LEVEL_SOLID", "CONSIDER"),
    (48.0, "JUNIOR_STRONG", "CONSIDER_JUNIOR_ROLE"),
    (0.0, "FRESHER_JUNIOR", "PIPELINE_ONLY"),
]


def _classify_seniority(overall_score: float) -> tuple[str, str]:
    """Map overall weighted score to seniority level and hiring signal."""
    for threshold, level, signal in _SENIORITY_BANDS:
        if overall_score >= threshold:
            return level, signal
    return "FRESHER_JUNIOR", "PIPELINE_ONLY"


def _v3_from_rubric_scores(profile: CVProfile) -> dict[str, Any] | None:
    """Try to build V3 scoring result from AI-returned rubric_scores.

    Returns None if rubric_scores is unavailable, triggering V2 fallback.
    """
    rs = profile.rubric_scores
    if not rs or not isinstance(rs, dict):
        return None

    # Map from rubricScores keys (camelCase from AI) to canonical criterion keys
    _KEY_MAP: dict[str, str] = {
        "technicalDepth": "TECHNICAL_DEPTH",
        "technical_depth": "TECHNICAL_DEPTH",
        "TECHNICAL_DEPTH": "TECHNICAL_DEPTH",
        "impactMetrics": "IMPACT_METRICS",
        "impact_metrics": "IMPACT_METRICS",
        "IMPACT_METRICS": "IMPACT_METRICS",
        "enterpriseScale": "ENTERPRISE_SCALE",
        "enterprise_scale": "ENTERPRISE_SCALE",
        "ENTERPRISE_SCALE": "ENTERPRISE_SCALE",
        "education": "EDUCATION",
        "EDUCATION": "EDUCATION",
        "certifications": "CERTIFICATIONS",
        "CERTIFICATIONS": "CERTIFICATIONS",
        "languageProficiency": "LANGUAGE_PROFICIENCY",
        "language_proficiency": "LANGUAGE_PROFICIENCY",
        "LANGUAGE_PROFICIENCY": "LANGUAGE_PROFICIENCY",
    }

    criteria_scores: dict[str, dict[str, Any]] = {}
    found_any = False

    for raw_key, criterion_data in rs.items():
        canonical_key = _KEY_MAP.get(raw_key)
        if not canonical_key:
            continue

        if isinstance(criterion_data, CriterionScore):
            score_val = criterion_data.score
            tier_val = criterion_data.tier
            explanation_val = criterion_data.explanation
            evidence_val = criterion_data.evidence_summary
            name_val = criterion_data.name
        elif isinstance(criterion_data, dict):
            score_val = float(criterion_data.get("score", 0.0))
            tier_val = criterion_data.get("tier")
            explanation_val = criterion_data.get("explanation")
            evidence_val = criterion_data.get("evidenceSummary", [])
            name_val = criterion_data.get("name")
        else:
            continue

        # Clamp to 0-100 and apply ceiling gate validation
        score_val = max(0.0, min(100.0, score_val))
        score_val = _validate_ceiling_gate(canonical_key, score_val, tier_val)

        criteria_scores[canonical_key] = {
            "name": name_val or _V3_CRITERIA_NAMES.get(canonical_key, canonical_key),
            "score": round(score_val, 1),
            "tier": tier_val,
            "explanation": explanation_val or "",
            "evidenceSummary": evidence_val if isinstance(evidence_val, list) else [],
            "evaluationMethod": "llm_calibrated_rubric_v3",
        }
        found_any = True

    if not found_any:
        return None

    # Fill missing criteria with defaults
    for key, name in _V3_CRITERIA_NAMES.items():
        if key not in criteria_scores:
            criteria_scores[key] = {
                "name": name,
                "score": 0.0,
                "tier": None,
                "explanation": "Không có thông tin từ AI.",
                "evidenceSummary": [],
                "evaluationMethod": "default_missing",
            }

    # Compute weighted overall score
    weighted_sum = sum(
        criteria_scores[k]["score"] * _V3_WEIGHTS.get(k, 0.0)
        for k in criteria_scores
    )
    overall_score = round(weighted_sum, 1)
    seniority_level, hiring_signal = _classify_seniority(overall_score)

    return {
        "criteriaScores": criteria_scores,
        "Score": overall_score,
        "maxScore": 100.0,
        "summary": profile.executive_summary or "",
        "scoringRationale": f"Điểm tổng chuẩn ngành: {overall_score}đ. Cấp bậc: {seniority_level}.",
        "seniorityCalibratedLevel": seniority_level,
        "hiringSignal": hiring_signal,
        "scoringVersion": "v3_rubric",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_it_rubric_scoring(profile: CVProfile) -> dict[str, Any]:
    """Compute 6-criteria IT rubric scoring on 100-point scale per criterion.

    V3 path: If profile.rubric_scores is available (AI-calibrated), use those
    scores directly with Ceiling Gate validation guardrail.

    V2 fallback: If rubric_scores is unavailable, fall back to tier-based
    mapping from evaluated_tiers with heuristic fallback.

    Returns a dict with keys:
        - ``criteriaScores``: mapping of criterion key → {name, score, tier, explanation, ...}
        - ``maxScore``: 100.0
        - ``summary``: executive summary (from LLM or empty)
        - ``Score``: weighted overall score (V3) or max score (V2)
        - ``scoringVersion``: 'v3_rubric' or 'v2_tier_mapping'
    """
    # Try V3 first (AI-calibrated rubric scores)
    v3_result = _v3_from_rubric_scores(profile)
    if v3_result is not None:
        return v3_result

    # V2 fallback: tier-based mapping
    tiers = profile.evaluated_tiers

    edu_score, edu_reason, edu_method = _score_education(profile, tiers)
    exp_score, exp_reason, exp_method = _score_experience_years(profile)
    ent_score, ent_reason, ent_method = _score_enterprise_scale(profile, tiers)
    skl_score, skl_reason, skl_method = _score_technical_skills(profile, tiers)
    prj_score, prj_reason, prj_method = _score_projects_portfolio(profile, tiers)
    crt_score, crt_reason, crt_method = _score_certifications_language(profile, tiers)

    criteria_scores: dict[str, dict[str, Any]] = {
        "EDUCATION": {
            "name": CRITERIA_NAMES["EDUCATION"],
            "score": edu_score,
            "reason": edu_reason,
            "evaluationMethod": edu_method,
        },
        "EXPERIENCE_YEARS": {
            "name": CRITERIA_NAMES["EXPERIENCE_YEARS"],
            "score": exp_score,
            "reason": exp_reason,
            "evaluationMethod": exp_method,
        },
        "ENTERPRISE_SCALE": {
            "name": CRITERIA_NAMES["ENTERPRISE_SCALE"],
            "score": ent_score,
            "reason": ent_reason,
            "evaluationMethod": ent_method,
        },
        "TECHNICAL_SKILLS": {
            "name": CRITERIA_NAMES["TECHNICAL_SKILLS"],
            "score": skl_score,
            "reason": skl_reason,
            "evaluationMethod": skl_method,
        },
        "PROJECTS_PORTFOLIO": {
            "name": CRITERIA_NAMES["PROJECTS_PORTFOLIO"],
            "score": prj_score,
            "reason": prj_reason,
            "evaluationMethod": prj_method,
        },
        "CERTIFICATIONS_LANGUAGE": {
            "name": CRITERIA_NAMES["CERTIFICATIONS_LANGUAGE"],
            "score": crt_score,
            "reason": crt_reason,
            "evaluationMethod": crt_method,
        },
    }

    all_scores = [v["score"] for v in criteria_scores.values()]
    max_score = max(all_scores) if all_scores else 0.0

    return {
        "criteriaScores": criteria_scores,
        "maxScore": max_score,
        "summary": profile.executive_summary or "",
        "scoringVersion": "v2_tier_mapping",
    }


# ---------------------------------------------------------------------------
# Legacy adapter (keeps existing compute_rubric_scoring contract working)
# ---------------------------------------------------------------------------

def compute_rubric_scoring(result: ProcessingResult) -> tuple[float, list[dict[str, Any]]]:
    """Backward-compatible 4-criteria rubric score matching Codebase B's UI output contract.

    This function preserves the original contract: returns (total_score, scoring_details_list).
    Internally it delegates to the V2 6-criteria engine and maps back.
    """
    from ai_core.api.scoring_rubric import (
        _score_certifications as _legacy_score_certifications,
        _score_education as _legacy_score_education,
        _score_experience as _legacy_score_experience,
        _score_skills as _legacy_score_skills,
        MAX_SCORE,
    )

    profile = result.profile
    if profile is None:
        return 0.0, [
            {"criterionName": k, "score": 0.0, "maxScore": MAX_SCORE[k], "reason": "Không có thông tin profile."}
            for k in ["Học vấn", "Kinh nghiệm", "Kỹ năng", "Chứng chỉ"]
        ]

    edu_score, edu_reason = _legacy_score_education(profile)
    exp_score, exp_reason = _legacy_score_experience(profile)
    skl_score, skl_reason = _legacy_score_skills(profile)
    crt_score, crt_reason = _legacy_score_certifications(profile)

    scoring_details = [
        {"criterionName": "Học vấn", "score": edu_score, "maxScore": MAX_SCORE["Học vấn"], "reason": edu_reason},
        {"criterionName": "Kinh nghiệm", "score": exp_score, "maxScore": MAX_SCORE["Kinh nghiệm"], "reason": exp_reason},
        {"criterionName": "Kỹ năng", "score": skl_score, "maxScore": MAX_SCORE["Kỹ năng"], "reason": skl_reason},
        {"criterionName": "Chứng chỉ", "score": crt_score, "maxScore": MAX_SCORE["Chứng chỉ"], "reason": crt_reason},
    ]

    total_score = round(edu_score + exp_score + skl_score + crt_score, 1)
    return min(100.0, max(0.0, total_score)), scoring_details
