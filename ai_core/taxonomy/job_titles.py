"""Job titles taxonomy and entity extractor based on Codebase B's dictionary."""

from __future__ import annotations

import re
from ai_core.schemas import CVProfile

JOB_TITLES: dict[str, list[str]] = {
    "Backend Developer": ["backend developer", "backend engineer", "lập trình viên backend", "phát triển backend", "be dev", "backend dev"],
    "Frontend Developer": ["frontend developer", "frontend engineer", "lập trình viên frontend", "phát triển frontend", "fe dev", "frontend dev"],
    "Fullstack Developer": ["fullstack developer", "fullstack engineer", "lập trình viên fullstack", "full stack", "fullstack dev"],
    "Mobile Developer": ["mobile developer", "mobile engineer", "lập trình viên di động", "android developer", "ios developer", "flutter developer", "react native developer"],
    "DevOps Engineer": ["devops engineer", "devops", "sre", "site reliability engineer", "kỹ sư devops"],
    "Data Engineer": ["data engineer", "kỹ sư dữ liệu"],
    "Data Scientist": ["data scientist", "nhà khoa học dữ liệu"],
    "Data Analyst": ["data analyst", "chuyên viên phân tích dữ liệu"],
    "AI/ML Engineer": ["ai engineer", "ml engineer", "machine learning engineer", "kỹ sư ai", "kỹ sư trí tuệ nhân tạo"],
    "QA/QC Engineer": ["qa engineer", "qc engineer", "tester", "kiểm thử phần mềm", "automation tester", "manual tester"],
    "System Administrator": ["system administrator", "sysadmin", "quản trị hệ thống"],
    "Security Engineer": ["security engineer", "cybersecurity", "chuyên viên an toàn thông tin"],
    "UI/UX Designer": ["ui/ux designer", "ui designer", "ux designer", "thiết kế giao diện"],
    "Product Manager": ["product manager", "quản lý sản phẩm"],
    "Project Manager": ["project manager", "quản lý dự án"],
    "Scrum Master": ["scrum master"],
    "Business Analyst": ["business analyst", "phân tích nghiệp vụ"],
    "Embedded Engineer": ["embedded engineer", "kỹ sư nhúng"],
    "Cloud Engineer": ["cloud engineer", "cloud architect"],
    "Solution Architect": ["solution architect", "kiến trúc sư giải pháp"],
    "Software Engineer": ["software engineer", "lập trình viên", "kỹ sư phần mềm", "developer"],
}


def extract_job_titles(profile: CVProfile) -> list[str]:
    """Derive canonical job titles from candidate profile using taxonomy matching."""
    matched_titles: list[str] = []
    seen: set[str] = set()

    sources: list[str] = []
    if profile.headline:
        sources.append(profile.headline)

    for exp in profile.experiences:
        if exp.job_title:
            sources.append(exp.job_title)

    for project in profile.projects:
        if project.role:
            sources.append(project.role)

    combined_text = " ".join(sources).lower()

    for canonical, aliases in JOB_TITLES.items():
        for alias in aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, combined_text, re.IGNORECASE):
                if canonical not in seen:
                    matched_titles.append(canonical)
                    seen.add(canonical)
                break

    # Fallback to raw titles if no taxonomy match was found
    if not matched_titles:
        for src in sources:
            clean = src.strip()
            if clean and clean.lower() not in seen:
                matched_titles.append(clean)
                seen.add(clean.lower())

    return matched_titles
