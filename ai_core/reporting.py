"""Markdown reports and deterministic LLM cost calculations."""

from __future__ import annotations

from pathlib import Path

from ai_core.schemas import ProcessingResult

INPUT_TOKEN_USD = 0.00000005
OUTPUT_TOKEN_USD = 0.00000020
VND_PER_USD = 25_400


def write_process_report(result: ProcessingResult, path: Path) -> None:
    profile = result.profile
    audit = result.audit
    tokens = audit.get("tokens", {})
    input_tokens = int(tokens.get("input", 0))
    output_tokens = int(tokens.get("output", 0))
    evidence = sum(len(values) for values in (profile.field_evidence if profile else {}).values())
    evidence += sum(len(item.evidence) for item in (profile.experiences if profile else []))
    evidence += sum(len(item.evidence) for item in (profile.projects if profile else []))
    evidence += sum(len(item.evidence) for item in (profile.education if profile else []))
    scores = {score.name: score.score for score in result.scores}
    lines = [
        "# Smart CV AI Core Processing Report",
        "",
        "## File information",
        "",
        "| File | SHA-256 | Pages | Blocks |",
        "|---|---|---:|---:|",
        f"| {result.metadata.source_name if result.metadata else '-'} | {result.source_id} | "
        f"{result.metadata.page_count if result.metadata else 0} | {audit.get('blockCount', 0)} |",
        "",
        "## Stage latencies (ms)",
        "",
        "| Validation | Parsing | Unified | LLM extraction | Reconciliation | "
        "Scoring | Total E2E |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            str(result.timings_ms.get(key, 0))
            for key in (
                "inputValidation",
                "parsingAndPreprocessing",
                "unifiedExtraction",
                "profileExtraction",
                "reconciliation",
                "scoring",
                "total",
            )
        )
        + " |",
        "",
        "## Token statistics",
        "",
        "| Input tokens | Output tokens |",
        "|---:|---:|",
        f"| {input_tokens} | {output_tokens} |",
        "",
        "| Prompt document tokens before pruning | After pruning | Reduction |",
        "|---:|---:|---:|",
        f"| {tokens.get('documentBefore', 0)} | {tokens.get('documentAfter', 0)} | "
        f"{_reduction(tokens)}% |",
        "",
        "## Quality",
        "",
        "| Validation status | Completeness | CV quality | Grounded evidence |",
        "|---|---:|---:|---:|",
        f"| {profile.validation_status.value if profile else '-'} | "
        f"{scores.get('profile_completeness.v2')} | {scores.get('cv_quality.v2')} | {evidence} |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_human_review_report(results: list[ProcessingResult], path: Path) -> None:
    succeeded = [
        result for result in results if result.status.value == "succeeded" and result.profile
    ]
    failed = [
        result for result in results if result.status.value != "succeeded" or not result.profile
    ]
    lines = [
        "# Human Review Executive Summary",
        "",
        f"**Successful profiles:** {len(succeeded)} | **Processing failures:** {len(failed)}",
        "",
    ]
    if results:
        lines.extend(
            [
                "## Extraction Funnel (entity counts)",
                "",
                "| File | Initial LLM E/P | Initial errors | Correction LLM E/P | "
                "Correction errors | DTO-valid E/P | Before reconciliation E/P | Final E/P |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in results:
            source = result.metadata.source_name if result.metadata else result.source_id
            funnel = result.audit.get("extractionFunnel")
            provider = funnel.get("provider", {}) if isinstance(funnel, dict) else {}
            initial = provider.get("initial")
            correction = provider.get("correction")
            initial = initial if isinstance(initial, dict) else {}
            correction = correction if isinstance(correction, dict) else {}
            reconciliation_funnel = funnel if isinstance(funnel, dict) else {}
            before_reconciliation = reconciliation_funnel.get("beforeReconciliation")
            after_reconciliation = reconciliation_funnel.get("afterReconciliation")
            lines.append(
                f"| {source} | {_funnel_pair(initial.get('llmRecognized'))} | "
                f"{_validation_error_count(initial)} | "
                f"{_funnel_pair(correction.get('llmRecognized')) if correction else '-'} | "
                f"{_validation_error_count(correction) if correction else '-'} | "
                f"{_funnel_pair((correction or initial).get('dtoValidated'))} | "
                f"{_funnel_pair(before_reconciliation)} | "
                f"{_funnel_pair(after_reconciliation)} |"
            )
        lines.append("")
    for idx, res in enumerate(succeeded, start=1):
        prof = res.profile
        name = prof.candidate_name if (prof and prof.candidate_name) else "Unknown Candidate"
        headline = (
            prof.headline
            if (prof and prof.headline)
            else ((prof.summary[:80] + "...") if (prof and prof.summary) else "N/A")
        )
        email = prof.email if (prof and prof.email) else "N/A"
        phone = prof.phone if (prof and prof.phone) else "N/A"
        exp_years = (
            f"{prof.total_experience_years} years"
            if (prof and prof.total_experience_years is not None)
            else "N/A"
        )

        edu_list = []
        if prof and prof.education:
            for ed in prof.education:
                deg = ed.degree or ed.field_of_study or "Degree"
                inst = ed.institution or "Institution"
                gpa = (
                    f" (GPA {ed.gpa.value:g}/{ed.gpa.scale:g})"
                    if ed.gpa and ed.gpa.scale
                    else f" (GPA {ed.gpa.value:g})"
                    if ed.gpa
                    else ""
                )
                edu_list.append(f"{deg} @ {inst}{gpa}")
        edu_str = ", ".join(edu_list) if edu_list else "N/A"

        skills_str = "N/A"
        if prof and prof.skills:
            skills_str = ", ".join(s.name for s in prof.skills[:8])
            if len(prof.skills) > 8:
                skills_str += f" (+{len(prof.skills) - 8} more)"

        exp_list = []
        if prof and prof.experiences:
            for ex in prof.experiences[:2]:
                title = ex.job_title or "Role"
                comp = ex.company or "Company"
                exp_list.append(f"{title} @ {comp}")
        exp_str = ", ".join(exp_list) if exp_list else "N/A"

        achievements = []
        if prof:
            for ex in prof.experiences:
                achievements.extend(ex.achievements)
            for pr in prof.projects:
                achievements.extend(pr.achievements)
        ach_str = f"{len(achievements)} metric(s)" if achievements else "None"
        if achievements:
            first_ach = achievements[0]
            sample_ach = first_ach[:60] + "..." if len(first_ach) > 60 else first_ach
            ach_str += f' (e.g. "{sample_ach}")'

        language_details = []
        if prof:
            for language in prof.language_proficiencies:
                detail = language.language
                if language.exam and language.score is not None:
                    detail += f" ({language.exam} {language.score:g})"
                elif language.proficiency:
                    detail += f" ({language.proficiency})"
                language_details.append(detail)
        language_str = ", ".join(language_details) if language_details else "N/A"
        awards_str = (
            ", ".join(item.title for item in (prof.honors_awards if prof else [])[:3]) or "N/A"
        )

        scores = {s.name: s.score for s in res.scores}
        status = prof.validation_status.value if prof else "N/A"
        comp_score = scores.get("profile_completeness.v2", "N/A")
        qual_score = scores.get("cv_quality.v2", "N/A")
        file_name = res.metadata.source_name if res.metadata else res.source_id

        lines.extend(
            [
                "---",
                f"### 👤 {idx}. {name.upper()}",
                f"* **File:** `{file_name}` | **Status:** `{status}`",
                f"* **Headline:** {headline}",
                f"* **Contact:** Email: `{email}` | Phone: `{phone}`",
                f"* **Total Experience:** `{exp_years}` | **Recent Roles:** {exp_str}",
                f"* **Education:** {edu_str}",
                f"* **Language Credentials:** {language_str}",
                f"* **Honors & Awards:** {awards_str}",
                f"* **Top Skills:** {skills_str}",
                f"* **Quantified Achievements:** {ach_str}",
                f"* **Scores:** Completeness: `{comp_score}%` | CV Quality: `{qual_score}`",
                "",
            ]
        )

    if failed:
        lines.extend(["## Processing Failures", ""])
        for result in failed:
            source = result.metadata.source_name if result.metadata else result.source_id
            codes = ", ".join(issue.code.value for issue in result.errors) or "UNKNOWN_FAILURE"
            lines.append(f"- `{source}` — `{codes}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reduction(tokens: object) -> float:
    if not isinstance(tokens, dict):
        return 0.0
    before = int(tokens.get("documentBefore", 0))
    after = int(tokens.get("documentAfter", 0))
    return round((before - after) / before * 100, 2) if before else 0.0


def _funnel_pair(value: object) -> str:
    if not isinstance(value, dict):
        return "N/A"
    experiences = value.get("experiences")
    projects = value.get("projects")
    if not isinstance(experiences, dict) or not isinstance(projects, dict):
        return "N/A"
    return f"{experiences.get('entities', 0)}/{projects.get('entities', 0)}"


def _validation_error_count(attempt: object) -> str:
    if not isinstance(attempt, dict):
        return "N/A"
    errors = attempt.get("validationErrors")
    return str(len(errors)) if isinstance(errors, list) else "0"
