"""Search package exporting dense JSON search, JD/Query analysis, and BGE-M3 vector generation."""

from ai_core.search.analyzer import (
    _prune_jd_noise,
    analyze_query_or_jd,
    build_query_text,
)
from ai_core.search.legacy_search import (
    SearchMatch,
    read_jd,
    search_results,
    write_search_report,
)

__all__ = [
    "read_jd",
    "search_results",
    "write_search_report",
    "SearchMatch",
    "analyze_query_or_jd",
    "build_query_text",
    "_prune_jd_noise",
]
