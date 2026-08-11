"""Create local, no-LLM artefacts for CV layout-boundary review.

This script deliberately calls only the main Docling adapter.  It does not
load the extraction provider, environment credentials, OCR fallback, or any
remote service.  It is intended to answer whether layout is already degraded
before a UnifiedDocument reaches the LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pymupdf

from ai_core.parsers.unified import DoclingAdapter
from ai_core.schemas import UnifiedDocument
from ai_core.validation import validate_input


def _safe_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)


def _pymupdf_blocks(pdf_path: Path) -> tuple[list[dict[str, Any]], int]:
    pages: list[dict[str, Any]] = []
    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            raw_blocks = page.get_text("blocks", sort=True)
            blocks = [
                {
                    "bbox": [round(float(value), 2) for value in block[:4]],
                    "text": str(block[4]).strip(),
                    "blockNumber": int(block[5]),
                    "blockType": int(block[6]),
                }
                for block in raw_blocks
                if str(block[4]).strip()
            ]
            pages.append(
                {
                    "pageNumber": index,
                    "width": round(float(page.rect.width), 2),
                    "height": round(float(page.rect.height), 2),
                    "text": page.get_text("text", sort=True),
                    "blocks": blocks,
                }
            )
        return pages, document.page_count


def _render_pdf(pdf_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
            pixmap.save(output_dir / f"page-{index:03}.png")


def _audit_row(
    source: Path,
    output_dir: Path,
    pdf_pages: int,
    pymupdf_pages: list[dict[str, Any]],
    unified: Any,
) -> dict[str, object]:
    unified_pages = unified.pages
    unified_blocks = [block for page in unified_pages for block in page.blocks]
    page_numbers = sorted({block.id.split("-")[0] for block in unified_blocks})
    return {
        "file": source.name,
        "output": str(output_dir),
        "pdfPages": pdf_pages,
        "pymupdfTextBlocks": sum(len(page["blocks"]) for page in pymupdf_pages),
        "doclingMarkdownCharacters": len(unified.markdown),
        "unifiedPages": len(unified_pages),
        "unifiedBlocks": len(unified_blocks),
        "unifiedBlockPagePrefixes": ", ".join(page_numbers) or "none",
        "unifiedBboxBlocks": sum(block.bbox is not None for block in unified_blocks),
        "diagnosticStatus": unified.diagnostics.status.value,
        "diagnosticIssues": ", ".join(issue.code.value for issue in unified.diagnostics.issues)
        or "none",
    }


def _write_case(source: Path, output_root: Path, adapter: DoclingAdapter) -> dict[str, object]:
    output_dir = output_root / _safe_stem(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    validated = validate_input(source)
    unified = adapter.extract(validated)
    pymupdf_pages, pdf_pages = _pymupdf_blocks(source)

    (output_dir / "docling_markdown.md").write_text(unified.markdown, encoding="utf-8")
    (output_dir / "unified_document.json").write_text(
        unified.model_dump_json(indent=2), encoding="utf-8"
    )
    (output_dir / "pymupdf_layout_blocks.json").write_text(
        json.dumps(pymupdf_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _render_pdf(source, output_dir / "rendered_pages")
    return _audit_row(source, output_dir, pdf_pages, pymupdf_pages, unified)


def _read_existing_case(source: Path, output_root: Path) -> dict[str, object]:
    output_dir = output_root / _safe_stem(source)
    unified_path = output_dir / "unified_document.json"
    if not unified_path.is_file():
        raise FileNotFoundError(f"No completed audit artefact for {source.name}")
    unified = UnifiedDocument.model_validate_json(unified_path.read_text(encoding="utf-8"))
    pymupdf_pages, pdf_pages = _pymupdf_blocks(source)
    return _audit_row(source, output_dir, pdf_pages, pymupdf_pages, unified)


def _report(rows: list[dict[str, object]]) -> str:
    headers = [
        "File",
        "PDF pages",
        "PyMuPDF blocks",
        "Docling chars",
        "Unified pages",
        "Unified blocks",
        "Unified page prefixes",
        "BBoxes kept",
        "Diagnostics",
    ]
    lines = [
        "# Layout Boundary Audit",
        "",
        "Local-only Docling and PyMuPDF inspection. No LLM request was made.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| {file} | {pdfPages} | {pymupdfTextBlocks} | {doclingMarkdownCharacters} | "
            "{unifiedPages} | {unifiedBlocks} | {unifiedBlockPagePrefixes} | "
            "{unifiedBboxBlocks} | {diagnosticStatus}: {diagnosticIssues} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Review order",
            "",
            "For each case, compare `rendered_pages/` with `docling_markdown.md`, then compare "
            "that Markdown with `unified_document.json`. `pymupdf_layout_blocks.json` is a local "
            "visual-layout reference (coordinates and sorted text blocks), not an extraction "
            "result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit PDF -> Docling -> UnifiedDocument boundaries."
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing the selected PDF test suite."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Empty/new local audit directory."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Build the summary from completed existing case artefacts without rerunning Docling.",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.pdf"))
    if not files:
        parser.error(f"No PDF files found in {args.input_dir}")
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        rows = [_read_existing_case(path, args.output) for path in files]
    else:
        adapter = DoclingAdapter()
        rows = [_write_case(path, args.output, adapter) for path in files]
    (args.output / "layout-audit-summary.md").write_text(_report(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
