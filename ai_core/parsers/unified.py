"""Provider-neutral unified document extraction and diagnostics."""

from __future__ import annotations

import importlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from ai_core.errors import Issue, WarningCode
from ai_core.schemas import (
    BlockType,
    DiagnosticStatus,
    ExtractionDiagnostics,
    UnifiedBlock,
    UnifiedDocument,
    UnifiedPage,
)
from ai_core.sections import heading_kind
from ai_core.validation import ValidatedInput


@lru_cache(maxsize=1)
def hybrid_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "configs" / "hybrid_v2.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


class UnifiedExtractor(Protocol):
    name: str

    def extract(self, document: ValidatedInput) -> UnifiedDocument: ...


class FallbackExtractor(Protocol):
    name: str

    def extract(self, document: ValidatedInput, failed: UnifiedDocument) -> UnifiedDocument: ...


_IMAGE_ONLY_MARKDOWN = re.compile(
    r"<!--\s*image\s*-->|!\[[^\]]*\]\([^)]*\)",
    re.IGNORECASE,
)


def effective_document_text(markdown: str) -> str:
    """Return text that can actually support downstream extraction.

    Docling represents raster-only PDF pages as image comments.  Those comments
    are transport markup, not recoverable CV text, and must not prevent the
    local OCR fallback from running.
    """

    without_images = _IMAGE_ONLY_MARKDOWN.sub("", markdown)
    return "\n".join(line.strip() for line in without_images.splitlines() if line.strip())


def diagnose(
    markdown: str,
    pages: list[UnifiedPage],
    *,
    expected_page_count: int | None = None,
    expect_geometry: bool = False,
) -> ExtractionDiagnostics:
    thresholds = hybrid_config()["diagnostics"]
    issues: list[Issue] = []
    effective_text = effective_document_text(markdown)
    character_count = len(effective_text)
    raw_character_count = len(markdown.strip())
    blocks = [block for page in pages for block in page.blocks]
    heading_count = sum(block.type == BlockType.SECTION_HEADER for block in blocks)
    reading_orders = [block.reading_order for page in pages for block in page.blocks]
    if expected_page_count is not None and len(pages) != expected_page_count:
        issues.append(
            Issue(
                code=WarningCode.PAGE_COUNT_MISMATCH,
                message="Unified document page count differs from the source layout document.",
                stage="extraction_diagnostics",
                details={"expectedPageCount": expected_page_count, "unifiedPageCount": len(pages)},
            )
        )
    if expect_geometry and blocks and not any(block.bbox is not None for block in blocks):
        issues.append(
            Issue(
                code=WarningCode.MISSING_LAYOUT_GEOMETRY,
                message="Layout extractor produced text blocks without usable geometry.",
                stage="extraction_diagnostics",
            )
        )
    if any(block.type == BlockType.TABLE for block in blocks):
        issues.append(
            Issue(
                code=WarningCode.TABLE_LAYOUT_AMBIGUITY,
                message="Table content requires cell-aware attachment during structured "
                "extraction.",
                stage="extraction_diagnostics",
            )
        )
    for page in pages:
        x_starts = sorted(block.bbox[0] for block in page.blocks if block.bbox is not None)
        if len(x_starts) < 6:
            continue
        largest_gap, split_at = max(
            (
                (right - left, index)
                for index, (left, right) in enumerate(zip(x_starts, x_starts[1:], strict=False))
            ),
            default=(0.0, 0),
        )
        if (
            largest_gap >= 80
            and split_at + 1 >= 3
            and len(x_starts) - split_at - 1 >= 3
        ):
            issues.append(
                Issue(
                    code=WarningCode.SUSPICIOUS_READING_ORDER,
                    message="Page has distinct horizontal block clusters; column attachment is "
                    "ambiguous.",
                    stage="extraction_diagnostics",
                    details={
                        "pageNumber": page.page_number,
                        "largestHorizontalGap": round(largest_gap, 2),
                    },
                )
            )
    if character_count < int(thresholds["minimumCharacters"]):
        issues.append(
            Issue(
                code=WarningCode.FALLBACK_REQUIRED,
                message="Unified document text is too short for reliable extraction.",
                stage="extraction_diagnostics",
                details={
                    "characterCount": character_count,
                    "rawCharacterCount": raw_character_count,
                    "imagePlaceholderCount": len(_IMAGE_ONLY_MARKDOWN.findall(markdown)),
                },
            )
        )
    if len(blocks) < int(thresholds["minimumBlocks"]):
        issues.append(
            Issue(
                code=WarningCode.FALLBACK_REQUIRED,
                message="Unified document contains too few structural blocks.",
                stage="extraction_diagnostics",
                details={"blockCount": len(blocks)},
            )
        )
    if character_count >= int(thresholds["longDocumentCharacters"]) and heading_count < int(
        thresholds["minimumHeadingsForLongDocument"]
    ):
        issues.append(
            Issue(
                code=WarningCode.SUSPICIOUS_READING_ORDER,
                message="Long document has insufficient recovered section structure.",
                stage="extraction_diagnostics",
                details={"headingCount": heading_count},
            )
        )
    if reading_orders != sorted(set(reading_orders)):
        issues.append(
            Issue(
                code=WarningCode.FALLBACK_REQUIRED,
                message="Block reading order is duplicated or non-monotonic.",
                stage="extraction_diagnostics",
            )
        )
    has_fallback_issue = any(issue.code == WarningCode.FALLBACK_REQUIRED for issue in issues)
    status = DiagnosticStatus.FALLBACK_REQUIRED if has_fallback_issue else DiagnosticStatus.USABLE
    return ExtractionDiagnostics(
        status=status,
        issues=issues,
        text_character_count=character_count,
        block_count=len(blocks),
        heading_count=heading_count,
    )


def unified_from_lines(
    document: ValidatedInput,
    extractor: str,
    page_lines: list[list[str]],
    *,
    markdown: str | None = None,
) -> UnifiedDocument:
    pages: list[UnifiedPage] = []
    order = 1
    for page_number, lines in enumerate(page_lines, start=1):
        blocks: list[UnifiedBlock] = []
        for line in lines:
            text = line.strip()
            if not text:
                continue
            blocks.append(
                UnifiedBlock(
                    id=f"p{page_number}-b{len(blocks) + 1}",
                    type=BlockType.SECTION_HEADER if heading_kind(text) else BlockType.TEXT,
                    text=text,
                    reading_order=order,
                )
            )
            order += 1
        pages.append(UnifiedPage(page_number=page_number, blocks=blocks))
    content = (
        markdown
        if markdown is not None
        else "\n".join(block.text for page in pages for block in page.blocks)
    )
    return UnifiedDocument(
        file_hash=document.metadata.sha256,
        extractor=extractor,
        markdown=content,
        pages=pages,
        diagnostics=diagnose(content, pages),
    )


def _docling_block_type(item: Any) -> BlockType:
    label = str(getattr(item, "label", ""))
    if label == "section_header":
        return BlockType.SECTION_HEADER
    if label == "list_item":
        return BlockType.LIST_ITEM
    if label == "table":
        return BlockType.TABLE
    return BlockType.TEXT


def _docling_table_text(item: Any) -> str:
    data = getattr(item, "data", None)
    cells = getattr(data, "table_cells", None)
    if not cells:
        return ""
    rows: dict[int, dict[int, str]] = {}
    for cell in cells:
        row = int(getattr(cell, "start_row_offset_idx", 0))
        column = int(getattr(cell, "start_col_offset_idx", 0))
        rows.setdefault(row, {})[column] = str(getattr(cell, "text", "")).strip()
    width = max((max(row) for row in rows.values() if row), default=-1) + 1
    return "\n".join(
        "| " + " | ".join(columns.get(column, "") for column in range(width)) + " |"
        for _, columns in sorted(rows.items())
    )


def _docling_item_text(item: Any) -> str:
    text = getattr(item, "text", None)
    if text is not None:
        return str(text).strip()
    if _docling_block_type(item) == BlockType.TABLE:
        return _docling_table_text(item)
    return ""


def _docling_provenance(item: Any) -> tuple[int, tuple[float, float, float, float] | None] | None:
    for provenance in getattr(item, "prov", []) or []:
        page_number = getattr(provenance, "page_no", None)
        bbox = getattr(provenance, "bbox", None)
        if page_number is None:
            continue
        if bbox is None:
            return int(page_number), None
        return (
            int(page_number),
            (float(bbox.l), float(bbox.b), float(bbox.r), float(bbox.t)),
        )
    return None


def unified_from_docling_document(
    document: ValidatedInput,
    extractor: str,
    docling_document: Any,
    *,
    markdown: str,
    leading_lines: list[str] | None = None,
) -> UnifiedDocument:
    """Preserve Docling-native page provenance instead of rebuilding one fake page."""

    page_numbers = sorted(
        {
            int(getattr(page, "page_no", page_number))
            for page_number, page in getattr(docling_document, "pages", {}).items()
        }
    )
    page_items: dict[int, list[tuple[BlockType, str, tuple[float, float, float, float] | None]]] = {
        page_number: [] for page_number in page_numbers
    }
    for item, _ in docling_document.iterate_items():
        text = _docling_item_text(item)
        provenance = _docling_provenance(item)
        if not text or provenance is None:
            continue
        page_number, bbox = provenance
        page_items.setdefault(page_number, []).append((_docling_block_type(item), text, bbox))

    if leading_lines:
        first_page = page_numbers[0] if page_numbers else 1
        page_items.setdefault(first_page, [])[:0] = [
            (BlockType.TEXT, line.strip(), None) for line in leading_lines if line.strip()
        ]
        if first_page not in page_numbers:
            page_numbers.append(first_page)
            page_numbers.sort()

    pages: list[UnifiedPage] = []
    reading_order = 1
    for page_number in page_numbers:
        blocks: list[UnifiedBlock] = []
        for block_type, text, bbox in page_items.get(page_number, []):
            blocks.append(
                UnifiedBlock(
                    id=f"p{page_number}-b{len(blocks) + 1}",
                    type=block_type,
                    text=text,
                    bbox=bbox,
                    reading_order=reading_order,
                )
            )
            reading_order += 1
        pages.append(UnifiedPage(page_number=page_number, blocks=blocks))
    return UnifiedDocument(
        file_hash=document.metadata.sha256,
        extractor=extractor,
        markdown=markdown,
        pages=pages,
        diagnostics=diagnose(
            markdown,
            pages,
            expected_page_count=len(page_numbers),
            expect_geometry=True,
        ),
    )


class PyMuPdf4LlmAdapter:
    name = "pymupdf4llm"

    def extract(self, document: ValidatedInput) -> UnifiedDocument:
        if document.metadata.extension != ".pdf":
            raise ValueError("PyMuPDF4LLM adapter only accepts PDF")
        try:
            pymupdf4llm = importlib.import_module("pymupdf4llm")
        except ImportError as exc:
            raise RuntimeError("pymupdf4llm dependency is not installed") from exc
        markdown = str(pymupdf4llm.to_markdown(str(document.path)))
        page_lines = [markdown.splitlines()]
        return unified_from_lines(document, self.name, page_lines, markdown=markdown)


_INVALID_NAME_PATTERN = re.compile(
    r"\b(graduated|bachelor|master|degree|engineer|developer|architect|analyst|intern|fresher|scientist)\b",
    re.I,
)


def _extract_pymupdf_header_name(pdf_path: Path) -> str | None:
    try:
        pymupdf = importlib.import_module("pymupdf")
        pdf = pymupdf.open(pdf_path)
        with pdf:
            if pdf.page_count > 0:
                first_page = pdf.load_page(0)
                text = str(first_page.get_text("text", sort=True))
                for raw_line in text.splitlines()[:5]:
                    line = raw_line.strip()
                    if not line or "@" in line or "+" in line or "http" in line:
                        continue
                    line = re.sub(r"[\d ().-]{7,}", "", line).strip()
                    if "," in line:
                        parts = [p.strip() for p in line.split(",") if p.strip()]
                        line = " ".join(reversed(parts))
                    words = line.split()
                    if (
                        2 <= len(words) <= 5
                        and line.replace(" ", "").isalpha()
                        and not _INVALID_NAME_PATTERN.search(line)
                    ):
                        return line
    except Exception:
        pass
    return None


class DoclingAdapter:
    name = "docling"

    def __init__(self) -> None:
        self._converter: Any | None = None

    def extract(self, document: ValidatedInput) -> UnifiedDocument:
        if document.metadata.extension != ".pdf":
            raise ValueError("Docling adapter proof of concept only accepts PDF")
        try:
            converter_module = importlib.import_module("docling.document_converter")
            options_module = importlib.import_module("docling.datamodel.pipeline_options")
            models_module = importlib.import_module("docling.datamodel.base_models")
        except ImportError as exc:
            raise RuntimeError("docling dependency is not installed") from exc
        if self._converter is None:
            pipeline_options = options_module.PdfPipelineOptions()
            pipeline_options.do_ocr = False
            self._converter = converter_module.DocumentConverter(
                format_options={
                    models_module.InputFormat.PDF: converter_module.PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        result = self._converter.convert(str(document.path))
        markdown = str(result.document.export_to_markdown())
        lines = markdown.splitlines()
        fallback_name: str | None = None

        has_name_in_docling = any(
            2 <= len(line.strip().split()) <= 5
            and line.strip().replace(" ", "").isalpha()
            and not _INVALID_NAME_PATTERN.search(line.strip())
            for line in lines[:5]
            if line.strip() and not line.startswith("#")
        )
        if not has_name_in_docling:
            fallback_name = _extract_pymupdf_header_name(document.path)
            if fallback_name:
                markdown = f"{fallback_name}\n\n{markdown}"

        return unified_from_docling_document(
            document,
            self.name,
            result.document,
            markdown=markdown,
            leading_lines=[fallback_name] if fallback_name else None,
        )


class DoclingRapidOcrAdapter:
    """Native local OCR fallback using Docling's RapidOCR integration."""

    name = "docling-rapidocr"
    _INLINE_SECTION = re.compile(
        r"(?=\b(?:SKILLS|EXPERIENCE|EDUCATION|PROJECTS|RESEARCH|CERTIFICATIONS)\s*:)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._converter: Any | None = None

    def extract(self, document: ValidatedInput, failed: UnifiedDocument) -> UnifiedDocument:
        if document.metadata.extension != ".pdf":
            raise ValueError("Docling RapidOCR fallback only accepts PDF")
        try:
            converter_module = importlib.import_module("docling.document_converter")
            options_module = importlib.import_module("docling.datamodel.pipeline_options")
            models_module = importlib.import_module("docling.datamodel.base_models")
            importlib.import_module("rapidocr")
            importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise RuntimeError(
                "Docling RapidOCR fallback requires rapidocr and onnxruntime."
            ) from exc
        if self._converter is None:
            pipeline_options = options_module.PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = options_module.RapidOcrOptions(
                lang=["english"],
                backend="onnxruntime",
                force_full_page_ocr=True,
                print_verbose=False,
            )
            self._converter = converter_module.DocumentConverter(
                format_options={
                    models_module.InputFormat.PDF: converter_module.PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        result = self._converter.convert(str(document.path))
        markdown = str(result.document.export_to_markdown())
        return unified_from_docling_document(
            document,
            self.name,
            result.document,
            markdown=markdown,
        )


class ParsedDocumentAdapter:
    """Compatibility adapter for DOCX and rollback of the baseline parser."""

    name = "parsed-document-compatibility"

    def __init__(self, page_texts: list[str]) -> None:
        self.page_texts = page_texts

    def extract(self, document: ValidatedInput) -> UnifiedDocument:
        return unified_from_lines(
            document,
            self.name,
            [text.splitlines() for text in self.page_texts],
        )


@lru_cache(maxsize=1)
def configured_main_extractor() -> UnifiedExtractor:
    selected = str(hybrid_config()["mainExtractor"])
    if selected == "docling":
        return DoclingAdapter()
    if selected == "pymupdf4llm":
        return PyMuPdf4LlmAdapter()
    raise ValueError(f"Unsupported main extractor: {selected}")


@lru_cache(maxsize=1)
def configured_fallback_extractor() -> FallbackExtractor | None:
    selected = str(hybrid_config()["fallbackExtractor"])
    if selected == "none":
        return None
    if selected == "docling_rapidocr":
        return DoclingRapidOcrAdapter()
    raise ValueError(f"Unsupported fallback extractor: {selected}")


def extract_with_fallback(
    document: ValidatedInput,
    main: UnifiedExtractor,
    fallback: FallbackExtractor | None = None,
) -> UnifiedDocument:
    unified = main.extract(document)
    if unified.diagnostics.status == DiagnosticStatus.USABLE or fallback is None:
        return unified
    return fallback.extract(document, unified)
