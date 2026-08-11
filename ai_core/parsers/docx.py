"""DOCX parser preserving paragraph/table order."""

from __future__ import annotations

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table

from ai_core.errors import CoreError, ErrorCode, Issue, WarningCode
from ai_core.preprocessing import preprocess_pages
from ai_core.schemas import PageContent, ParsedDocument
from ai_core.validation import ValidatedInput

MIN_TEXT_CHARACTERS = 40


def _table_lines(table: Table) -> list[str]:
    lines: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return lines


class DocxParser:
    def parse(self, document: ValidatedInput) -> ParsedDocument:
        try:
            docx = Document(str(document.path))
        except (PackageNotFoundError, ValueError, KeyError) as exc:
            raise CoreError(
                ErrorCode.CORRUPT_DOCX,
                "DOCX could not be opened by the parser.",
                stage="parsing",
                details={"sourceName": document.path.name},
            ) from exc

        lines: list[str] = []
        for block in docx.iter_inner_content():
            if isinstance(block, Table):
                lines.extend(_table_lines(block))
            elif block.text.strip():
                lines.append(block.text)
        pages = [PageContent(page_number=1, text="\n".join(lines))]
        processed = preprocess_pages(pages)
        warnings = list(processed.warnings)
        if len(processed.normalized_text) < MIN_TEXT_CHARACTERS:
            warnings.append(
                Issue(
                    code=WarningCode.TEXT_TOO_SHORT,
                    message="Extracted DOCX text is unusually short.",
                    stage="parsing",
                    details={
                        "characterCount": len(processed.normalized_text),
                        "minimumCharacterCount": MIN_TEXT_CHARACTERS,
                    },
                )
            )
        properties = docx.core_properties
        metadata = document.metadata.model_copy(
            update={
                "page_count": None,
                "author": properties.author or None,
                "title": properties.title or None,
            }
        )
        return ParsedDocument(
            metadata=metadata,
            pages=pages,
            raw_text=processed.raw_text,
            normalized_text=processed.normalized_text,
            warnings=warnings,
        )
