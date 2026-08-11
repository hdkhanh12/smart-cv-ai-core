"""Text-based PDF parsing through PyMuPDF."""

from __future__ import annotations

from typing import Any

import pymupdf

from ai_core.errors import CoreError, ErrorCode, Issue, WarningCode
from ai_core.preprocessing import preprocess_pages
from ai_core.schemas import PageContent, ParsedDocument
from ai_core.validation import ValidatedInput

MIN_TEXT_CHARACTERS = 40


class PdfParser:
    def parse(self, document: ValidatedInput) -> ParsedDocument:
        try:
            pdf = pymupdf.open(document.path)  # type: ignore[no-untyped-call]
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise CoreError(
                ErrorCode.CORRUPT_PDF,
                "PDF could not be opened by the parser.",
                stage="parsing",
                details={"sourceName": document.path.name},
            ) from exc

        with pdf:
            if pdf.needs_pass:
                raise CoreError(
                    ErrorCode.ENCRYPTED_PDF,
                    "Encrypted PDF files are not supported.",
                    stage="parsing",
                    details={"sourceName": document.path.name},
                )
            pages: list[PageContent] = []
            for index in range(pdf.page_count):
                page: pymupdf.Page = pdf.load_page(index)  # type: ignore[no-untyped-call]
                pages.append(
                    PageContent(
                        page_number=index + 1,
                        text=page.get_text(  # type: ignore[no-untyped-call]
                            "text",
                            sort=True,
                        ),
                    )
                )
            pdf_metadata: dict[str, Any] = pdf.metadata or {}
            metadata = document.metadata.model_copy(
                update={
                    "page_count": pdf.page_count,
                    "author": pdf_metadata.get("author") or None,
                    "title": pdf_metadata.get("title") or None,
                    "is_encrypted": False,
                }
            )

        processed = preprocess_pages(pages)
        warnings = list(processed.warnings)
        character_count = len(processed.normalized_text)
        if character_count == 0:
            warnings.append(
                Issue(
                    code=WarningCode.NO_TEXT_LAYER,
                    message="PDF contains no extractable text layer; OCR is not enabled.",
                    stage="parsing",
                )
            )
        elif character_count < MIN_TEXT_CHARACTERS:
            warnings.append(
                Issue(
                    code=WarningCode.TEXT_TOO_SHORT,
                    message="Extracted PDF text is unusually short.",
                    stage="parsing",
                    details={
                        "characterCount": character_count,
                        "minimumCharacterCount": MIN_TEXT_CHARACTERS,
                    },
                )
            )
        return ParsedDocument(
            metadata=metadata,
            pages=pages,
            raw_text=processed.raw_text,
            normalized_text=processed.normalized_text,
            warnings=warnings,
        )
