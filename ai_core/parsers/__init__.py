"""Document parser interfaces."""

from ai_core.parsers.base import DocumentParser
from ai_core.parsers.docx import DocxParser
from ai_core.parsers.pdf import PdfParser
from ai_core.schemas import ParsedDocument
from ai_core.validation import ValidatedInput


def parse_document(document: ValidatedInput) -> ParsedDocument:
    if document.metadata.extension == ".pdf":
        return PdfParser().parse(document)
    return DocxParser().parse(document)


__all__ = ["DocumentParser", "DocxParser", "PdfParser", "parse_document"]
