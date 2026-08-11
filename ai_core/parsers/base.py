"""Common parser protocol implemented by PDF and DOCX parsers in Milestone 2."""

from __future__ import annotations

from typing import Protocol

from ai_core.schemas import ParsedDocument
from ai_core.validation import ValidatedInput


class DocumentParser(Protocol):
    def parse(self, document: ValidatedInput) -> ParsedDocument:
        """Parse a validated document while preserving page-level content."""
        ...
