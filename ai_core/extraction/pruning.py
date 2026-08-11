"""Lossless-for-evidence prompt pruning for UnifiedDocument payloads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ai_core.schemas import UnifiedBlock, UnifiedDocument, UnifiedPage


def estimate_tokens(value: str) -> int:
    """Stable conservative estimate suitable for provider cost reporting."""

    return max(1, (len(value) + 3) // 4) if value else 0


@dataclass(frozen=True)
class PrunedDocument:
    document: UnifiedDocument
    input_tokens_before: int
    input_tokens_after: int


def prune_document(document: UnifiedDocument) -> PrunedDocument:
    """Remove empty blocks while preserving evidence and layout provenance."""

    before = document.model_dump(by_alias=True)
    before_tokens = estimate_tokens(json.dumps(before, ensure_ascii=False, separators=(",", ":")))
    pages: list[UnifiedPage] = []
    for page in document.pages:
        blocks = [
            UnifiedBlock(
                id=block.id,
                type=block.type,
                text=re.sub(r"\s+", " ", block.text).strip(),
                bbox=block.bbox,
                reading_order=block.reading_order,
            )
            for block in page.blocks
            if block.text.strip()
        ]
        pages.append(UnifiedPage(page_number=page.page_number, blocks=blocks))
    # `pages[].blocks[]` retains every evidence-bearing text fragment, ID, and
    # layout provenance. The latter is necessary for multi-column attachment.
    # The full Markdown is duplicate prompt content, so omit it from the remote
    # payload to achieve a material token reduction without losing grounding.
    # Keep a compact leading context for provider readability; the complete
    # evidence-bearing representation remains in pages/blocks.
    pruned = document.model_copy(update={"markdown": document.markdown[:240], "pages": pages})
    after = pruned.model_dump(by_alias=True, exclude_none=True)
    after_tokens = estimate_tokens(json.dumps(after, ensure_ascii=False, separators=(",", ":")))
    return PrunedDocument(pruned, before_tokens, after_tokens)
