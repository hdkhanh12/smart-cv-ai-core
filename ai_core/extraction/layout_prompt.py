"""Compact, layout-aware transcript for the one structured LLM request."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_core.schemas import UnifiedBlock, UnifiedDocument

_ROW_TOLERANCE = 16.0
_SEPARATOR_NOISE = re.compile(r"^[-_\*=~#\s]{2,}$")


@dataclass(frozen=True)
class _PlacedBlock:
    block: UnifiedBlock
    column: str
    vertical_center: float | None


def _column_threshold(blocks: list[UnifiedBlock]) -> float | None:
    starts = sorted(block.bbox[0] for block in blocks if block.bbox is not None)
    if len(starts) < 6:
        return None
    largest_gap, index = max(
        (
            (right - left, offset)
            for offset, (left, right) in enumerate(zip(starts, starts[1:], strict=False))
        ),
        default=(0.0, 0),
    )
    if largest_gap < 80 or index + 1 < 3 or len(starts) - index - 1 < 3:
        return None
    return (starts[index] + starts[index + 1]) / 2


def _placed_blocks(blocks: list[UnifiedBlock], threshold: float | None) -> list[_PlacedBlock]:
    placed: list[_PlacedBlock] = []
    for block in blocks:
        if block.bbox is None:
            placed.append(_PlacedBlock(block, "UNPOSITIONED", None))
            continue
        left, bottom, right, top = block.bbox
        center = (bottom + top) / 2
        if threshold is None:
            column = "FULL"
        elif (left + right) / 2 < threshold:
            column = "LEFT"
        else:
            column = "RIGHT"
        placed.append(_PlacedBlock(block, column, center))
    return placed


def _block_line(placed: _PlacedBlock) -> str:
    block = placed.block
    clean_text = " ".join(block.text.split())
    if block.type.value == "section_header":
        return f"\n## {clean_text}\n"
    if block.type.value == "list_item":
        return f"- {clean_text}"
    return clean_text


def _positioned_center(item: _PlacedBlock) -> float:
    assert item.vertical_center is not None
    return item.vertical_center


def _single_column_lines(placed: list[_PlacedBlock]) -> list[str]:
    ordered = sorted(
        placed,
        key=lambda item: (
            item.vertical_center is None,
            -(item.vertical_center or 0),
            item.block.reading_order,
        ),
    )
    return [_block_line(item) for item in ordered]


def _two_column_lines(placed: list[_PlacedBlock]) -> list[str]:
    positioned = [item for item in placed if item.vertical_center is not None]
    unpositioned = [item for item in placed if item.vertical_center is None]
    positioned.sort(key=lambda item: (-_positioned_center(item), item.block.reading_order))
    rows: list[list[_PlacedBlock]] = []
    for item in positioned:
        previous_center = _positioned_center(rows[-1][0]) if rows else None
        is_new_row = (
            previous_center is None
            or abs(previous_center - _positioned_center(item)) > _ROW_TOLERANCE
        )
        if is_new_row:
            rows.append([item])
        else:
            rows[-1].append(item)

    lines: list[str] = []
    for row in rows:
        for column in ("FULL", "LEFT", "RIGHT"):
            for item in row:
                if item.column == column:
                    lines.append(_block_line(item))
    for item in unpositioned:
        lines.append(_block_line(item))
    return lines


def serialize_layout_transcript(document: UnifiedDocument) -> str:
    """Serialize a UnifiedDocument as a natural, clean document reading transcript."""
    lines: list[str] = []
    for page in document.pages:
        blocks = [
            block for block in page.blocks
            if block.text.strip() and not _SEPARATOR_NOISE.match(block.text.strip())
        ]
        threshold = _column_threshold(blocks)
        placed = _placed_blocks(blocks, threshold)
        if threshold is None:
            lines.extend(_single_column_lines(placed))
        else:
            lines.extend(_two_column_lines(placed))
    return "\n".join(lines).strip()
