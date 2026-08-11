from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_core.errors import WarningCode
from ai_core.extraction.layout_prompt import serialize_layout_transcript
from ai_core.extraction.pruning import prune_document
from ai_core.parsers.unified import unified_from_docling_document
from ai_core.schemas import BlockType
from ai_core.validation import validate_input


class _BBox:
    def __init__(self, left: float, bottom: float, right: float, top: float) -> None:
        self.__dict__.update({"l": left, "b": bottom, "r": right, "t": top})


@dataclass
class _Provenance:
    page_no: int
    bbox: _BBox


@dataclass
class _Page:
    page_no: int


@dataclass
class _Item:
    label: str
    text: str
    prov: list[_Provenance]


class _Document:
    pages = {1: _Page(1), 2: _Page(2)}

    def iterate_items(self) -> list[tuple[_Item, int]]:
        return [
            (
                _Item(
                    "section_header", "WORK EXPERIENCE", [_Provenance(1, _BBox(30, 700, 220, 720))]
                ),
                0,
            ),
            (_Item("text", "Company A", [_Provenance(1, _BBox(30, 650, 190, 670))]), 0),
            (_Item("list_item", "Built pipeline", [_Provenance(1, _BBox(310, 650, 520, 670))]), 0),
            (_Item("text", "Company B", [_Provenance(1, _BBox(30, 610, 190, 630))]), 0),
            (_Item("list_item", "Reduced cost", [_Provenance(1, _BBox(310, 610, 520, 630))]), 0),
            (_Item("text", "Education", [_Provenance(1, _BBox(30, 570, 190, 590))]), 0),
            (_Item("text", "Project", [_Provenance(1, _BBox(310, 570, 520, 590))]), 0),
            (_Item("section_header", "EDUCATION", [_Provenance(2, _BBox(30, 700, 200, 720))]), 0),
        ]


def _pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n%%EOF")
    return path


def test_docling_native_adapter_preserves_layout_provenance(tmp_path: Path) -> None:
    unified = unified_from_docling_document(
        validate_input(_pdf(tmp_path / "fixture.pdf")),
        "docling",
        _Document(),
        markdown="# fixture",
    )

    assert [page.page_number for page in unified.pages] == [1, 2]
    assert unified.pages[0].blocks[2].type == BlockType.LIST_ITEM
    assert unified.pages[0].blocks[1].bbox == (30.0, 650.0, 190.0, 670.0)
    reading_orders = [block.reading_order for page in unified.pages for block in page.blocks]
    assert reading_orders == list(range(1, 9))
    assert WarningCode.SUSPICIOUS_READING_ORDER in {
        issue.code for issue in unified.diagnostics.issues
    }
    assert WarningCode.MISSING_LAYOUT_GEOMETRY not in {
        issue.code for issue in unified.diagnostics.issues
    }
    assert prune_document(unified).document.pages[0].blocks[1].bbox == (
        30.0,
        650.0,
        190.0,
        670.0,
    )
    transcript = serialize_layout_transcript(unified)
    assert "PAGE 1 — TWO-COLUMN" in transcript
    assert "ROW 1 (top-to-bottom alignment):" in transcript
    assert "LEFT: [p1-b2 | text] Company A" in transcript
    assert "RIGHT: [p1-b3 | list_item] Built pipeline" in transcript
