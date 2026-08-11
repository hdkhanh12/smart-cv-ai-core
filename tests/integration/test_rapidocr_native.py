"""Native Docling + RapidOCR coverage using a generated non-PII scan fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from ai_core.parsers.unified import DoclingRapidOcrAdapter, unified_from_lines
from ai_core.schemas import DiagnosticStatus
from ai_core.validation import validate_input

pytestmark = pytest.mark.integration


def _synthetic_scan(path: Path) -> None:
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 80), "SYNTHETIC CANDIDATE", fill="black", font_size=48)
    draw.text((80, 180), "SKILLS: Python, SQL", fill="black", font_size=36)
    draw.text((80, 260), "EXPERIENCE: Built reliable data pipelines", fill="black", font_size=36)
    image_path = path.with_suffix(".png")
    image.save(image_path)
    pdf = pymupdf.open()
    page = pdf.new_page(width=1600, height=900)
    page.insert_image(page.rect, filename=image_path)
    pdf.save(path)
    pdf.close()


@pytest.mark.skipif(
    importlib.util.find_spec("rapidocr") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="Native RapidOCR dependencies are not installed.",
)
def test_docling_rapidocr_recovers_text_from_synthetic_scan(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-scan.pdf"
    _synthetic_scan(source)
    validated = validate_input(source)
    failed = unified_from_lines(validated, "fake-main", [["short"]])

    result = DoclingRapidOcrAdapter().extract(validated, failed)

    assert result.extractor == "docling-rapidocr"
    assert result.diagnostics.status == DiagnosticStatus.USABLE
    assert "SYNTHETIC" in result.markdown.upper()
    assert "PYTHON" in result.markdown.upper()
