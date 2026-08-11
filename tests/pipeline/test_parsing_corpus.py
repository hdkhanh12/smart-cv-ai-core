from __future__ import annotations

from pathlib import Path

import pymupdf

from evaluation.parsing_smoke import run_parsing_smoke


def test_corpus_runner_isolates_invalid_file_and_omits_text_and_paths(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    valid = pymupdf.open()
    page = valid.new_page()
    page.insert_text((72, 72), "Synthetic CV content with enough text for parser smoke testing")
    valid.save(corpus / "valid.pdf")
    valid.close()
    (corpus / "private-name.pdf").write_bytes(b"%PDF-1.4\nbroken\n%%EOF\n")
    output = tmp_path / "report.json"

    report = run_parsing_smoke(corpus, output)
    serialized = output.read_text(encoding="utf-8")

    assert report["inputFileCount"] == 2
    assert report["succeeded"] == 1
    assert report["failed"] == 1
    assert "Synthetic CV content" not in serialized
    assert "private-name.pdf" not in serialized
    assert report["privacy"]["rawTextPersisted"] is False
