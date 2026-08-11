from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ai_core.errors import CoreError, ErrorCode
from ai_core.validation import validate_input


def write_pdf(path: Path, body: bytes = b"1 0 obj\n<<>>\nendobj\n") -> Path:
    path.write_bytes(b"%PDF-1.4\n" + body + b"%%EOF\n")
    return path


def write_docx(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
    return path


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        (lambda tmp: tmp / "missing.pdf", ErrorCode.FILE_NOT_FOUND),
        (lambda tmp: tmp, ErrorCode.NOT_A_FILE),
        (lambda tmp: tmp / "cv.txt", ErrorCode.UNSUPPORTED_EXTENSION),
        (lambda tmp: tmp / "empty.pdf", ErrorCode.EMPTY_FILE),
    ],
)
def test_rejects_invalid_path_cases(
    tmp_path: Path,
    setup: object,
    code: ErrorCode,
) -> None:
    path = setup(tmp_path)  # type: ignore[operator]
    if path.suffix and not path.exists() and code != ErrorCode.FILE_NOT_FOUND:
        path.write_bytes(b"" if code == ErrorCode.EMPTY_FILE else b"text")
    with pytest.raises(CoreError) as caught:
        validate_input(path)
    assert caught.value.issue.code == code


def test_rejects_oversized_file(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "large.pdf")
    with pytest.raises(CoreError) as caught:
        validate_input(path, max_size_bytes=4)
    assert caught.value.issue.code == ErrorCode.FILE_TOO_LARGE


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"not a pdf", ErrorCode.CORRUPT_PDF),
        (b"%PDF-1.4\n/Encrypt true\n%%EOF", ErrorCode.ENCRYPTED_PDF),
    ],
)
def test_rejects_bad_pdf(tmp_path: Path, content: bytes, code: ErrorCode) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(content)
    with pytest.raises(CoreError) as caught:
        validate_input(path)
    assert caught.value.issue.code == code


def test_rejects_corrupt_docx(tmp_path: Path) -> None:
    path = tmp_path / "bad.docx"
    path.write_bytes(b"not a zip")
    with pytest.raises(CoreError) as caught:
        validate_input(path)
    assert caught.value.issue.code == ErrorCode.CORRUPT_DOCX


@pytest.mark.parametrize("creator", [write_pdf, write_docx])
def test_accepts_supported_document(tmp_path: Path, creator: object) -> None:
    extension = ".pdf" if creator is write_pdf else ".docx"
    path = creator(tmp_path / f"cv{extension}")  # type: ignore[operator]
    result = validate_input(path)
    assert result.path == path.resolve()
    assert result.metadata.extension == extension
    assert len(result.metadata.sha256) == 64


def test_accepts_pdf_with_leading_transport_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "leading-whitespace.pdf"
    path.write_bytes(b"\n \t%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    assert validate_input(path).metadata.extension == ".pdf"
