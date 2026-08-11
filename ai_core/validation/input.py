"""Input-file validation before a parser is selected."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ai_core.errors import CoreError, ErrorCode
from ai_core.schemas import DocumentMetadata

DEFAULT_MAX_FILE_SIZE = 20 * 1024 * 1024
SUPPORTED_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@dataclass(frozen=True, slots=True)
class ValidatedInput:
    path: Path
    metadata: DocumentMetadata


def _raise(code: ErrorCode, message: str, path: Path, **details: object) -> None:
    raise CoreError(
        code,
        message,
        stage="input_validation",
        details={"sourceName": path.name, **details},
    )


def _validate_pdf(path: Path) -> None:
    with path.open("rb") as stream:
        head = stream.read(1024)
        stream.seek(max(0, path.stat().st_size - 4096))
        tail = stream.read()
    # ISO 32000 readers tolerate leading whitespace/transport bytes before the
    # header, and real-world generators occasionally emit them.
    if b"%PDF-" not in head or b"%%EOF" not in tail:
        _raise(ErrorCode.CORRUPT_PDF, "PDF header or end marker is invalid.", path)
    sample = head + tail
    if b"/Encrypt" in sample:
        _raise(ErrorCode.ENCRYPTED_PDF, "Encrypted PDF files are not supported.", path)


def _validate_docx(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {"[Content_Types].xml", "word/document.xml"}
            if not required.issubset(names) or archive.testzip() is not None:
                _raise(ErrorCode.CORRUPT_DOCX, "DOCX package is incomplete or corrupt.", path)
    except zipfile.BadZipFile as exc:
        raise CoreError(
            ErrorCode.CORRUPT_DOCX,
            "DOCX package is not a valid ZIP archive.",
            stage="input_validation",
            details={"sourceName": path.name},
        ) from exc


def validate_input(
    value: str | Path,
    *,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
) -> ValidatedInput:
    """Validate a local PDF/DOCX and return safe, serializable metadata."""

    path = Path(value)
    if not path.exists():
        _raise(ErrorCode.FILE_NOT_FOUND, "Input file does not exist.", path)
    if not path.is_file():
        _raise(ErrorCode.NOT_A_FILE, "Input path is not a regular file.", path)

    extension = path.suffix.lower()
    if extension not in SUPPORTED_MEDIA_TYPES:
        _raise(
            ErrorCode.UNSUPPORTED_EXTENSION,
            "Only PDF and DOCX files are supported.",
            path,
            extension=extension,
        )

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        _raise(ErrorCode.EMPTY_FILE, "Input file is empty.", path)
    if size_bytes > max_size_bytes:
        _raise(
            ErrorCode.FILE_TOO_LARGE,
            f"Input file exceeds the {max_size_bytes}-byte limit.",
            path,
            sizeBytes=size_bytes,
            maxSizeBytes=max_size_bytes,
        )

    if extension == ".pdf":
        _validate_pdf(path)
    else:
        _validate_docx(path)

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    metadata = DocumentMetadata(
        source_name=path.name,
        extension=extension,
        media_type=SUPPORTED_MEDIA_TYPES[extension],
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )
    return ValidatedInput(path=path.resolve(), metadata=metadata)
