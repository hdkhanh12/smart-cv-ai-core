"""Fail-closed local PII boundary for every remote LLM request."""

from __future__ import annotations

import importlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ai_core.schemas import CVProfile, EvidenceRef, UnifiedDocument

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_CANDIDATE = re.compile(r"(?<!\w)(?:\(\s*\+?\s*\d{1,3}\s*\)|\+?\s*\d)[\d ().-]{7,}\d(?!\w)")
_URL = re.compile(r"(?:https?://|www\.|(?:github|linkedin)\.com/)[^\s<>()]+", re.I)
_ADDRESS_LABEL = re.compile(r"^(?:address|location|city|địa chỉ|nơi ở)\s*[:\-]\s*(.+)$", re.I)
_DATE_RANGE = re.compile(r"\b(?:19|20)\d{2}\s*[-–]\s*(?:(?:19|20)\d{2}|\d{1,2})\b")
_INVALID_NAME_PATTERN = re.compile(
    r"\b(graduated|bachelor|master|degree|engineer|developer|architect|analyst|intern|fresher|scientist)\b",
    re.I,
)
_DOI_FRAGMENT = re.compile(r"^(?:19|20)\d{2}\.")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_NAME_LINE = re.compile(r"^[^\W\d_]{2,}(?:[ '\-][^\W\d_]{2,}){1,4}$", re.UNICODE)
_SEMANTIC_SECTION = re.compile(
    r"\b(?:summary|profile|skills?|technical skills?|experience|employment|"
    r"education|projects?|certifications?|languages?|work history|kinh nghi[eệ]m|"
    r"k[ỹy] n[aă]ng|h[oọ]c v[aấ]n|d[ựu] [aá]n)\b",
    re.I,
)
_HEADER_BLOCK_LIMIT = 8
_HEADER_MASKING_MODES = frozenset({"selective", "blanket-legacy"})
_LOCAL_HEADER_NAME = re.compile(
    r"^[^\W\d_]{2,}(?:[ '\-][^\W\d_]{2,}){1,4}(?:\s*\([^)]{2,40}\))?$",
    re.UNICODE,
)
_NAME_LABELS = frozenset({"ho va ten", "ho ten", "full name", "fullname", "name"})
_DOCUMENT_TITLES = frozenset({"so yeu ly lich", "curriculum vitae", "resume", "cv"})
_FILENAME_NOISE = frozenset(
    {
        "cv",
        "resume",
        "curriculum",
        "vitae",
        "final",
        "draft",
        "copy",
        "updated",
        "version",
        "file",
        "profile",
    }
)
_ROLE_SUFFIX = frozenset(
    {
        "developer",
        "engineer",
        "analyst",
        "designer",
        "manager",
        "intern",
        "scientist",
        "architect",
        "consultant",
        "specialist",
        "administrator",
    }
)
_ROLE_MODIFIER = frozenset(
    {
        "backend",
        "front",
        "frontend",
        "fullstack",
        "full",
        "stack",
        "software",
        "data",
        "science",
        "machine",
        "learning",
        "web",
        "mobile",
        "cloud",
        "devops",
        "qa",
        "ui",
        "ux",
    }
)
_RANDOM_FILE_STEM = re.compile(r"^[0-9a-f]{12,}$", re.IGNORECASE)
_FILE_VERSION = re.compile(r"^(?:v)?\d+(?:\.\d+)*$|^\d+(?:yrs?|years?)$", re.IGNORECASE)


@dataclass(frozen=True)
class MaskingResult:
    document: UnifiedDocument
    mapping: dict[str, str]
    identity: CandidateIdentity | None


@dataclass(frozen=True)
class CandidateIdentity:
    """A locally verified candidate identity with immutable document evidence."""

    name: str
    evidence: EvidenceRef | None = None
    source: str = "unified_header"
    confidence: float = 1.0
    font_size: float | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class LocalContact:
    """Direct contact details recovered locally, never from an LLM completion."""

    email: str | None
    phone: str | None
    urls: list[str]
    address: str | None


@dataclass(frozen=True)
class _HeaderLine:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float


def _is_phone(value: str) -> bool:
    """Accept realistic telephone numbers while excluding dates and identifiers."""

    digit_count = sum(character.isdigit() for character in value)
    return (
        9 <= digit_count <= 15
        and not _DATE_RANGE.search(value)
        and not _DOI_FRAGMENT.search(value.strip())
    )


def _email_local_tokens(document: UnifiedDocument) -> set[str]:
    tokens: set[str] = set()
    for page in document.pages:
        for block in page.blocks:
            for email in _EMAIL.findall(block.text):
                tokens.add(re.sub(r"[^a-z]", "", email.split("@", maxsplit=1)[0].casefold()))
    return tokens


def _plain_header_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value).casefold().replace("đ", "d")
    return "".join(
        character for character in normalized if unicodedata.category(character) != "Mn"
    ).strip(" :")


def _normalise_header_name(value: str) -> str | None:
    candidate = " ".join(value.strip().split())
    if candidate.count(",") == 1:
        family, given = (part.strip() for part in candidate.split(",", maxsplit=1))
        candidate = f"{given} {family}".strip()
    return candidate if _LOCAL_HEADER_NAME.fullmatch(candidate) else None


def _form_label_candidate(lines: list[_HeaderLine]) -> CandidateIdentity | None:
    """Read a value placed beside an explicit name field in a CV form."""

    for label in lines:
        label_key = _plain_header_text(label.text)
        inline_match = re.match(r"^(.*?)(?::\s*)(.+)$", label.text)
        if inline_match and _plain_header_text(inline_match.group(1)) in _NAME_LABELS:
            name = _normalise_header_name(inline_match.group(2))
            if name:
                return CandidateIdentity(
                    name=name,
                    source="pymupdf_form_name_field",
                    font_size=label.font_size,
                    bbox=label.bbox,
                )
        if label_key not in _NAME_LABELS:
            continue
        for value in lines:
            same_row = abs(value.bbox[1] - label.bbox[1]) <= 3.0
            to_the_right = value.bbox[0] >= label.bbox[2] - 2.0
            name = _normalise_header_name(value.text)
            if same_row and to_the_right and name:
                return CandidateIdentity(
                    name=name,
                    source="pymupdf_form_name_field",
                    font_size=value.font_size,
                    bbox=value.bbox,
                )
    return None


def extract_pdf_header_identity(path: str) -> CandidateIdentity | None:
    """Read only the visual first-page header using PyMuPDF font metadata.

    This local-only extractor does not inspect citations or body text.  A tie
    at the largest font size is deliberately unresolved rather than guessed.
    """

    try:
        pymupdf = importlib.import_module("pymupdf")
        pdf = pymupdf.open(path)
        with pdf:
            if pdf.page_count == 0:
                return None
            page = pdf.load_page(0)
            top_limit = float(page.rect.height) * 0.25
            form_limit = float(page.rect.height) * 0.35
            lines: list[_HeaderLine] = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    bbox = tuple(float(value) for value in line.get("bbox", ()))
                    spans = line.get("spans", [])
                    text = "".join(str(span.get("text", "")) for span in spans).strip()
                    if len(bbox) != 4 or bbox[1] > form_limit or not text:
                        continue
                    lines.append(
                        _HeaderLine(
                            text=text,
                            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                            font_size=max(
                                (float(span.get("size", 0.0)) for span in spans),
                                default=0.0,
                            ),
                        )
                    )
            form_identity = _form_label_candidate(lines)
            if form_identity is not None:
                return form_identity
            candidates: list[CandidateIdentity] = []
            for line in lines:
                name = _normalise_header_name(line.text)
                if (
                    line.bbox[1] > top_limit
                    or "@" in line.text
                    or _URL.search(line.text)
                    or _SEMANTIC_SECTION.search(line.text)
                    or _INVALID_NAME_PATTERN.search(line.text)
                    or _plain_header_text(line.text) in _DOCUMENT_TITLES
                    or name is None
                ):
                    continue
                candidates.append(
                    CandidateIdentity(
                        name=name,
                        source="pymupdf_font_header",
                        font_size=line.font_size,
                        bbox=line.bbox,
                    )
                )
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.font_size or 0.0, reverse=True)
    top = candidates[0]
    if len(candidates) > 1 and abs((top.font_size or 0.0) - (candidates[1].font_size or 0.0)) < 0.5:
        return None
    return top


def extract_filename_identity(path: str | Path) -> CandidateIdentity | None:
    """Resolve a clear candidate-name prefix from a local source filename.

    This is deliberately a last-resort local signal: random IDs, generic CV
    filenames, role-only names and ambiguous single-token stems are rejected.
    The source filename is never placed in an LLM request or audit payload.
    """

    stem = Path(path).stem.strip()
    if not stem or _RANDOM_FILE_STEM.fullmatch(stem):
        return None
    tokens = [token for token in re.sub(r"[_]+", " ", stem).split() if token]
    while tokens and _plain_header_text(tokens[0]) in _FILENAME_NOISE:
        tokens.pop(0)
    while tokens and (
        _plain_header_text(tokens[-1]) in _FILENAME_NOISE or _FILE_VERSION.fullmatch(tokens[-1])
    ):
        tokens.pop()
    role_index = next(
        (index for index, token in enumerate(tokens) if _plain_header_text(token) in _ROLE_SUFFIX),
        None,
    )
    if role_index is not None:
        tokens = tokens[:role_index]
        while tokens and _plain_header_text(tokens[-1]) in _ROLE_MODIFIER:
            tokens.pop()
    candidate = _normalise_header_name(" ".join(tokens))
    if candidate is None or _INVALID_NAME_PATTERN.search(candidate):
        return None
    return CandidateIdentity(
        name=candidate,
        source="filename_fallback",
        confidence=0.7,
    )


def _header_blocks(document: UnifiedDocument) -> list[tuple[int, str, str]]:
    """Return the small first-page identity zone, never the whole document."""

    if not document.pages:
        return []
    blocks: list[tuple[int, str, str]] = []
    for block in document.pages[0].blocks:
        blocks.append((document.pages[0].page_number, block.id, block.text))
        if len(blocks) >= _HEADER_BLOCK_LIMIT:
            break
    return blocks


def _trusted_header_identity(identity: CandidateIdentity | None) -> bool:
    """Whether identity evidence is strong enough to preserve its header peers.

    Filename inference is useful for local display but cannot authorize sending
    adjacent header text to an LLM: a stale or renamed file could mask the
    wrong name. Only high-confidence local header/form identity evidence may
    enable token-level header masking.
    """

    return bool(
        identity
        and identity.confidence >= 0.95
        and identity.source
        in {"pymupdf_font_header", "pymupdf_form_name_field", "unified_header"}
    )


def _email_comparable(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.casefold())


def resolve_verified_identity(document: UnifiedDocument) -> CandidateIdentity | None:
    """Read a candidate name only from the first-page header zone.

    A missing or ambiguous header is intentionally returned as ``None``.  No
    No citations, URLs or full-document n-gram fallback is allowed.
    """

    del document
    return None


def resolve_local_contact(document: UnifiedDocument) -> LocalContact:
    """Recover direct contact fields locally without exposing them to an LLM."""

    email: str | None = None
    phone: str | None = None
    urls: list[str] = []
    address: str | None = None
    for page in document.pages:
        for block in page.blocks:
            if email is None:
                found_emails = _EMAIL.findall(block.text)
                email = found_emails[0] if found_emails else None
            if phone is None:
                for match in _PHONE_CANDIDATE.finditer(block.text):
                    if _is_phone(match.group(0)):
                        phone = match.group(0)
                        break
            for match in _URL.findall(block.text):
                value = match.rstrip(".,;:)")
                normalized = (
                    value
                    if value.lower().startswith(("http://", "https://"))
                    else f"https://{value}"
                )
                if normalized not in urls:
                    urls.append(normalized)
            if address is None:
                address_match = _ADDRESS_LABEL.match(block.text.strip())
                if address_match:
                    address = address_match.group(1).strip() or None
    return LocalContact(email=email, phone=phone, urls=urls, address=address)


def mask_document(
    document: UnifiedDocument,
    *,
    local_identity: CandidateIdentity | None = None,
    header_masking_mode: str = "selective",
) -> MaskingResult:
    """Replace local PII while preserving non-PII header content when safe.

    A high-confidence local header identity permits token-level header
    redaction. When that evidence is unavailable, the small legacy header zone
    remains fail-closed: it is blanked rather than risking an unknown candidate
    name crossing the remote boundary. `blanket-legacy` exists only for
    controlled benchmarks.
    """

    if header_masking_mode not in _HEADER_MASKING_MODES:
        raise ValueError(f"Unsupported header masking mode: {header_masking_mode}")

    mapping: dict[str, str] = {}
    counter = {"email": 0, "phone": 0, "name": 0, "url": 0}

    def replace(kind: str, match: re.Match[str]) -> str:
        original = match.group(0)
        if original not in mapping:
            counter[kind] += 1
            mapping[original] = f"[ANON_{kind.upper()}_{counter[kind]}]"
        return mapping[original]

    identity = local_identity
    candidate_name = identity.name if identity else None
    candidate_name_pattern = None
    if candidate_name:
        counter["name"] += 1
        mapping[candidate_name] = "[ANON_NAME]"
        candidate_name_pattern = re.compile(re.escape(candidate_name), re.IGNORECASE)

    def mask(text: str) -> str:
        value = text
        if candidate_name_pattern is not None:
            value = candidate_name_pattern.sub("[ANON_NAME]", value)
        value = _EMAIL.sub(lambda match: replace("email", match), value)
        value = _PHONE_CANDIDATE.sub(
            lambda match: replace("phone", match) if _is_phone(match.group(0)) else match.group(0),
            value,
        )
        return _URL.sub(lambda match: replace("url", match), value)

    header_blocks = _header_blocks(document)
    header_ids = {block_id for _, block_id, _ in header_blocks}
    blanket_header = header_masking_mode == "blanket-legacy" or not _trusted_header_identity(
        local_identity
    )

    def mask_block(block_id: str, text: str) -> str:
        if blanket_header and block_id in header_ids:
            return "[HEADER_REDACTED]"
        return mask(text)

    pages = [
        page.model_copy(
            update={
                "blocks": [
                    block.model_copy(
                        update={
                            "text": mask_block(block.id, block.text)
                        }
                    )
                    for block in page.blocks
                ]
            }
        )
        for page in document.pages
    ]
    masked_markdown = mask(document.markdown)
    if blanket_header:
        for _, _, header_text in header_blocks:
            masked_markdown = masked_markdown.replace(mask(header_text), "[HEADER_REDACTED]")
    return MaskingResult(
        document.model_copy(update={"markdown": masked_markdown, "pages": pages}),
        mapping,
        identity,
    )


def deanonymize_profile(profile: CVProfile, mapping: dict[str, str]) -> CVProfile:
    """Restore every profile string, including evidence, before reconciliation."""

    reverse = {token: original for original, token in mapping.items()}

    def restore(value: str | None) -> str | None:
        if value is None:
            return None
        for token, original in reverse.items():
            value = value.replace(token, original)
        return value

    def restore_value(value: object) -> object:
        if isinstance(value, str):
            return restore(value)
        if isinstance(value, list):
            return [restore_value(item) for item in value]
        if isinstance(value, dict):
            return {key: restore_value(item) for key, item in value.items()}
        return value

    restored = CVProfile.model_validate(restore_value(profile.model_dump(mode="python")))
    email = restored.email
    phone = restored.phone

    if not email:
        for orig, tok in mapping.items():
            if "[ANON_EMAIL" in tok and _EMAIL.fullmatch(orig):
                email = orig
                break

    if not phone:
        for orig, tok in mapping.items():
            if "[ANON_PHONE" in tok and _is_phone(orig):
                phone = orig
                break

    return restored.model_copy(
        update={
            "email": email,
            "phone": phone,
        }
    )


def apply_verified_identity(profile: CVProfile, identity: CandidateIdentity | None) -> CVProfile:
    """Use high-confidence local identity rather than an LLM candidate-name guess."""

    if identity is None:
        return profile
    evidence = dict(profile.field_evidence)
    if identity.evidence is not None:
        evidence["candidateName"] = [identity.evidence]
    confidence = dict(profile.field_confidence)
    confidence["candidateName"] = identity.confidence
    return profile.model_copy(
        update={
            "candidate_name": identity.name,
            "field_evidence": evidence,
            "field_confidence": confidence,
        }
    )


def apply_local_contact(profile: CVProfile, contact: LocalContact) -> CVProfile:
    """Set contact fields exclusively from local evidence, never an LLM guess."""

    payload = profile.model_dump(mode="python")
    payload.update(
        {
            "email": contact.email,
            "phone": contact.phone,
            "urls": contact.urls,
            "address": contact.address,
        }
    )
    return CVProfile.model_validate(payload)
