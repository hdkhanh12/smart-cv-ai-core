from __future__ import annotations

from pathlib import Path

from ai_core.extraction.privacy import (
    CandidateIdentity,
    apply_verified_identity,
    deanonymize_profile,
    extract_filename_identity,
    extract_pdf_header_identity,
    mask_document,
    resolve_local_contact,
    resolve_verified_identity,
)
from ai_core.extraction.provider import build_request
from ai_core.reconciliation import reconcile_profile
from ai_core.schemas import (
    CVProfile,
    DiagnosticStatus,
    EvidenceRef,
    ExtractionDiagnostics,
    UnifiedBlock,
    UnifiedDocument,
    UnifiedPage,
)


def _document() -> UnifiedDocument:
    blocks = [
        "Australia Sydney",
        "+61 414 538 472",
        "tungduynguyen811@gmail.com",
        "Nguyen Duy Tung",
        "2024 - 2026",
    ]
    return UnifiedDocument(
        file_hash="a" * 64,
        extractor="fixture",
        markdown="\n".join(blocks),
        pages=[
            UnifiedPage(
                page_number=1,
                blocks=[
                    UnifiedBlock(id=f"p1-b{index}", text=text, reading_order=index)
                    for index, text in enumerate(blocks, start=1)
                ],
            )
        ],
        diagnostics=ExtractionDiagnostics(
            status=DiagnosticStatus.USABLE,
            text_character_count=100,
            block_count=len(blocks),
            heading_count=0,
        ),
    )


def test_masking_redacts_header_and_never_masks_date_ranges() -> None:
    result = mask_document(_document())

    assert result.mapping["+61 414 538 472"] == "[ANON_PHONE_1]"
    assert "2024 - 2026" not in result.mapping
    assert result.document.pages[0].blocks[0].text == "[HEADER_REDACTED]"


def test_selective_header_masking_preserves_company_role_and_dates_with_known_identity() -> None:
    document = UnifiedDocument(
        file_hash="b" * 64,
        extractor="fixture",
        markdown=(
            "Nguyen Duy Tung\ntungduynguyen811@gmail.com\n+61 414 538 472\n"
            "Atrae, Inc - Japan\nSenior Data Scientist - AI Engineer\nNovember 2024 - Now"
        ),
        pages=[
            UnifiedPage(
                page_number=1,
                blocks=[
                    UnifiedBlock(id="p1-b1", text="Nguyen Duy Tung", reading_order=1),
                    UnifiedBlock(
                        id="p1-b2", text="tungduynguyen811@gmail.com", reading_order=2
                    ),
                    UnifiedBlock(id="p1-b3", text="+61 414 538 472", reading_order=3),
                    UnifiedBlock(id="p1-b4", text="Atrae, Inc - Japan", reading_order=4),
                    UnifiedBlock(
                        id="p1-b5", text="Senior Data Scientist - AI Engineer", reading_order=5
                    ),
                    UnifiedBlock(id="p1-b6", text="November 2024 - Now", reading_order=6),
                ],
            )
        ],
        diagnostics=ExtractionDiagnostics(
            status=DiagnosticStatus.USABLE,
            text_character_count=100,
            block_count=6,
            heading_count=0,
        ),
    )

    result = mask_document(
        document,
        local_identity=CandidateIdentity(name="Nguyen Duy Tung"),
    )
    request_json = build_request(
        result.document,
        local_identity=CandidateIdentity(name="Nguyen Duy Tung"),
    ).document.model_dump_json(by_alias=True)

    assert [block.text for block in result.document.pages[0].blocks] == [
        "[ANON_NAME]",
        "[ANON_EMAIL_1]",
        "[ANON_PHONE_1]",
        "Atrae, Inc - Japan",
        "Senior Data Scientist - AI Engineer",
        "November 2024 - Now",
    ]
    for direct_identifier in (
        "Nguyen Duy Tung",
        "tungduynguyen811@gmail.com",
        "+61 414 538 472",
    ):
        assert direct_identifier not in request_json
    assert "Atrae, Inc - Japan" in request_json
    assert "Senior Data Scientist - AI Engineer" in request_json


def test_legacy_header_masking_remains_available_for_controlled_benchmark() -> None:
    result = mask_document(
        _document(),
        local_identity=CandidateIdentity(name="Nguyen Duy Tung"),
        header_masking_mode="blanket-legacy",
    )

    assert result.document.pages[0].blocks[0].text == "[HEADER_REDACTED]"


def test_filename_identity_keeps_header_masking_fail_closed() -> None:
    result = mask_document(
        _document(),
        local_identity=CandidateIdentity(
            name="Nguyen Duy Tung",
            source="filename_fallback",
            confidence=0.7,
        ),
    )

    assert result.document.pages[0].blocks[0].text == "[HEADER_REDACTED]"


def test_local_contact_recovers_urls_and_explicit_address_without_llm() -> None:
    document = _document().model_copy(
        update={
            "pages": [
                UnifiedPage(
                    page_number=1,
                    blocks=[
                        UnifiedBlock(id="p1-b1", text="Address: Ho Chi Minh City", reading_order=1),
                        UnifiedBlock(
                            id="p1-b2",
                            text="Portfolio: github.com/synthetic-candidate",
                            reading_order=2,
                        ),
                    ],
                )
            ]
        }
    )

    contact = resolve_local_contact(document)

    assert contact.address == "Ho Chi Minh City"
    assert contact.urls == ["https://github.com/synthetic-candidate"]


def test_deanonymization_restores_evidence_before_reconciliation() -> None:
    document = _document()
    masked_profile = CVProfile(
        email="[ANON_EMAIL_1]",
        phone="[ANON_PHONE_1]",
    )

    restored = deanonymize_profile(
        masked_profile,
        {
            "tungduynguyen811@gmail.com": "[ANON_EMAIL_1]",
            "+61 414 538 472": "[ANON_PHONE_1]",
        },
    )
    reconciled = reconcile_profile(
        restored,
        document,
        provider_name="beeknoee-openai-compatible",
        allow_identity_fallback=False,
    )

    assert restored.email == "tungduynguyen811@gmail.com"
    assert restored.phone == "+61 414 538 472"
    assert reconciled.candidate_name is None


def test_local_font_identity_overrides_an_llm_location_guess() -> None:
    identity = CandidateIdentity(name="Nguyen Duy Tung", source="pymupdf_font_header")
    profile = CVProfile(candidate_name="Australia Sydney")

    result = apply_verified_identity(profile, identity)

    assert result.candidate_name == "Nguyen Duy Tung"
    assert result.field_confidence["candidateName"] == 1.0


def test_identity_resolver_does_not_scan_a_long_citation_block() -> None:
    document = _document().model_copy(
        update={
            "pages": [
                UnifiedPage(
                    page_number=1,
                    blocks=[
                        UnifiedBlock(
                            id="p1-b1",
                            text="tungduynguyen811@gmail.com",
                            reading_order=1,
                        ),
                        UnifiedBlock(
                            id="p1-b21",
                            text=(
                                "Bui Duc Trung, Ngo Tung Son, Nguyen Duy Tung, Kieu Anh Son, "
                                "and Phan Truong Lam. Educational Data Mining."
                            ),
                            reading_order=2,
                        ),
                    ],
                )
            ]
        }
    )

    identity = resolve_verified_identity(document)

    assert identity is None


def test_request_boundary_never_contains_local_direct_identifiers() -> None:
    document = _document().model_copy(
        update={
            "markdown": (
                "Nguyen Duy Tung\n+61 414 538 472\ntungduynguyen811@gmail.com\n"
                "https://www.linkedin.com/in/tungduynguyen"
            )
        }
    )

    request_json = build_request(document).document.model_dump_json(by_alias=True)

    for direct_identifier in (
        "Nguyen Duy Tung",
        "+61 414 538 472",
        "tungduynguyen811@gmail.com",
        "linkedin.com/in/tungduynguyen",
    ):
        assert direct_identifier not in request_json
    assert "[HEADER_REDACTED]" in request_json


def test_header_zone_is_redacted_even_when_no_name_is_confirmed() -> None:
    document = _document().model_copy(
        update={
            "pages": [
                UnifiedPage(
                    page_number=1,
                    blocks=[
                        UnifiedBlock(id="p1-b1", text="Australia Sydney", reading_order=1),
                        UnifiedBlock(
                            id="p1-b2",
                            text="tungduynguyen811@gmail.com",
                            reading_order=2,
                        ),
                        UnifiedBlock(id="p1-b3", text="SKILLS", reading_order=3),
                        UnifiedBlock(id="p1-b4", text="Python", reading_order=4),
                    ],
                )
            ],
            "markdown": "Australia Sydney\ntungduynguyen811@gmail.com\nSKILLS\nPython",
        }
    )

    result = mask_document(document)

    assert result.identity is None
    assert [block.text for block in result.document.pages[0].blocks] == [
        "[HEADER_REDACTED]",
        "[HEADER_REDACTED]",
        "[HEADER_REDACTED]",
        "[HEADER_REDACTED]",
    ]
    assert "tungduynguyen811@gmail.com" not in result.document.markdown


def test_pymupdf_header_identity_uses_largest_font_in_top_zone(tmp_path: Path) -> None:
    import importlib

    pymupdf = importlib.import_module("pymupdf")
    path = tmp_path / "header.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((60, 55), "DUY TUNG NGUYEN", fontsize=30)
    page.insert_text((360, 80), "Sydney Australia", fontsize=12)
    page.insert_text((360, 100), "tungduynguyen811@gmail.com", fontsize=12)
    pdf.save(path)
    pdf.close()

    identity = extract_pdf_header_identity(str(path))

    assert identity is not None
    assert identity.name == "DUY TUNG NGUYEN"
    assert identity.source == "pymupdf_font_header"
    assert identity.font_size == 30


def test_pymupdf_header_identity_normalises_comma_name_order(tmp_path: Path) -> None:
    import importlib

    pymupdf = importlib.import_module("pymupdf")
    path = tmp_path / "comma-name.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((180, 55), "Ngoc, Hong Nhu", fontsize=27)
    page.insert_text((230, 85), "Data Engineer", fontsize=16)
    pdf.save(path)
    pdf.close()

    identity = extract_pdf_header_identity(str(path))

    assert identity is not None
    assert identity.name == "Hong Nhu Ngoc"
    assert identity.source == "pymupdf_font_header"


def test_pymupdf_header_identity_prioritizes_explicit_fullname_field(tmp_path: Path) -> None:
    import importlib

    pymupdf = importlib.import_module("pymupdf")
    path = tmp_path / "form-name.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((200, 70), "CURRICULUM VITAE", fontsize=22)
    page.insert_text((60, 235), "Fullname:", fontsize=12)
    page.insert_text((180, 235), "Mai Duy Thang", fontsize=12)
    pdf.save(path)
    pdf.close()

    identity = extract_pdf_header_identity(str(path))

    assert identity is not None
    assert identity.name == "Mai Duy Thang"
    assert identity.source == "pymupdf_form_name_field"


def test_filename_identity_uses_clear_name_prefix_before_role_suffix() -> None:
    identity = extract_filename_identity("NGUYEN_HUU_NHAT_MINH_Backend_Developer_5yrs.pdf")

    assert identity is not None
    assert identity.name == "NGUYEN HUU NHAT MINH"
    assert identity.source == "filename_fallback"
    assert identity.confidence == 0.7


def test_filename_identity_rejects_generic_random_and_role_only_names() -> None:
    assert extract_filename_identity("019ab4fe38644cf7.pdf") is None
    assert extract_filename_identity("CV_final.pdf") is None
    assert extract_filename_identity("Data_Science_Intern.pdf") is None


def test_filename_identity_preserves_name_casing_and_masks_ocr_case_variants() -> None:
    identity = extract_filename_identity("Nguyen_Huu_Nhat_Minh_Backend_Developer_5yrs.pdf")
    assert identity is not None
    document = _document().model_copy(
        update={"markdown": "NGUYEN HUU NHAT MINH\nPython", "pages": _document().pages}
    )

    masked = mask_document(document, local_identity=identity)

    assert "NGUYEN HUU NHAT MINH" not in masked.document.markdown
    assert "[ANON_NAME]" in masked.document.markdown


def test_identity_resolver_rejects_email_url_and_single_letter_ngrams() -> None:
    document = _document().model_copy(
        update={
            "pages": [
                UnifiedPage(
                    page_number=1,
                    blocks=[
                        UnifiedBlock(id="p1-b1", text="n tu", reading_order=1),
                        UnifiedBlock(
                            id="p1-b2",
                            text="tuan25092003@gmail.com",
                            reading_order=2,
                        ),
                        UnifiedBlock(
                            id="p1-b3",
                            text="https://example.test/in/tuan-nguyen",
                            reading_order=3,
                        ),
                    ],
                )
            ]
        }
    )

    assert resolve_verified_identity(document) is None


def test_deanonymization_restores_placeholder_url_in_evidence() -> None:
    profile = CVProfile(
        field_evidence={
            "p1-b10": [EvidenceRef(text="[ANON_URL_1]", page_number=1, block_id="p1-b10")]
        }
    )

    restored = deanonymize_profile(
        profile,
        {"https://github.com/example/project": "[ANON_URL_1]"},
    )

    assert restored.field_evidence["p1-b10"][0].text == "https://github.com/example/project"
