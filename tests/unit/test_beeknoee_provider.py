from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from ai_core.errors import CoreError, ErrorCode
from ai_core.extraction.beeknoee import (
    MISSING_API_KEY_MESSAGE,
    BeeknoeeSettings,
    BeeknoeeStructuredExtractionProvider,
    load_beeknoee_settings,
)
from ai_core.extraction.dto import LLMExtractedProfile, normalize_llm_profile_payload
from ai_core.extraction.provider import build_request
from ai_core.parsers.unified import unified_from_lines
from ai_core.validation import validate_input


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self.responses:
            response = self.responses.pop(0)
            self._last_response = response
        else:
            response = getattr(self, "_last_response", FakeResponse(500, {}))
        if isinstance(response, Exception):
            raise response
        return response


def _request(tmp_path: Path):
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    document = unified_from_lines(
        validate_input(source),
        "fixture",
        [["ALEX MORGAN", "Sydney, Australia", "PERSONAL PROJECTS", "Parser toolkit"]],
    )
    return build_request(document)


def _completion(profile: dict[str, Any] | None = None) -> FakeResponse:
    return FakeResponse(
        200,
        {
            "choices": [{"message": {"content": json.dumps(profile or {})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


def _provider(
    session: FakeSession,
    **kwargs: object,
) -> BeeknoeeStructuredExtractionProvider:
    return BeeknoeeStructuredExtractionProvider(
        settings=BeeknoeeSettings(
            base_url="https://platform.beeknoee.com/v1/",
            api_key="fixture-value",
            model="gemini-2.5-flash-lite",
        ),
        session=session,  # type: ignore[arg-type]
        **kwargs,
    )


def test_required_initial_evidence_contract_relaxes_for_correction(tmp_path: Path) -> None:
    provider = _provider(
        FakeSession([]),
        evidence_contract="required-initial",
        output_scope="entities-only",
    )
    request = _request(tmp_path)
    response_format: dict[str, object] = {
        "type": "json_schema",
        "json_schema": {"name": "fixture", "strict": True, "schema": request.json_schema},
    }

    initial = provider._payload(request, response_format)
    initial_schema = initial["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    initial_experience = initial_schema["$defs"]["LLMExtractedExperience"]  # type: ignore[index]
    assert "evidence" in initial_experience["required"]
    assert initial_experience["properties"]["evidence"]["minItems"] == 1

    correction_response = {"choices": [{"message": {"content": "{}"}}]}
    correction = provider._correction_payload(
        initial,
        correction_response,
        ["experiences.0"],
        request,
    )
    correction_schema = correction["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    correction_experience = correction_schema["$defs"]["LLMExtractedExperience"]  # type: ignore[index]
    assert "evidence" not in correction_experience.get("required", [])


def test_loads_fake_dotenv_without_exposing_secret(
    tmp_path: Path,
) -> None:
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "BEE_API_KEY=fixture-value\nBEE_BASE_URL=https://example.test/v1\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    names = ("BEE_API_KEY", "BEE_BASE_URL", "LLM_MODEL")
    original = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        settings = load_beeknoee_settings(dotenv_path=dotenv_file, override=True)

        assert settings.base_url == "https://example.test/v1"
        assert settings.model == "test-model"
        assert "fixture-value" not in repr(settings)
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_missing_api_key_fails_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEE_API_KEY", raising=False)
    with pytest.raises(CoreError) as error:
        load_beeknoee_settings(dotenv_path=Path("does-not-exist"))

    assert error.value.issue.code == ErrorCode.LLM_CONFIGURATION_ERROR
    assert error.value.issue.message == MISSING_API_KEY_MESSAGE


def test_posts_one_full_document_with_schema_and_parses_response(tmp_path: Path) -> None:
    session = FakeSession(
        [
            _completion(
                {
                    "candidateName": None,
                    "skills": [
                        {
                            "name": "Python",
                            "evidence": [
                                {
                                    "text": "[HEADER_REDACTED]",
                                    "pageNumber": 1,
                                    "blockId": "p1-b1",
                                }
                            ],
                        }
                    ],
                    "evidence": {"candidateName": []},
                }
            )
        ]
    )
    provider = _provider(session)
    request = _request(tmp_path)

    profile = provider.extract(request)

    assert profile.candidate_name is None
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://platform.beeknoee.com/v1/chat/completions"
    assert call["json"]["model"] == "gemini-2.5-flash-lite"
    assert call["json"]["temperature"] == 0.1
    assert call["json"]["top_p"] == 0.95
    assert call["json"]["max_tokens"] == 3000
    assert call["json"]["response_format"]["type"] == "json_schema"
    assert call["json"]["response_format"]["json_schema"]["name"] == "llm_extracted_profile"
    assert "ALEX MORGAN" not in call["json"]["messages"][1]["content"]
    assert "[HEADER_REDACTED]" in call["json"]["messages"][1]["content"]
    assert "p1-b1" in call["json"]["messages"][1]["content"]
    assert "LAYOUT TRANSCRIPT" in call["json"]["messages"][1]["content"]
    assert "PAGE 1 — SINGLE-COLUMN/UNPOSITIONED" in call["json"]["messages"][1]["content"]
    assert "UnifiedDocument (complete" not in call["json"]["messages"][1]["content"]
    assert "personalProjects" in call["json"]["messages"][1]["content"]
    assert "Northstar" not in call["json"]["messages"][1]["content"]
    assert "Berlin, Germany" not in call["json"]["messages"][1]["content"]
    assert profile.skills[0].canonical_name == "Python"
    assert "fixture-value" not in repr(call["json"])
    funnel = provider.last_audit["funnel"]
    assert isinstance(funnel, dict)
    initial = funnel["initial"]
    assert isinstance(initial, dict)
    recognized = initial["llmRecognized"]
    assert isinstance(recognized, dict)
    assert recognized["experiences"]["entities"] == 0
    assert "ALEX MORGAN" not in repr(funnel)


def test_falls_back_to_json_object_when_json_schema_is_unsupported(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(400, {"error": "unsupported"}, "response_format json_schema unsupported"),
            _completion({"candidateName": "ALEX MORGAN"}),
        ]
    )

    profile = _provider(session).extract(_request(tmp_path))

    assert profile.candidate_name == "ALEX MORGAN"
    assert [call["json"].get("response_format", {}).get("type") for call in session.calls] == [
        "json_schema",
        "json_object",
    ]


def test_layout_native_is_an_explicit_experiment_representation(tmp_path: Path) -> None:
    session = FakeSession([_completion({"candidateName": "ALEX MORGAN"})])
    provider = BeeknoeeStructuredExtractionProvider(
        settings=BeeknoeeSettings(
            base_url="https://platform.beeknoee.com/v1",
            api_key="fixture-value",
            model="gemini-2.5-flash-lite",
        ),
        session=session,  # type: ignore[arg-type]
        input_representation="layout-native",
    )

    provider.extract(_request(tmp_path))

    user_message = session.calls[0]["json"]["messages"][1]["content"]
    assert "Layout-native UnifiedDocument JSON" in user_message
    assert "LAYOUT TRANSCRIPT" not in user_message
    assert provider.last_audit["inputRepresentation"] == "layout-native"


def test_canonical_text_is_an_explicit_layout_free_experiment_representation(
    tmp_path: Path,
) -> None:
    session = FakeSession([_completion({"candidateName": "ALEX MORGAN"})])
    provider = BeeknoeeStructuredExtractionProvider(
        settings=BeeknoeeSettings(
            base_url="https://platform.beeknoee.com/v1",
            api_key="fixture-value",
            model="gemini-2.5-flash-lite",
        ),
        session=session,  # type: ignore[arg-type]
        input_representation="canonical-text",
    )

    provider.extract(_request(tmp_path))

    user_message = session.calls[0]["json"]["messages"][1]["content"]
    assert "Canonical text in source order" in user_message
    assert "[1|p1-b1]" in user_message
    assert "LAYOUT TRANSCRIPT" not in user_message
    assert "Layout-native UnifiedDocument JSON" not in user_message
    assert provider.last_audit["inputRepresentation"] == "canonical-text"


def test_entity_bound_evidence_mode_requires_entity_local_citations(tmp_path: Path) -> None:
    session = FakeSession([_completion({"candidateName": "ALEX MORGAN"})])
    provider = BeeknoeeStructuredExtractionProvider(
        settings=BeeknoeeSettings(
            base_url="https://platform.beeknoee.com/v1",
            api_key="fixture-value",
            model="gemini-2.5-flash-lite",
        ),
        session=session,  # type: ignore[arg-type]
        evidence_mode="entity-bound",
    )

    provider.extract(_request(tmp_path))

    system_message = session.calls[0]["json"]["messages"][0]["content"]
    assert "entity-local evidence object" in system_message
    assert provider.last_audit["evidenceMode"] == "entity-bound"


def test_entities_only_scope_restricts_schema_and_prompt(tmp_path: Path) -> None:
    session = FakeSession([_completion({"experiences": [], "projects": [], "evidence": {}})])
    provider = BeeknoeeStructuredExtractionProvider(
        settings=BeeknoeeSettings(
            base_url="https://platform.beeknoee.com/v1",
            api_key="fixture-value",
            model="gemini-2.5-flash-lite",
        ),
        session=session,  # type: ignore[arg-type]
        output_scope="entities-only",
    )

    provider.extract(_request(tmp_path))

    call = session.calls[0]["json"]
    user_message = call["messages"][1]["content"]
    assert '"experiences":[],"projects":[],"evidence":{}' in user_message
    assert '"candidateName":null' not in user_message
    properties = call["response_format"]["json_schema"]["schema"]["properties"]
    assert set(properties) == {"experiences", "projects", "evidence"}
    assert provider.last_audit["outputScope"] == "entities-only"


def test_entity_inventory_scope_restricts_to_identity_and_dates(tmp_path: Path) -> None:
    session = FakeSession([_completion({"experiences": [], "projects": []})])
    provider = BeeknoeeStructuredExtractionProvider(
        settings=BeeknoeeSettings(
            base_url="https://platform.beeknoee.com/v1",
            api_key="fixture-value",
            model="gemini-2.5-flash-lite",
        ),
        session=session,  # type: ignore[arg-type]
        output_scope="entity-inventory",
    )

    provider.extract(_request(tmp_path))

    call = session.calls[0]["json"]
    system_message = call["messages"][0]["content"]
    user_message = call["messages"][1]["content"]
    assert "identity inventory" in user_message
    assert "entity-local evidence" not in system_message
    assert '"experiences":[],"projects":[]' in user_message
    properties = call["response_format"]["json_schema"]["schema"]["properties"]
    assert set(properties) == {"experiences", "projects"}
    definitions = call["response_format"]["json_schema"]["schema"]["$defs"]
    assert set(definitions["LLMExtractedExperience"]["properties"]) == {
        "jobTitle",
        "company",
        "startDate",
        "endDate",
        "isCurrent",
    }
    assert set(definitions["LLMExtractedProject"]["properties"]) == {
        "title",
        "role",
        "startDate",
        "endDate",
        "isCurrent",
    }
    assert provider.last_audit["outputScope"] == "entity-inventory"


def test_retries_invalid_json_before_returning_valid_profile(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"choices": [{"message": {"content": "not json"}}]}),
            _completion({"candidateName": "ALEX MORGAN"}),
        ]
    )

    profile = _provider(session).extract(_request(tmp_path))

    assert profile.candidate_name == "ALEX MORGAN"
    assert len(session.calls) == 2
    correction = session.calls[1]["json"]["messages"][-1]["content"]
    assert correction == (
        "Correct the previous JSON only. Validation errors by field path: $: valid JSON required. "
        "Preserve every existing experiences, projects, and education entry and its semantic "
        "content. Do not delete entities, reduce collection length, or replace a populated "
        "collection with [] to fix validation. Correct only invalid fields; use null or [] for "
        "an invalid optional value. Return JSON only."
    )
    assert "ALEX MORGAN" not in correction


def test_does_not_map_legacy_semantic_aliases_and_corrects_once(tmp_path: Path) -> None:
    session = FakeSession(
        [
            _completion({"name": "ALEX MORGAN", "workHistory": []}),
            _completion({"candidateName": "ALEX MORGAN"}),
        ]
    )

    profile = _provider(session).extract(_request(tmp_path))

    assert profile.candidate_name == "ALEX MORGAN"
    assert len(session.calls) == 2
    correction = session.calls[1]["json"]["messages"][-1]["content"]
    assert "name" in correction
    assert "workHistory" in correction


def test_fails_transparently_when_correction_removes_recognized_entity(tmp_path: Path) -> None:
    session = FakeSession(
        [
            _completion({"experiences": [{"jobTitle": "Data Engineer", "startDate": "bad"}]}),
            _completion({"experiences": []}),
        ]
    )

    with pytest.raises(CoreError) as error:
        _provider(session).extract(_request(tmp_path))

    assert error.value.issue.code == ErrorCode.LLM_SEMANTIC_REGRESSION
    details = error.value.issue.details
    assert details["lostEntityCounts"]["experiences"] == 1
    assert "Data Engineer" not in repr(details)


def test_opt_in_raw_capture_has_expiry_and_only_whitelisted_headers(tmp_path: Path) -> None:
    session = FakeSession([_completion({"candidateName": "ALEX MORGAN"})])
    provider = BeeknoeeStructuredExtractionProvider(
        settings=BeeknoeeSettings(
            base_url="https://platform.beeknoee.com/v1",
            api_key="fixture-value",
            model="gemini-2.5-flash-lite",
        ),
        session=session,  # type: ignore[arg-type]
        raw_capture_dir=tmp_path / "raw-capture",
    )

    provider.extract(_request(tmp_path))

    captures = list((tmp_path / "raw-capture").glob("*.raw-capture.json"))
    assert len(captures) == 1
    capture = json.loads(captures[0].read_text(encoding="utf-8"))
    assert capture["kind"] == "smart_cv_raw_capture"
    assert capture["stage"] == "initial"
    assert capture["expiresAt"] > capture["capturedAt"]
    assert capture["request"]["messages"]
    assert capture["responseHeaders"] == {}


def test_normalizes_harmless_project_shapes_before_strict_dto_validation() -> None:
    raw = {
        "projects": [
            {
                "name": "CV Parser Toolkit",
                "description": "Built a local parser for structured CV data.",
                "skills": "Python",
                "achievements": "Validated JSON output",
            }
        ]
    }

    with pytest.raises(ValidationError):
        LLMExtractedProfile.model_validate(raw)
    normalized = LLMExtractedProfile.model_validate(normalize_llm_profile_payload(raw))

    assert normalized.projects[0].title == "CV Parser Toolkit"
    assert normalized.projects[0].description == ["Built a local parser for structured CV data."]
    assert normalized.projects[0].skills == ["Python"]
    assert normalized.projects[0].achievements == ["Validated JSON output"]


def test_normalizes_observed_technologies_and_plain_skills() -> None:
    raw = {
        "skills": ["Python"],
        "projects": [{"title": "Synthetic Project", "technologies": ["Python", "SQL"]}],
    }

    normalized = LLMExtractedProfile.model_validate(normalize_llm_profile_payload(raw))

    assert normalized.skills[0].name == "Python"
    assert normalized.projects[0].skills == ["Python", "SQL"]


def test_normalizes_object_items_in_experience_achievements() -> None:
    raw = {
        "experiences": [
            {
                "jobTitle": "Data Engineer",
                "achievements": [{"description": "Processed 1.2M events/day"}],
            }
        ]
    }

    normalized = LLMExtractedProfile.model_validate(normalize_llm_profile_payload(raw))

    assert normalized.experiences[0].achievements == ["Processed 1.2M events/day"]


def test_normalizes_year_range_dates_before_dto_validation() -> None:
    raw = {
        "education": [
            {
                "institution": "Example University",
                "startDate": "2018 - 2022",
                "endDate": "2018 - 2022",
            }
        ]
    }

    normalized = LLMExtractedProfile.model_validate(normalize_llm_profile_payload(raw))

    assert str(normalized.education[0].start_date) == "2018-01-01"
    assert str(normalized.education[0].end_date) == "2022-01-01"


def test_normalizes_single_evidence_object_to_a_list_before_dto_validation() -> None:
    raw = {
        "headline": "Data Science Intern",
        "evidence": {
            "headline": {
                "text": "Data Science Intern",
                "pageNumber": 1,
                "blockId": "p1-b2",
            }
        },
    }

    with pytest.raises(ValidationError):
        LLMExtractedProfile.model_validate(raw)
    normalized = LLMExtractedProfile.model_validate(normalize_llm_profile_payload(raw))

    assert len(normalized.evidence["headline"]) == 1
    assert normalized.evidence["headline"][0].text == "Data Science Intern"


def test_normalizes_observed_gpa_notes_and_evidence_aliases() -> None:
    raw = {
        "education": [
            {
                "institution": "Example University",
                "gpa": "2.91/4",
                "notes": "omit this non-contract field",
                "evidence": {"quote": "Example University", "page": 1, "block": "p1-b4"},
            }
        ],
        "experiences": [
            {
                "jobTitle": "Data Engineer",
                "evidence": {"sourceText": "Data Engineer", "page": 1, "block_id": "p1-b2"},
            }
        ],
    }

    normalized = LLMExtractedProfile.model_validate(normalize_llm_profile_payload(raw))

    assert normalized.education[0].gpa is not None
    assert normalized.education[0].gpa.value == 2.91
    assert normalized.education[0].gpa.scale == 4
    assert normalized.education[0].evidence[0].block_id == "p1-b4"
    assert normalized.experiences[0].evidence[0].text == "Data Engineer"


def test_normalizes_generic_employment_aliases_and_month_year_dates() -> None:
    raw = {
        "quantifiedAchievements": ["must not be a top-level contract field"],
        "experiences": [
            {
                "title": "Senior Data Engineer",
                "employer": "Example Co",
                "startDate": "July 2023",
                "endDate": "Present",
                "notes": "non-contract commentary",
                "evidence": {"quote": "Senior Data Engineer", "page": 1, "block": "p1-b8"},
            }
        ],
        "education": [
            {
                "university": "Example University",
                "specialization": "Computer Science",
                "startDate": "09/2018",
                "endDate": "June 2022",
                "description": "non-contract detail",
            }
        ],
    }

    normalized = LLMExtractedProfile.model_validate(normalize_llm_profile_payload(raw))

    experience = normalized.experiences[0]
    assert experience.job_title == "Senior Data Engineer"
    assert experience.company == "Example Co"
    assert str(experience.start_date) == "2023-07-01"
    assert experience.end_date is None
    assert experience.is_current is True
    education = normalized.education[0]
    assert education.institution == "Example University"
    assert education.field_of_study == "Computer Science"
    assert str(education.start_date) == "2018-09-01"
    assert str(education.end_date) == "2022-06-01"


def test_normalizes_observed_year_month_and_structural_aliases() -> None:
    raw = {
        "experiences": [
            {
                "jobTitle": "Data Engineer",
                "startDate": "2024-09",
                "responsibilities": ["Built reliable data pipelines."],
                "techStack": ["Python", "SQL"],
            }
        ],
        "education": [
            {
                "institution": "Example University",
                "details": "Major: Computer Science.",
                "achievements": ["Dean's list"],
                "thesis": "A non-contract research title",
                "honors": ["Dean's list"],
            }
        ],
    }

    normalized = LLMExtractedProfile.model_validate(normalize_llm_profile_payload(raw))

    assert str(normalized.experiences[0].start_date) == "2024-09-01"
    assert normalized.experiences[0].description == ["Built reliable data pipelines."]
    assert normalized.experiences[0].skills == ["Python", "SQL"]
    assert normalized.education[0].coursework == ["Major: Computer Science."]


def test_correction_date_normalization_does_not_trigger_semantic_regression(tmp_path: Path) -> None:
    session = FakeSession(
        [
            _completion(
                {
                    "experiences": [
                        {
                            "jobTitle": "Data Engineer",
                            "company": "Example Co",
                            "startDate": "September 2024",
                            "unexpected": "remove only this field",
                        }
                    ]
                }
            ),
            _completion(
                {
                    "experiences": [
                        {
                            "jobTitle": "Data Engineer",
                            "company": "Example Co",
                            "startDate": "2024-09",
                        }
                    ]
                }
            ),
        ]
    )

    profile = _provider(session).extract(_request(tmp_path))

    assert len(profile.experiences) == 1


def test_correction_cannot_remove_evidence_reference_from_surviving_entity(tmp_path: Path) -> None:
    session = FakeSession(
        [
            _completion(
                {
                    "experiences": [
                        {
                            "jobTitle": "Data Engineer",
                            "company": "Example Co",
                            "unexpected": "remove only this field",
                            "evidence": [
                                {"text": "Data Engineer", "pageNumber": 1, "blockId": "p1-b2"}
                            ],
                        }
                    ]
                }
            ),
            _completion(
                {
                    "experiences": [
                        {"jobTitle": "Data Engineer", "company": "Example Co", "evidence": []}
                    ]
                }
            ),
        ]
    )

    with pytest.raises(CoreError) as error:
        _provider(session).extract(_request(tmp_path))

    assert error.value.issue.code == ErrorCode.LLM_SEMANTIC_REGRESSION
    regression = error.value.issue.details["semanticRegression"]
    assert regression["lostEvidenceReferences"]["experiences"] == 1


def test_detects_length_limited_repeated_null_completion(tmp_path: Path) -> None:
    response = FakeResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": '{"items":["null","null","null","null",'
                        '"null","null","null","null"]}'
                    },
                    "finish_reason": "length",
                }
            ]
        },
    )

    with pytest.raises(CoreError) as error:
        _provider(FakeSession([response])).extract(_request(tmp_path))

    assert error.value.issue.code == ErrorCode.LLM_DEGENERATE_RESPONSE
    assert error.value.issue.details["degeneration"]["classification"] == "degenerate_repetition"


def test_authentication_timeout_and_rate_limit_are_safe_errors(tmp_path: Path) -> None:
    cases: list[tuple[list[FakeResponse | Exception], ErrorCode]] = [
        ([FakeResponse(401, {"error": "unauthorized"})], ErrorCode.LLM_AUTHENTICATION_ERROR),
        ([requests.Timeout()] * 3, ErrorCode.LLM_TIMEOUT),
        ([FakeResponse(429, {"error": "limited"})] * 3, ErrorCode.LLM_RATE_LIMIT),
    ]
    for responses, expected in cases:
        with pytest.raises(CoreError) as error:
            _provider(FakeSession(responses)).extract(_request(tmp_path))
        assert error.value.issue.code == expected
