"""The GPT-Luna arm must hand clean, code-parsed constraints to the checker."""

import json
from pathlib import Path

import pytest

from tenderai.luna_arm import (
    build_extraction_prompt,
    parse_luna_payload,
    requirements_from_luna_json,
    requirements_from_luna_payload,
)
from tenderai.luna_client import call_luna


def test_luna_json_parses_into_checkable_constraints(tmp_path: Path):
    payload = {
        "tender_id": "T777",
        "source_file": "tenders/02/tz.pdf",
        "positions": [
            {
                "position_title": "Позиция 1",
                "requirements": [
                    {
                        "name": "Объем заполнения",
                        "requirement_text": ">= 100 <= 124",
                        "unit": "см[3*];^мл.",
                        "source_quote": "Объем заполнения: >= 100 <= 124 см3; мл.",
                        "page": 5,
                    },
                    {
                        "name": "Болюс",
                        "requirement_text": "Нет",
                        "unit": "",
                        "source_quote": "Болюс: Нет.",
                        "page": 5,
                    },
                    {
                        "name": "Скорость инфузии",
                        "requirement_text": "2; 4; 6",
                        "unit": "мл/час",
                        "source_quote": "Скорость инфузии, мл/час: 2; 4; 6.",
                        "page": 5,
                    },
                ],
            }
        ],
    }
    path = tmp_path / "luna.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    requirements = requirements_from_luna_json(path)
    assert len(requirements) == 3
    by_name = {r.name: r for r in requirements}

    volume = by_name["Объем заполнения"].constraint
    assert volume.kind == "range"
    assert (volume.low, volume.high) == (100.0, 124.0)
    assert volume.unit == "ml"

    assert by_name["Болюс"].constraint.kind == "bool"
    assert by_name["Болюс"].constraint.value == 0

    speed = by_name["Скорость инфузии"].constraint
    assert speed.kind == "set"
    assert speed.values == [2.0, 4.0, 6.0]
    assert speed.unit == "ml/h"

    # evidence and provenance survive
    first = by_name["Объем заполнения"]
    assert first.tender_id == "T777"
    assert first.span.file == "tenders/02/tz.pdf"
    assert first.span.locator == "page/5"
    assert "Объем заполнения" in first.span.quote


def test_parse_luna_payload_tolerates_fences_and_prose():
    body = '{"tender_id":"T1","positions":[]}'
    assert parse_luna_payload(body) == {"tender_id": "T1", "positions": []}
    assert parse_luna_payload(f"```json\n{body}\n```") == {"tender_id": "T1", "positions": []}
    assert parse_luna_payload(f"Here is the JSON:\n{body}\nDone.") == {
        "tender_id": "T1",
        "positions": [],
    }


def test_requirements_from_payload_matches_json_loader():
    payload = {
        "tender_id": "T1",
        "source_file": "f.pdf",
        "positions": [
            {"position_title": "P1", "requirements": [
                {"name": "Болюс", "requirement_text": "Да", "unit": "", "source_quote": "Болюс: Да", "page": 1}
            ]}
        ],
    }
    reqs = requirements_from_luna_payload(payload)
    assert len(reqs) == 1
    assert reqs[0].constraint.kind == "bool" and reqs[0].constraint.value == 1


def test_build_extraction_prompt_carries_text_and_schema():
    prompt = build_extraction_prompt("T1", "f.pdf", "Болюс: Нет.")
    assert "Болюс: Нет." in prompt
    assert "positions" in prompt and "requirement_text" in prompt
    assert "ONLY valid JSON" in prompt


def test_call_luna_without_key_raises_before_network(monkeypatch):
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
        call_luna("hi", api_key=None)
