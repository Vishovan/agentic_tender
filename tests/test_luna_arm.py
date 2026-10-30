"""The GPT-Luna arm must hand clean, code-parsed constraints to the checker."""

import json
from pathlib import Path

from tenderai.luna_arm import requirements_from_luna_json


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
