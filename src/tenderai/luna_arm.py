"""Turn GPT-Luna's prose extraction into requirements the code checker can use.

The model's only job is to find each characteristic and quote it verbatim; the
numbers, units, operators, sets and negations are parsed here by
``parse_constraint`` and judged by the deterministic checker. This keeps the
LLM/code boundary the task requires: semantics stay in code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tenderai.models import Requirement, Span
from tenderai.parse import fold, parse_constraint

_SCHEMA = (
    '{"tender_id":"<id>","source_file":"<path>","positions":[{"position_title":'
    '"<short id incl. distinguishing value>","requirements":[{"name":"<characteristic>",'
    '"requirement_text":"<verbatim value, keep >=, <=, sets like 2; 4; 6, negations>",'
    '"unit":"<verbatim unit or empty>","source_quote":"<verbatim snippet>","page":<int or null>}]}]}'
)


def build_extraction_prompt(tender_id: str, source_file: str, text: str) -> str:
    """Prompt GPT-Luna to extract requirements; numbers stay for the code step."""
    return (
        "You are the requirement-extraction step of a procurement technical-matching "
        "pipeline for elastomeric infusion pumps (госзакупки). Read the technical "
        "specification text below and emit every TECHNICAL characteristic as strict JSON. "
        "A deterministic code step parses the numbers/units/operators, so preserve them "
        "VERBATIM and do NOT decide match/mismatch.\n\n"
        "Rules: group by procurement POSITION; keep operators (>=, <=), sets (\"2; 4; 6\"), "
        "negations (\"Нет\"), numbers and units exactly as written; skip price, НМЦК, delivery "
        "term, address, warranty, justification prose; never invent a value; copy quotes verbatim.\n\n"
        f"Output ONLY valid JSON (no markdown fences, no commentary), shape:\n{_SCHEMA}\n\n"
        f'Use tender_id \"{tender_id}\" and source_file \"{source_file}\".\n\n'
        f"=== SPECIFICATION TEXT ===\n{text}\n=== END TEXT ==="
    )


def parse_luna_payload(raw: str) -> dict:
    """Read the model's JSON, tolerating stray prose or code fences around it."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw.strip("`")
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def requirements_from_luna_payload(payload: dict) -> list[Requirement]:
    return _requirements(payload)


def requirements_from_luna_json(path: Path) -> list[Requirement]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _requirements(payload)


def _requirements(payload: dict) -> list[Requirement]:
    tender_id = str(payload.get("tender_id", ""))
    source_file = str(payload.get("source_file", ""))
    found: list[Requirement] = []
    for index, position in enumerate(payload.get("positions", []), start=1):
        title = str(position.get("position_title") or f"Позиция {index}")
        position_id = f"{tender_id}:{index}:{_slug(title)}"
        for requirement in position.get("requirements", []):
            name = str(requirement.get("name", "")).strip()
            raw_text = str(requirement.get("requirement_text", "")).strip()
            if not name or not raw_text:
                continue
            unit = str(requirement.get("unit", "") or "")
            quote = str(requirement.get("source_quote") or f"{name}: {raw_text}")
            page = requirement.get("page")
            constraint = parse_constraint(raw_text, unit=unit, name=name)
            requirement_id = hashlib.sha1(
                f"{position_id}|{name}|{quote}".encode()
            ).hexdigest()[:12]
            found.append(
                Requirement(
                    requirement_id=requirement_id,
                    tender_id=tender_id,
                    position_id=position_id,
                    position_title=title,
                    name=name,
                    raw_text=raw_text,
                    mandatory=True,
                    constraint=constraint,
                    span=Span(
                        file=source_file,
                        locator=f"page/{page}" if page is not None else "page/?",
                        quote=" ".join(quote.split()),
                    ),
                )
            )
    return found


def _slug(title: str) -> str:
    slug = "".join(char if char.isalnum() else "-" for char in fold(title))
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:48] or "position"
