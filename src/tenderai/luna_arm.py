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


def requirements_from_luna_json(path: Path) -> list[Requirement]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tender_id = str(payload.get("tender_id", ""))
    source_file = str(payload.get("source_file", str(path)))
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
