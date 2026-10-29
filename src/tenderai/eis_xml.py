"""Read KTRU characteristics from an EIS notice (Извещение.xml).

The notice encodes each required characteristic in a
`characteristicsUsingReferenceInfo` block with a name, an OKEI unit, and one or
more values: a text `qualityDescription`, a numeric `concreteValue`, or a
`rangeSet` with `greaterOrEqual` / `lessOrEqual` style operators. These are
structured, so they are parsed by code rather than by the model.
"""

from __future__ import annotations

import hashlib
import re

from tenderai.inventory import FileRecord
from tenderai.models import Requirement, Span
from tenderai.parse import fold, parse_constraint

_BLOCK = re.compile(
    r"<(?:\w+:)?characteristicsUsingReferenceInfo>(.*?)</(?:\w+:)?characteristicsUsingReferenceInfo>",
    re.S,
)
_NOTATION = {
    "greaterorequal": ">=",
    "greater": ">",
    "lessorequal": "<=",
    "less": "<",
}


def has_characteristics(raw: str) -> bool:
    return "characteristicsUsingReferenceInfo" in raw


def requirements_from_eis_xml(record: FileRecord, raw: str) -> list[Requirement]:
    product = _first(raw, r"<(?:\w+:)?purchaseObjects?>.*?<(?:\w+:)?name>(.*?)</") or "Насос инфузионный эластомерный"
    position_id = f"{record.tender_id}:1:{_slug(product)}"
    found: list[Requirement] = []
    for index, block in enumerate(_BLOCK.findall(raw), start=1):
        name = _first(block, r"<(?:\w+:)?name>(.*?)</")
        if not name:
            continue
        unit = _first(block, r"<(?:\w+:)?nationalCode>(.*?)</") or _first(
            block, r"<(?:\w+:)?OKEI>.*?<(?:\w+:)?name>(.*?)</", re.S
        )
        code = _first(block, r"<(?:\w+:)?code>(.*?)</") or _first(block, r"<(?:\w+:)?sid>(.*?)</") or str(index)
        constraint, raw_text = _constraint(block, name, unit)
        found.append(
            Requirement(
                requirement_id=hashlib.sha1(f"{position_id}|{name}|{code}".encode()).hexdigest()[:12],
                tender_id=record.tender_id or "",
                position_id=position_id,
                position_title=product,
                name=name,
                raw_text=raw_text,
                mandatory=True,
                constraint=constraint,
                span=Span(
                    file=record.path,
                    locator=f"characteristic/{code}",
                    quote=f"{name}: {raw_text}",
                ),
            )
        )
    return found


def _constraint(block: str, name: str, unit: str | None):
    value_range = _first(block, r"<(?:\w+:)?valueRange>(.*?)</(?:\w+:)?valueRange>", re.S)
    if value_range:
        low = _first(value_range, r"<(?:\w+:)?min>(.*?)</")
        high = _first(value_range, r"<(?:\w+:)?max>(.*?)</")
        op_low = _NOTATION.get(fold(_first(value_range, r"<(?:\w+:)?minMathNotation>(.*?)</") or ""), ">=")
        op_high = _NOTATION.get(fold(_first(value_range, r"<(?:\w+:)?maxMathNotation>(.*?)</") or ""), "<=")
        raw_text = f"{op_low} {low} и {op_high} {high}".strip()
        return (
            parse_constraint("", unit=unit, name=name, operator_low=op_low, low=low, operator_high=op_high, high=high),
            raw_text,
        )

    concretes = re.findall(r"<(?:\w+:)?concreteValue>(.*?)</", block)
    if len(concretes) > 1:
        raw_text = " / ".join(concretes)
        return parse_constraint(raw_text, unit=unit, name=name), raw_text
    if len(concretes) == 1:
        return parse_constraint(concretes[0], unit=unit, name=name), concretes[0]

    quality = _first(block, r"<(?:\w+:)?qualityDescription>(.*?)</")
    if quality:
        return parse_constraint(quality, unit=unit, name=name), quality
    return parse_constraint("", unit=unit, name=name), ""


def _first(text: str, pattern: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _slug(title: str) -> str:
    slug = "".join(char if char.isalnum() else "-" for char in fold(title))
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:48] or "position"
