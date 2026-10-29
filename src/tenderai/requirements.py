"""Read technical rows from OOZ-style tables. Price and delivery stay out."""

from __future__ import annotations

import hashlib

from tenderai.extract import Cell, Table
from tenderai.inventory import FileRecord
from tenderai.models import Requirement, Span
from tenderai.parse import fold, parse_constraint

_SKIP_NAME = (
    "описание",
    "цена",
    "срок постав",
    "адрес",
    "срок годност",
    "обеспечен",
    "ндс",
)


def requirements_from_document(record: FileRecord, tables: list[Table]) -> list[Requirement]:
    found: list[Requirement] = []
    for table in tables:
        header = _header(table.rows)
        if header is None:
            continue
        header_index, columns = header
        position_title = ""
        position_number = ""
        for row in table.rows[header_index + 1 :]:
            if _is_index_row(row):
                continue
            title = _value(row, columns.get("title"))
            number = _value(row, columns.get("number"))
            if title:
                position_title = title
            if number:
                position_number = number
            name = _value(row, columns.get("name"))
            if not name or _skip(name):
                continue
            raw_bits = [
                _value(row, columns.get("text")),
                _value(row, columns.get("concrete")),
                _value(row, columns.get("low")),
                _value(row, columns.get("high")),
            ]
            raw_text = " | ".join(bit for bit in raw_bits if bit) or name
            constraint = parse_constraint(
                _value(row, columns.get("text")),
                unit=_value(row, columns.get("unit")),
                name=name,
                operator_low=_value(row, columns.get("op_low")),
                low=_value(row, columns.get("low")),
                operator_high=_value(row, columns.get("op_high")),
                high=_value(row, columns.get("high")),
                concrete=_value(row, columns.get("concrete")),
            )
            name_cell = _find(row, columns.get("name"))
            position_id = f"{record.tender_id}:{position_number or '1'}:{_slug(position_title)}"
            requirement_id = hashlib.sha1(
                f"{position_id}|{name}|{name_cell.a1 if name_cell else raw_text}".encode()
            ).hexdigest()[:12]
            instruction = fold(_value(row, columns.get("instruction")))
            found.append(
                Requirement(
                    requirement_id=requirement_id,
                    tender_id=record.tender_id or "",
                    position_id=position_id,
                    position_title=position_title,
                    name=name,
                    raw_text=raw_text,
                    mandatory="не может изменяться" in instruction or not instruction,
                    constraint=constraint,
                    span=Span(
                        file=record.path,
                        locator=f"{table.sheet}!{name_cell.a1 if name_cell else '?'}",
                        quote=f"{name}: {raw_text}",
                    ),
                )
            )
    return found


def _header(rows: list[list[Cell]]) -> tuple[int, dict[str, str]] | None:
    for index, row in enumerate(rows):
        labels = {fold(cell.value): _letters(cell.a1) for cell in row}
        name_column = _match(
            labels,
            ("наименование характеристики", "наименование показателя", "характеристики с указанием"),
        )
        if not name_column:
            continue
        if not (
            _match(labels, ("значение характеристики", "текстовое описание", "значение показателя"))
            or _match(labels, ("min",))
            or "min" in labels
        ):
            continue
        columns = {
            "name": name_column,
            "text": _match(labels, ("значение характеристики", "текстовое описание", "значение показателя")),
            "unit": _match(
                labels,
                ("единица измерения характеристики", "единица измерения показателя", "единица измерения"),
            ),
            "title": _match(labels, ("наименование товара",)),
            "number": _match_exact(labels, ("№ п/п", "№")),
            "op_low": _match(labels, (">, >=", ">=", ">")),
            "low": _match_exact(labels, ("min",)),
            "op_high": _match(labels, ("<, <=", "<=", "<")),
            "high": _match_exact(labels, ("max",)),
            "concrete": _match(labels, ("конкретное значение",)),
            "instruction": _match(labels, ("инструкция",)),
        }
        return index, {key: value for key, value in columns.items() if value}
    return None


def _match(labels: dict[str, str], needles: tuple[str, ...]) -> str | None:
    for label, column in labels.items():
        if any(needle in label for needle in needles):
            return column
    return None


def _match_exact(labels: dict[str, str], needles: tuple[str, ...]) -> str | None:
    for needle in needles:
        if needle in labels:
            return labels[needle]
    return None


def _letters(a1: str) -> str:
    return "".join(char for char in a1 if char.isalpha())


def _find(row: list[Cell], column: str | None) -> Cell | None:
    if not column:
        return None
    for cell in row:
        if _letters(cell.a1) == column:
            return cell
    return None


def _value(row: list[Cell], column: str | None) -> str:
    found = _find(row, column)
    return found.value if found else ""


def _is_index_row(row: list[Cell]) -> bool:
    values = [fold(cell.value) for cell in row]
    numeric = [value for value in values if value.isdigit()]
    return len(numeric) >= 4 and len(numeric) == len([value for value in values if value])


def _skip(name: str) -> bool:
    text = fold(name)
    return any(needle in text for needle in _SKIP_NAME)


def _slug(title: str) -> str:
    slug = "".join(char if char.isalnum() else "-" for char in fold(title))
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:48] or "position"
