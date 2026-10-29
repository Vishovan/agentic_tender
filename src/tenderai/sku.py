"""Load the prepared SKU workbooks. They are evidence, not match answers."""

from __future__ import annotations

from pathlib import Path

from tenderai.extract import Cell, Table, extract_document
from tenderai.inventory import FileRecord
from tenderai.models import SkuCharacteristic, Span
from tenderai.parse import fold, parse_constraint

_KNOWLEDGE = {
    "указано": "stated",
    "не установлено": "not_established",
    "вывод по документации": "inferred",
}


def load_skus(dataset: Path, records: list[FileRecord]) -> list[SkuCharacteristic]:
    characteristics: list[SkuCharacteristic] = []
    for record in records:
        if record.role != "sku":
            continue
        document = extract_document(dataset, record)
        if document.status != "ok":
            continue
        characteristics.extend(_from_tables(record, document.tables))
    return characteristics


def _from_tables(record: FileRecord, tables: list[Table]) -> list[SkuCharacteristic]:
    found: list[SkuCharacteristic] = []
    for table in tables:
        header_index = _header_row(table.rows)
        if header_index is None:
            continue
        product_name = ""
        if header_index > 0 and table.rows[0]:
            product_name = table.rows[0][1].value if len(table.rows[0]) > 1 else ""
        columns = _columns(table.rows[header_index])
        for offset, row in enumerate(table.rows[header_index + 1 :], start=1):
            name = _cell(row, columns.get("name"))
            if not name:
                continue
            value = _cell(row, columns.get("value"))
            unit = _cell(row, columns.get("unit"))
            knowledge_raw = fold(_cell(row, columns.get("knowledge")))
            source = _cell(row, columns.get("source"))
            quote = _cell(row, columns.get("quote")) or value or name
            knowledge = _KNOWLEDGE.get(knowledge_raw, "stated")
            if fold(value) in {"не указано", "не установлено", "—", "-", ""}:
                knowledge = "not_established"
            name_cell = _find(row, columns.get("name"))
            constraint = parse_constraint(value, unit=unit, name=name, knowledge=knowledge)
            found.append(
                SkuCharacteristic(
                    sku_id=record.sku_id or Path(record.path).stem,
                    characteristic_id=f"{record.sku_id}:{table.sheet}:{offset}",
                    name=name,
                    value_raw=value,
                    unit_raw=unit,
                    knowledge=knowledge,
                    constraint=constraint,
                    product_name=product_name,
                    span=Span(
                        file=record.path,
                        locator=f"{table.sheet}!{name_cell.a1 if name_cell else '?'}",
                        quote=quote if not source else f"{quote} [{source}]",
                    ),
                )
            )
    return found


def _header_row(rows: list[list[Cell]]) -> int | None:
    for index, row in enumerate(rows):
        if any(fold(cell.value) == "характеристика" for cell in row):
            return index
    return None


def _columns(row: list[Cell]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cell in row:
        label = fold(cell.value)
        if label == "характеристика":
            mapping["name"] = cell.a1[0] if cell.a1[:1].isalpha() else cell.a1
        elif label == "значение":
            mapping["value"] = _column_letters(cell.a1)
        elif label == "единица":
            mapping["unit"] = _column_letters(cell.a1)
        elif label == "сведения":
            mapping["knowledge"] = _column_letters(cell.a1)
        elif label == "источник":
            mapping["source"] = _column_letters(cell.a1)
        elif "формулиров" in label:
            mapping["quote"] = _column_letters(cell.a1)
    if "name" in mapping and not mapping["name"].isalpha():
        mapping["name"] = _column_letters(next(cell.a1 for cell in row if fold(cell.value) == "характеристика"))
    else:
        mapping["name"] = _column_letters(
            next(cell.a1 for cell in row if fold(cell.value) == "характеристика")
        )
    return mapping


def _column_letters(a1: str) -> str:
    letters = []
    for char in a1:
        if char.isalpha():
            letters.append(char)
        else:
            break
    return "".join(letters)


def _cell(row: list[Cell], column: str | None) -> str:
    found = _find(row, column)
    return found.value if found else ""


def _find(row: list[Cell], column: str | None) -> Cell | None:
    if not column:
        return None
    for cell in row:
        if _column_letters(cell.a1) == column:
            return cell
    return None
