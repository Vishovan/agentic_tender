"""Parse the phrases that the checker is allowed to compare."""

from __future__ import annotations

import re

from tenderai.models import Constraint

_MISSING = {
    "",
    "-",
    "—",
    "–",
    "не указано",
    "не установлено",
    "нет данных",
    "н/д",
}

_FALSE = {"нет", "отсутствует", "не предусмотрено"}
_TRUE = {"да", "наличие", "есть", "имеется"}

_UNIT_SPEED = ("мл/час", "мл/ч", "мл/h", "ml/h")
_UNIT_VOLUME = ("кубическ", "миллилитр", "см3", "см^3", "мл", "ml")


def fold(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_unit(unit: str | None, name: str = "") -> str | None:
    blob = fold(f"{unit or ''} {name}")
    if any(token in blob for token in _UNIT_SPEED):
        return "ml/h"
    if "мкм" in blob or "микрон" in blob:
        return "um"
    if any(token in blob for token in _UNIT_VOLUME) or "кубический сантиметр" in blob:
        return "ml"
    if re.search(r"\bсм\b", blob) or "сантиметр" in blob:
        return "cm"
    if "%" in blob or "процент" in blob:
        return "pct"
    if "мин" in blob:
        return "min"
    return None


def _number(token: str) -> float | None:
    token = token.strip().replace(" ", "").replace(",", ".")
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _split_numbers(text: str) -> list[float] | None:
    parts = re.split(r"\s*/\s*|\s*;\s*", text)
    if len(parts) < 2:
        return None
    numbers = [_number(part) for part in parts]
    if any(number is None for number in numbers):
        return None
    return [number for number in numbers if number is not None]


def parse_constraint(
    raw: str | None,
    unit: str | None = None,
    name: str = "",
    knowledge: str = "stated",
    operator_low: str | None = None,
    low: str | None = None,
    operator_high: str | None = None,
    high: str | None = None,
    concrete: str | None = None,
) -> Constraint:
    unit_norm = normalize_unit(unit, name)
    low_number = _number(low or "")
    high_number = _number(high or "")
    concrete_number = _number(concrete or "")
    if low_number is not None or high_number is not None or concrete_number is not None:
        return _from_columns(
            unit_norm,
            knowledge,
            operator_low,
            low_number,
            operator_high,
            high_number,
            concrete_number,
        )

    text = fold(raw or "")
    if text in _MISSING:
        return Constraint(kind="unknown", unit=unit_norm, knowledge="not_established")
    if text in _FALSE:
        return Constraint(kind="bool", value=0, text="нет", unit=unit_norm, knowledge=knowledge)
    if text in _TRUE:
        return Constraint(kind="bool", value=1, text="да", unit=unit_norm, knowledge=knowledge)
    if text == "соответствие":
        return Constraint(kind="reference", text=text, unit=unit_norm, knowledge=knowledge)

    listed = _split_numbers(raw or "")
    if listed:
        return Constraint(kind="set", values=listed, unit=unit_norm, knowledge=knowledge)

    bracket = re.search(
        r"[\[(]\s*(-?\d+(?:[.,]\d+)?)\s*[;–-]\s*(-?\d+(?:[.,]\d+)?)\s*[\])]",
        text,
    )
    if bracket:
        return Constraint(
            kind="range",
            low=_number(bracket.group(1)),
            high=_number(bracket.group(2)),
            unit=unit_norm,
            knowledge=knowledge,
        )

    pair = re.search(
        r"(>=|≤|<=|≥|>|<)\s*(-?\d+(?:[.,]\d+)?)\s*(?:и|,)?\s*(>=|≤|<=|≥|>|<)\s*(-?\d+(?:[.,]\d+)?)",
        text,
    )
    if pair:
        return _from_columns(
            unit_norm,
            knowledge,
            pair.group(1),
            _number(pair.group(2)),
            pair.group(3),
            _number(pair.group(4)),
            None,
        )

    single = re.fullmatch(r"(>=|≤|<=|≥|>|<)\s*(-?\d+(?:[.,]\d+)?)", text)
    if single:
        return _from_columns(
            unit_norm,
            knowledge,
            single.group(1),
            _number(single.group(2)),
            None,
            None,
            None,
        )

    phrase = re.search(
        r"(не более|не менее|не выше|не ниже|не превыша\w*|максимум|минимум|более|менее|свыше)\s*(-?\d+(?:[.,]\d+)?)",
        text,
    )
    if phrase:
        word = phrase.group(1)
        number = _number(phrase.group(2))
        if word in {"не более", "не выше", "максимум"} or word.startswith("не превыша"):
            return _from_columns(unit_norm, knowledge, None, None, "<=", number, None)
        if word in {"не менее", "не ниже", "минимум"}:
            return _from_columns(unit_norm, knowledge, ">=", number, None, None, None)
        if word in {"более", "свыше"}:
            return _from_columns(unit_norm, knowledge, ">", number, None, None, None)
        if word == "менее":
            return _from_columns(unit_norm, knowledge, None, None, "<", number, None)

    bare = _number(text)
    if bare is not None:
        return Constraint(
            kind="number",
            operator="eq",
            value=bare,
            unit=unit_norm,
            knowledge=knowledge,
        )
    return Constraint(kind="text", text=text, unit=unit_norm, knowledge=knowledge)


def _inclusive(operator: str | None, side: str) -> bool:
    token = (operator or "").strip()
    if side == "low":
        return token in {"", ">=", "≥", "=>"}
    return token in {"", "<=", "≤"}


def _from_columns(
    unit: str | None,
    knowledge: str,
    operator_low: str | None,
    low: float | None,
    operator_high: str | None,
    high: float | None,
    concrete: float | None,
) -> Constraint:
    if concrete is not None and low is None and high is None:
        return Constraint(
            kind="number",
            operator="eq",
            value=concrete,
            unit=unit,
            knowledge=knowledge,
        )
    if low is not None and high is not None:
        return Constraint(
            kind="range",
            low=low,
            high=high,
            low_inclusive=_inclusive(operator_low, "low"),
            high_inclusive=_inclusive(operator_high, "high"),
            unit=unit,
            knowledge=knowledge,
        )
    if low is not None:
        operator = "gte" if _inclusive(operator_low, "low") else "gt"
        return Constraint(
            kind="number",
            operator=operator,
            value=low,
            unit=unit,
            knowledge=knowledge,
        )
    if high is not None:
        operator = "lte" if _inclusive(operator_high, "high") else "lt"
        return Constraint(
            kind="number",
            operator=operator,
            value=high,
            unit=unit,
            knowledge=knowledge,
        )
    return Constraint(kind="unknown", unit=unit, knowledge="not_established")
