"""Compare one requirement with one characteristic. Similarity is not an input."""

from __future__ import annotations

from tenderai.models import Constraint

_FALSE_TEXT = {"нет", "постоянная", "фиксированная", "отсутствует", "не предусмотрено"}
_TRUE_TEXT = {"да", "наличие", "есть", "регулируемая", "регулируемый", "имеется"}


def check_constraints(requirement: Constraint, sku: Constraint) -> tuple[str, str, str]:
    if requirement.kind in {"unknown", "reference"}:
        return "unknown", "requirement_unparsed", "У требования нет локального значения для проверки."
    if sku.kind == "unknown" or sku.knowledge == "not_established":
        return "unknown", "missing_sku_value", "В карточке SKU значение не установлено."
    if requirement.unit and sku.unit and requirement.unit != sku.unit:
        return "unknown", "unit_incompatible", "Единицы требования и SKU не приведены к одному основанию."

    if requirement.kind == "bool" and sku.kind == "bool":
        if requirement.value == sku.value:
            return "match", "bool_equal", "Логические значения совпали."
        return "mismatch", "bool_conflict", "Логические значения противоречат друг другу."

    if requirement.kind == "bool" and sku.kind == "text":
        return _bool_against_text(requirement, sku.text or "")
    if requirement.kind == "text" and sku.kind == "bool":
        return _text_against_bool(requirement.text or "", sku)

    if requirement.kind == "text" and sku.kind == "text":
        if requirement.text == sku.text:
            return "match", "text_equal", "Формулировки после нормализации совпали."
        return "mismatch", "text_conflict", "Категориальные значения различаются."

    if requirement.kind == "set":
        return _set_requirement(requirement, sku)
    if sku.kind == "set" and requirement.kind == "number" and requirement.operator == "eq":
        if _contains(sku.values, requirement.value):
            return "match", "value_in_sku_set", "Требуемое значение входит в набор SKU."
        return "mismatch", "value_not_in_sku_set", "Требуемого значения нет в наборе SKU."

    if requirement.kind == "range":
        return _range_requirement(requirement, sku)
    if requirement.kind == "number" and sku.kind == "number":
        return _number_requirement(requirement, sku.value)
    if requirement.kind == "number" and sku.kind == "range":
        return _point_against_sku_range(requirement, sku)

    return "unknown", "kinds_incompatible", "Формы значения нечем сравнивать детерминированно."


def _bool_against_text(requirement: Constraint, text: str) -> tuple[str, str, str]:
    wants = requirement.value == 1
    if text in (_TRUE_TEXT if wants else _FALSE_TEXT):
        return "match", "bool_text_compatible", "Текст SKU совместим с логическим требованием."
    if text in (_FALSE_TEXT if wants else _TRUE_TEXT):
        return "mismatch", "bool_text_conflict", "Текст SKU противоречит логическому требованию."
    return "unknown", "bool_text_unclear", "Текст SKU не сводится к да или нет."


def _text_against_bool(text: str, sku: Constraint) -> tuple[str, str, str]:
    sku_yes = sku.value == 1
    if text in _TRUE_TEXT:
        if sku_yes:
            return "match", "text_bool_compatible", "Требование подтверждено положительным значением SKU."
        return "mismatch", "text_bool_conflict", "Требование не подтверждено значением SKU."
    if text in _FALSE_TEXT:
        if not sku_yes:
            return "match", "text_bool_compatible", "Отрицательные значения совпали."
        return "mismatch", "text_bool_conflict", "SKU утверждает наличие того, что требование исключает."
    if sku_yes:
        return "match", "positive_attribute", "Характеристика SKU подтверждает требуемое свойство."
    return "mismatch", "negative_attribute", "Характеристика SKU отрицает требуемое свойство."


def _set_requirement(requirement: Constraint, sku: Constraint) -> tuple[str, str, str]:
    if sku.kind == "set":
        if all(_contains(sku.values, value) for value in requirement.values):
            return "match", "set_covered", "SKU поддерживает все требуемые значения."
        return "mismatch", "set_incomplete", "SKU не поддерживает часть требуемых значений."
    if sku.kind == "number":
        if len(requirement.values) == 1 and _contains([sku.value or 0], requirement.values[0]):
            return "match", "set_covered", "Единственное требуемое значение совпало с SKU."
        return "mismatch", "set_incomplete", "Одного значения SKU недостаточно для требуемого набора."
    if sku.kind == "range":
        if all(_inside(value, sku) for value in requirement.values):
            return "match", "set_inside_range", "Все требуемые значения лежат в диапазоне SKU."
        return "mismatch", "set_outside_range", "Часть требуемых значений вне диапазона SKU."
    return "unknown", "kinds_incompatible", "Набор требования не с чем сравнить."


def _range_requirement(requirement: Constraint, sku: Constraint) -> tuple[str, str, str]:
    if sku.kind == "number":
        if _inside(sku.value, requirement):
            return "match", "point_inside_range", "Значение SKU входит в требуемый диапазон."
        return "mismatch", "point_outside_range", "Значение SKU вне требуемого диапазона."
    if sku.kind == "range":
        if _range_subset(sku, requirement):
            return "match", "range_subset", "Диапазон SKU целиком лежит в требовании."
        return "mismatch", "range_not_subset", "Диапазон SKU выходит за требование."
    if sku.kind == "set":
        if all(_inside(value, requirement) for value in sku.values):
            return "match", "set_inside_requirement", "Все значения SKU входят в требование."
        return "mismatch", "set_outside_requirement", "Часть значений SKU вне требования."
    return "unknown", "kinds_incompatible", "Диапазон требования не с чем сравнить."


def _number_requirement(requirement: Constraint, sku_value: float | None) -> tuple[str, str, str]:
    if sku_value is None or requirement.value is None:
        return "unknown", "missing_sku_value", "Нет числа для сравнения."
    ok = _compare(sku_value, requirement.operator or "eq", requirement.value)
    if ok:
        return "match", "number_ok", "Число SKU выполняет оператор требования."
    return "mismatch", "number_failed", "Число SKU не выполняет оператор требования."


def _point_against_sku_range(requirement: Constraint, sku: Constraint) -> tuple[str, str, str]:
    if requirement.operator != "eq" or requirement.value is None:
        return "unknown", "kinds_incompatible", "С диапазоном SKU сравнивается только точное значение."
    if _inside(requirement.value, sku):
        return "match", "point_inside_sku_range", "Требуемое значение входит в диапазон SKU."
    return "mismatch", "point_outside_sku_range", "Требуемое значение вне диапазона SKU."


def _compare(left: float, operator: str, right: float) -> bool:
    if operator == "eq":
        return abs(left - right) <= 1e-6
    if operator == "gte":
        return left + 1e-9 >= right
    if operator == "gt":
        return left > right
    if operator == "lte":
        return left <= right + 1e-9
    if operator == "lt":
        return left < right
    return False


def _contains(values: list[float], target: float | None) -> bool:
    if target is None:
        return False
    return any(abs(value - target) <= 1e-6 for value in values)


def _inside(value: float | None, bounds: Constraint) -> bool:
    if value is None:
        return False
    if bounds.low is not None:
        if bounds.low_inclusive and value + 1e-9 < bounds.low:
            return False
        if not bounds.low_inclusive and value <= bounds.low:
            return False
    if bounds.high is not None:
        if bounds.high_inclusive and value > bounds.high + 1e-9:
            return False
        if not bounds.high_inclusive and value >= bounds.high:
            return False
    return True


def _range_subset(inner: Constraint, outer: Constraint) -> bool:
    if inner.low is None or inner.high is None:
        return False
    return _inside(inner.low, outer) and _inside(inner.high, outer)
