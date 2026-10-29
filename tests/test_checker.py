from tenderai.aggregate import position_status, tender_status
from tenderai.checker import check_constraints
from tenderai.models import Check, Constraint, Span
from tenderai.parse import parse_constraint


def test_cubic_centimeter_is_a_millilitre():
    parsed = parse_constraint("275", unit="Кубический сантиметр;^миллилитр", name="Номинальный объем резервуара")
    assert parsed.unit == "ml"


def test_volume_point_inside_requirement_range():
    requirement = parse_constraint("≥ 250 и ≤ 299", unit="мл", name="Номинальный объем резервуара")
    sku = parse_constraint("275", unit="мл", name="Номинальный объём резервуара")
    status, reason, _ = check_constraints(requirement, sku)
    assert requirement.kind == "range"
    assert (status, reason) == ("match", "point_inside_range")


def test_volume_point_outside_requirement_range():
    requirement = parse_constraint("", unit="мл", name="объём", low="400", high="499", operator_low=">=", operator_high="<=")
    sku = parse_constraint("250", unit="мл")
    status, reason, _ = check_constraints(requirement, sku)
    assert (status, reason) == ("mismatch", "point_outside_range")


def test_missing_sku_value_is_unknown():
    requirement = parse_constraint("Да", name="Защита от УФ")
    sku = parse_constraint("Не указано", name="Защита от УФ", knowledge="not_established")
    status, reason, _ = check_constraints(requirement, sku)
    assert (status, reason) == ("unknown", "missing_sku_value")


def test_explicit_no_conflicts_with_yes():
    requirement = parse_constraint("Нет", name="Болюс")
    sku = parse_constraint("Да", name="Болюс")
    status, reason, _ = check_constraints(requirement, sku)
    assert (status, reason) == ("mismatch", "bool_conflict")


def test_speed_set_needs_every_value():
    requirement = parse_constraint("1/2/3/4/5/6/7", unit="мл/ч", name="Скорость инфузии")
    sku = parse_constraint("5", unit="мл/ч", name="Настройки скорости")
    status, reason, _ = check_constraints(requirement, sku)
    assert (status, reason) == ("mismatch", "set_incomplete")


def test_speed_set_covered_by_sku_set():
    requirement = parse_constraint("1/2/3/4/5/6/7", unit="мл/час")
    sku = parse_constraint("0 / 1 / 2 / 3 / 4 / 5 / 6 / 7", unit="мл/ч")
    status, reason, _ = check_constraints(requirement, sku)
    assert requirement.unit == "ml/h"
    assert (status, reason) == ("match", "set_covered")


def test_one_mismatch_decides_the_position():
    checks = [_check("match"), _check("unknown"), _check("mismatch")]
    assert position_status(checks)[0] == "mismatch"


def test_unresolved_requirement_blocks_match():
    checks = [_check("match"), _check("unknown")]
    assert position_status(checks) == ("unknown", "unresolved_requirements")


def test_coverage_gap_blocks_tender_mismatch():
    assert tender_status(["mismatch"], coverage_gap=True) == ("unknown", "coverage_gap")
    assert tender_status(["mismatch"], coverage_gap=False) == ("mismatch", "all_positions_conflict")


def test_any_matching_position_makes_the_tender_match():
    assert tender_status(["unknown", "match"], coverage_gap=True)[0] == "match"


def _check(status: str) -> Check:
    span = Span("file", "A1", "цитата")
    return Check(
        requirement_id="r",
        sku_id="SKU",
        characteristic_id="c" if status != "unknown" else None,
        status=status,
        reason=status,
        align_method="canon",
        align_score=1,
        requirement_span=span,
        sku_span=span if status == "match" else None,
        explanation="",
    )
