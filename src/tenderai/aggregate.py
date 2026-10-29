"""Roll requirement checks up to a position and a tender."""

from __future__ import annotations

from tenderai.models import Check


def position_status(checks: list[Check]) -> tuple[str, str]:
    if not checks:
        return "unknown", "no_requirements"
    if any(check.status == "mismatch" for check in checks):
        return "mismatch", "mandatory_conflict"
    if all(check.status == "match" and _has_evidence(check) for check in checks):
        return "match", "all_requirements_confirmed"
    return "unknown", "unresolved_requirements"


def tender_status(
    position_statuses: list[str],
    coverage_gap: bool,
    gap_reason: str = "coverage_gap",
) -> tuple[str, str]:
    if any(status == "match" for status in position_statuses):
        return "match", "suitable_position"
    if coverage_gap:
        return "unknown", gap_reason
    if position_statuses and all(status == "mismatch" for status in position_statuses):
        return "mismatch", "all_positions_conflict"
    return "unknown", "no_confirmed_position"


def _has_evidence(check: Check) -> bool:
    return bool(check.requirement_span.quote) and bool(check.sku_span and check.sku_span.quote)
