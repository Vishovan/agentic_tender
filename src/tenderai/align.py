"""Lexical alignment. A miss stays unknown; it is not a mismatch."""

from __future__ import annotations

import re

from tenderai.models import Requirement, SkuCharacteristic
from tenderai.parse import fold

# Longer, more specific keys are listed before the shorter ones they contain.
_CANON: list[tuple[str, tuple[str, ...]]] = [
    ("bolus_lock", ("блокир", "болюс")),
    ("speed_control", ("регулир", "скорост")),
    ("infusion_speed", ("скорост",)),
    ("reservoir_volume", ("объем", "резервуар")),
    ("fill_volume", ("объем", "заполнен")),
    ("residual_volume", ("остаточн", "объем")),
    ("uv", ("уф",)),
    ("filter_pore", ("пор", "фильтр")),
    ("clamp", ("зажим",)),
    ("route_iv", ("внутривен",)),
    ("route_epi", ("эпидураль",)),
    ("sterile", ("стериль",)),
    ("bolus", ("болюс",)),
    ("flow_limiter", ("ограничител", "поток")),
]


def canon_key(name: str) -> str | None:
    text = fold(name)
    for key, needles in _CANON:
        if all(needle in text for needle in needles):
            return key
    return None


_NUMERIC = {"number", "range", "set"}


def _comparable(requirement: Requirement, characteristic: SkuCharacteristic) -> bool:
    if characteristic.constraint.kind == "unknown":
        return False
    left = requirement.constraint.kind
    right = characteristic.constraint.kind
    if left in _NUMERIC and right in _NUMERIC:
        return True
    if left == "bool" and right in {"bool", "text"}:
        return True
    if left == "text" and right in {"bool", "text"}:
        return True
    return left == right


def _tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^0-9a-zа-я]+", fold(name)) if len(token) >= 4}


def align(
    requirement: Requirement,
    characteristics: list[SkuCharacteristic],
    threshold: float = 0.5,
) -> tuple[SkuCharacteristic | None, str | None, float | None]:
    requirement_key = canon_key(requirement.name)
    if requirement_key:
        candidates = [
            characteristic
            for characteristic in characteristics
            if canon_key(characteristic.name) == requirement_key
        ]
        comparable = [
            characteristic
            for characteristic in candidates
            if _comparable(requirement, characteristic)
        ]
        chosen = comparable[0] if comparable else (candidates[0] if candidates else None)
        if chosen is not None:
            return chosen, "canon", 1.0

    requirement_tokens = _tokens(requirement.name)
    best: SkuCharacteristic | None = None
    best_score = 0.0
    for characteristic in characteristics:
        tokens = _tokens(characteristic.name)
        if not requirement_tokens or not tokens:
            continue
        score = len(requirement_tokens & tokens) / len(requirement_tokens | tokens)
        if score > best_score:
            best = characteristic
            best_score = score
    if best is not None and best_score >= threshold:
        return best, "tokens", best_score
    return None, None, None
