"""Characteristic alignment: lexical baseline, dense FRIDA cosine, and hybrid RRF.

A miss stays ``unknown`` (``not_aligned``); it is never turned into ``mismatch``.
Semantic closeness only selects the candidate SKU characteristic — numbers,
ranges, sets, units and negations are verified deterministically in ``checker.py``.
"""

from __future__ import annotations

import re

from tenderai.embeddings import cosine
from tenderai.models import Requirement, SkuCharacteristic
from tenderai.parse import fold

DEFAULT_DENSE_THRESHOLD = 0.625
DEFAULT_RRF_K = 60

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


def _slot_compatible(requirement: Requirement, characteristic: SkuCharacteristic) -> bool:
    """Like ``_comparable``, but does not skip ``not_established`` SKU slots."""
    if characteristic.constraint.kind == "unknown":
        return True
    return _comparable(requirement, characteristic)


def _qualifier_compatible(req_name: str, sku_name: str) -> bool:
    """Prevent cross-matching conflicting metrological bounds or volume targets."""
    r = fold(req_name)
    s = fold(sku_name)
    if "объем" in r and "объем" in s:
        if ("остаточн" in r) != ("остаточн" in s):
            return False
        if ("заполнен" in r and "резервуар" in s) or ("резервуар" in r and "заполнен" in s):
            return False
    if "номинальн" in r and any(pole in s for pole in ("максимальн", "минимальн", "допустим", "остаточн", "таблиц")):
        return False
    if "максимальн" in r and any(pole in s for pole in ("номинальн", "минимальн", "остаточн")):
        return False
    if "минимальн" in r and any(pole in s for pole in ("номинальн", "максимальн", "остаточн")):
        return False
    return True


def _tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^0-9a-zа-я]+", fold(name)) if len(token) >= 4}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def align(
    requirement: Requirement,
    characteristics: list[SkuCharacteristic],
    threshold: float = 0.5,
) -> tuple[SkuCharacteristic | None, str | None, float | None]:
    """Lexical baseline aligner (canon keys + token Jaccard)."""
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
        score = _jaccard(requirement_tokens, tokens)
        if score > best_score:
            best = characteristic
            best_score = score
    if best is not None and best_score >= threshold:
        return best, "tokens", best_score
    return None, None, None


def rank_candidates(
    requirement: Requirement,
    characteristics: list[SkuCharacteristic],
    *,
    mode: str = "hybrid_rrf",
    req_vector: list[float] | None = None,
    sku_vectors: dict[str, list[float]] | None = None,
    lexical_threshold: float = 0.5,
    dense_threshold: float = DEFAULT_DENSE_THRESHOLD,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[SkuCharacteristic, str, float]]:
    """Return ranked candidate SKU characteristics for ``Recall@k`` and top-1 selection."""
    if mode == "lexical" or req_vector is None or sku_vectors is None:
        ranked: list[tuple[SkuCharacteristic, str, float]] = []
        seen: set[str] = set()
        requirement_key = canon_key(requirement.name)
        if requirement_key:
            canon_cands = [c for c in characteristics if canon_key(c.name) == requirement_key]
            comp = [c for c in canon_cands if _comparable(requirement, c)]
            rest = [c for c in canon_cands if c not in comp]
            for c in comp + rest:
                ranked.append((c, "canon", 1.0))
                seen.add(c.characteristic_id)
        req_tokens = _tokens(requirement.name)
        tok_scored = sorted(
            (
                (_jaccard(req_tokens, _tokens(c.name)), c)
                for c in characteristics
                if c.characteristic_id not in seen
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, c in tok_scored:
            if score >= lexical_threshold:
                ranked.append((c, "tokens", score))
        return ranked

    if mode == "dense":
        scored = sorted(
            (
                (cosine(req_vector, sku_vectors.get(c.characteristic_id, [])), c)
                for c in characteristics
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [(c, "dense", round(score, 4)) for score, c in scored]

    if mode == "hybrid_rrf":
        pool = [c for c in characteristics if _qualifier_compatible(requirement.name, c.name)]
        if not pool:
            return []

        dense_scores = {
            c.characteristic_id: cosine(req_vector, sku_vectors.get(c.characteristic_id, []))
            for c in pool
        }
        dense_order = sorted(pool, key=lambda c: dense_scores[c.characteristic_id], reverse=True)
        dense_rank = {c.characteristic_id: idx for idx, c in enumerate(dense_order, start=1)}

        # Lexical rank list over qualifier-compatible candidates
        lex_list: list[SkuCharacteristic] = []
        seen_lex: set[str] = set()
        req_key = canon_key(requirement.name)
        if req_key:
            canon_cands = [c for c in pool if canon_key(c.name) == req_key]
            comp = [c for c in canon_cands if _slot_compatible(requirement, c)]
            rest = [c for c in canon_cands if c not in comp]
            for c in comp + rest:
                lex_list.append(c)
                seen_lex.add(c.characteristic_id)
        req_tokens = _tokens(requirement.name)
        tok_cands = sorted(
            (
                (_jaccard(req_tokens, _tokens(c.name)), c)
                for c in pool
                if c.characteristic_id not in seen_lex
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, c in tok_cands:
            if score >= lexical_threshold:
                lex_list.append(c)
        lex_rank = {c.characteristic_id: idx for idx, c in enumerate(lex_list, start=1)}

        fused: list[tuple[float, float, SkuCharacteristic, str]] = []
        for c in pool:
            cid = c.characteristic_id
            d_cos = dense_scores[cid]
            rrf = 1.0 / (rrf_k + dense_rank[cid])
            method = "rrf_dense"
            if cid in lex_rank and d_cos >= dense_threshold:
                rrf += 1.0 / (rrf_k + lex_rank[cid])
                method = "rrf_lex_dense"
            fused.append((rrf, d_cos, c, method))
        fused.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(c, method, round(d_cos, 4)) for _, d_cos, c, method in fused]

    raise ValueError(f"Unknown alignment mode: {mode}")


def align_with_mode(
    requirement: Requirement,
    characteristics: list[SkuCharacteristic],
    *,
    mode: str = "hybrid_rrf",
    req_vector: list[float] | None = None,
    sku_vectors: dict[str, list[float]] | None = None,
    lexical_threshold: float = 0.5,
    dense_threshold: float = DEFAULT_DENSE_THRESHOLD,
    rrf_k: int = DEFAULT_RRF_K,
) -> tuple[SkuCharacteristic | None, str | None, float | None]:
    """Select the top-1 SKU characteristic under ``lexical``, ``dense`` or ``hybrid_rrf``."""
    if mode == "lexical" or req_vector is None or sku_vectors is None:
        return align(requirement, characteristics, threshold=lexical_threshold)

    ranked = rank_candidates(
        requirement,
        characteristics,
        mode=mode,
        req_vector=req_vector,
        sku_vectors=sku_vectors,
        lexical_threshold=lexical_threshold,
        dense_threshold=dense_threshold,
        rrf_k=rrf_k,
    )
    if not ranked:
        return None, None, None
    chosen, method, cos_score = ranked[0]
    if cos_score < dense_threshold:
        return None, None, None
    return chosen, method, cos_score
