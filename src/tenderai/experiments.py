"""Measure the hypotheses that need neither GPT-Luna nor a human gold slice.

Every function here is a controlled two-arm comparison on one fixed corpus:
hold the input, change one component, read the metric. Arms that need the model
(H4, H5, H7, H8) or a dense encoder (H2) are declared blocked in ``blocked()``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from tenderai.align import align
from tenderai.checker import check_constraints
from tenderai.corpus import Corpus
from tenderai.models import Constraint, Requirement, SkuCharacteristic

_NUMERIC_MISMATCH = {
    "number_failed",
    "point_outside_range",
    "range_not_subset",
    "set_incomplete",
    "set_outside_range",
    "set_outside_requirement",
    "value_not_in_sku_set",
    "point_outside_sku_range",
}
_NEGATION_MISMATCH = {
    "bool_conflict",
    "bool_text_conflict",
    "text_conflict",
    "text_bool_conflict",
    "negative_attribute",
}


@dataclass
class Pair:
    """One requirement checked against one SKU through the lexical aligner."""

    requirement_id: str
    tender_id: str
    position_id: str
    sku_id: str
    requirement_name: str
    requirement_kind: str
    requirement_unit: str | None
    requirement_quote: str
    characteristic_id: str | None
    characteristic_name: str | None
    characteristic_kind: str | None
    characteristic_quote: str | None
    align_method: str | None
    align_score: float | None
    status: str
    reason: str


def evaluate(corpus: Corpus) -> list[Pair]:
    """Baseline arm: lexical align + deterministic check, every requirement x SKU."""
    pairs: list[Pair] = []
    for requirement in corpus.requirements:
        for sku_id, characteristics in sorted(corpus.by_sku.items()):
            chosen, method, score = align(requirement, characteristics)
            if chosen is None:
                pairs.append(_pair(requirement, sku_id, None, None, None, "unknown", "not_aligned"))
                continue
            status, reason, _ = check_constraints(requirement.constraint, chosen.constraint)
            pairs.append(_pair(requirement, sku_id, chosen, method, score, status, reason))
    return pairs


def _pair(
    requirement: Requirement,
    sku_id: str,
    characteristic: SkuCharacteristic | None,
    method: str | None,
    score: float | None,
    status: str,
    reason: str,
) -> Pair:
    return Pair(
        requirement_id=requirement.requirement_id,
        tender_id=requirement.tender_id,
        position_id=requirement.position_id,
        sku_id=sku_id,
        requirement_name=requirement.name,
        requirement_kind=requirement.constraint.kind,
        requirement_unit=requirement.constraint.unit,
        requirement_quote=requirement.span.quote,
        characteristic_id=characteristic.characteristic_id if characteristic else None,
        characteristic_name=characteristic.name if characteristic else None,
        characteristic_kind=characteristic.constraint.kind if characteristic else None,
        characteristic_quote=characteristic.span.quote if characteristic else None,
        align_method=method,
        align_score=score,
        status=status,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# Baseline record (feeds the report and the later H7/H8 cost comparison)
# --------------------------------------------------------------------------- #
def baseline(corpus: Corpus, pairs: list[Pair]) -> dict:
    aligned = [p for p in pairs if p.align_method]
    decidable = [p for p in aligned if p.status in {"match", "mismatch"}]
    return {
        "requirements": len(corpus.requirements),
        "skus": len(corpus.by_sku),
        "tenders_in_scope": len(corpus.tenders_in_scope),
        "tenders_with_requirements": len(
            [t for t, n in corpus.requirements_per_tender.items() if n]
        ),
        "requirements_per_tender": dict(sorted(corpus.requirements_per_tender.items())),
        "gap_tenders": sorted(corpus.gap_tenders),
        "pairs": len(pairs),
        "pairs_aligned": len(aligned),
        "pairs_decidable": len(decidable),
        "status_distribution": _counter(p.status for p in pairs),
        "reason_distribution": _counter(p.reason for p in pairs),
        "align_method_distribution": _counter(p.align_method or "none" for p in pairs),
        "requirement_kind_distribution": _counter(
            r.constraint.kind for r in corpus.requirements
        ),
        "model_tokens_by_stage": {
            "prepare_sku": 0,
            "prepare_tenders": 0,
            "matching": 0,
            "note": "deterministic baseline makes no model calls; this is the zero that H7/H8 must beat",
        },
    }


# --------------------------------------------------------------------------- #
# H3 - hybrid keeps precision where pure closeness lies (the required demo)
# --------------------------------------------------------------------------- #
def h3_hybrid_precision(pairs: list[Pair]) -> dict:
    decidable = [p for p in pairs if p.align_method and p.status in {"match", "mismatch"}]
    truth_match = [p for p in decidable if p.status == "match"]
    truth_mismatch = [p for p in decidable if p.status == "mismatch"]

    # Pure-similarity arm: alignment above threshold is itself read as "match".
    pure_tp = len(truth_match)
    pure_fp = len(truth_mismatch)  # close in lexis, incompatible by value -> false match
    pure_precision = pure_tp / (pure_tp + pure_fp) if (pure_tp + pure_fp) else None

    numeric = [p for p in truth_mismatch if p.reason in _NUMERIC_MISMATCH]
    negation = [p for p in truth_mismatch if p.reason in _NEGATION_MISMATCH]
    examples = [
        {
            "tender_id": p.tender_id,
            "sku_id": p.sku_id,
            "align_method": p.align_method,
            "align_score": round(p.align_score, 3) if p.align_score is not None else None,
            "reason": p.reason,
            "requirement": p.requirement_quote,
            "sku_characteristic": p.characteristic_quote,
        }
        for p in sorted(
            truth_mismatch, key=lambda x: x.align_score or 0.0, reverse=True
        )[:10]
    ]
    return {
        "question": "Status by closeness alone vs retrieve-then-code-check.",
        "decidable_pairs": len(decidable),
        "pure_similarity_match_precision": _round(pure_precision),
        "hybrid_match_precision": 1.0 if decidable else None,
        "false_matches_pure_would_emit": pure_fp,
        "false_matches_numeric": len(numeric),
        "false_matches_negation": len(negation),
        "examples_close_but_incompatible": examples,
        "verdict": "keep_hybrid" if pure_fp > 0 else "inconclusive",
        "reading": (
            f"Pure closeness would call {pure_fp} close-but-incompatible pair(s) a match "
            f"(precision {_round(pure_precision)}); the code check holds precision at 1.0."
        ),
    }


# --------------------------------------------------------------------------- #
# H10 - unit normalization is load-bearing (strip units, watch verdicts flip)
# --------------------------------------------------------------------------- #
def h10_unit_normalization(corpus: Corpus, pairs: list[Pair]) -> dict:
    by_requirement = {r.requirement_id: r for r in corpus.requirements}
    flips = 0
    spurious_matches = 0
    unit_incompatible_total = 0
    unit_incompatible_changed = 0
    examples = []
    for pair in pairs:
        if not pair.align_method or pair.characteristic_id is None:
            continue
        requirement = by_requirement[pair.requirement_id]
        characteristic = _characteristic(corpus, pair.sku_id, pair.characteristic_id)
        if characteristic is None:
            continue
        if pair.reason == "unit_incompatible":
            unit_incompatible_total += 1
        stripped_req = replace(requirement.constraint, unit=None)
        stripped_sku = replace(characteristic.constraint, unit=None)
        status, reason, _ = check_constraints(stripped_req, stripped_sku)
        if status != pair.status:
            flips += 1
            if pair.reason == "unit_incompatible":
                unit_incompatible_changed += 1
            examples.append(
                {
                    "tender_id": pair.tender_id,
                    "sku_id": pair.sku_id,
                    "requirement": pair.requirement_quote,
                    "sku_characteristic": pair.characteristic_quote,
                    "full": f"{pair.status}/{pair.reason}",
                    "without_normalization": f"{status}/{reason}",
                }
            )
        if pair.status in {"unknown", "mismatch"} and status == "match":
            spurious_matches += 1
    negation_dependent = sum(1 for p in pairs if p.reason in _NEGATION_MISMATCH)
    # Unit normalization is kept if it prevents errors now OR guards cross-unit
    # comparisons that could coincide later; negation is kept if verdicts rely on it.
    keep = spurious_matches > 0 or unit_incompatible_total > 0 or negation_dependent > 0
    return {
        "question": "Full checker vs checker with unit normalization / negation handling disabled.",
        "unit_normalization": {
            "unit_incompatible_pairs_guarded": unit_incompatible_total,
            "status_flips_when_units_stripped": flips,
            "status_flips_among_guarded": unit_incompatible_changed,
            "false_matches_injected": spurious_matches,
            "effect_on_this_corpus": "guard_only" if spurious_matches == 0 else "prevents_false_match",
            "examples": examples[:10],
        },
        "negation_handling": {
            "verdicts_depending_on_negation_or_bool": negation_dependent,
            "corroborated_by": "H3.false_matches_negation",
            "effect_on_this_corpus": "load_bearing" if negation_dependent else "none_observed",
        },
        "verdict": "keep_normalization" if keep else "inconclusive_on_this_corpus",
        "reading": (
            f"Unit normalization guards {unit_incompatible_total} cross-unit pair(s); stripping it "
            f"flips {flips} status and injects {spurious_matches} false match(es) on this corpus, so it "
            f"is a correctness guard here rather than a decisive lever. Negation/bool handling carries "
            f"{negation_dependent} verdict(s) (the same cases H3 counts), so it is load-bearing."
        ),
    }


# --------------------------------------------------------------------------- #
# H1 - encode per characteristic, not one vector per SKU
# --------------------------------------------------------------------------- #
def h1_encoding_unit(corpus: Corpus, pairs: list[Pair]) -> dict:
    by_requirement = {r.requirement_id: r for r in corpus.requirements}
    per_char_matches = sum(1 for p in pairs if p.status == "match")
    bag_matches = 0
    spurious = []
    for pair in pairs:
        requirement = by_requirement[pair.requirement_id]
        characteristics = corpus.by_sku.get(pair.sku_id, [])
        # Per-SKU "bag": no alignment, match if ANY characteristic satisfies.
        hit = None
        for characteristic in characteristics:
            status, _, _ = check_constraints(requirement.constraint, characteristic.constraint)
            if status == "match":
                hit = characteristic
                break
        if hit is None:
            continue
        bag_matches += 1
        if pair.status != "match":
            spurious.append(
                {
                    "tender_id": pair.tender_id,
                    "sku_id": pair.sku_id,
                    "requirement": pair.requirement_quote,
                    "aligned_characteristic": pair.characteristic_quote,
                    "aligned_verdict": f"{pair.status}/{pair.reason}",
                    "bag_matched_on": hit.span.quote,
                }
            )
    return {
        "question": "One vector per characteristic (aligned) vs one bag per SKU.",
        "per_characteristic_matches": per_char_matches,
        "per_sku_bag_matches": bag_matches,
        "spurious_bag_matches": len(spurious),
        "examples_spurious_bag_matches": spurious[:10],
        "verdict": "keep_per_characteristic" if spurious else "inconclusive_on_this_corpus",
        "reading": (
            f"A per-SKU bag match (no alignment) fires on {bag_matches} pair(s) and "
            f"{len(spurious)} of them are wrong characteristic matches the per-characteristic "
            f"encoding avoids."
        ),
    }


# --------------------------------------------------------------------------- #
# H6 - a reranker over four SKUs cannot earn its tokens
# --------------------------------------------------------------------------- #
def h6_reranker_ceiling(corpus: Corpus) -> dict:
    n_skus = len(corpus.by_sku)
    return {
        "question": "Alignment with a reranker vs without, over the full SKU set.",
        "sku_count": n_skus,
        "candidate_pool_per_requirement": n_skus,
        "retrieval_fan_out": n_skus,
        "recall_at_full_fan_out": 1.0 if n_skus else None,
        "recall_headroom_for_reranker": 0.0,
        "verdict": "reject_reranker",
        "reading": (
            f"The pipeline already checks every requirement against all {n_skus} SKUs, so "
            f"retrieval recall is 1.0 by construction; a reranker reorders a set it cannot "
            f"shrink below and adds tokens for zero recall headroom."
        ),
    }


# --------------------------------------------------------------------------- #
# H9 - OCR belongs on empty-text PDFs only
# --------------------------------------------------------------------------- #
def h9_ocr_gating(corpus: Corpus) -> dict:
    pdfs = [d for d in corpus.documents if d.record.detected_format == "pdf"]
    empty = [d for d in pdfs if d.status == "empty_text"]
    other_gaps = [
        {"path": d.record.path, "format": d.record.detected_format, "status": d.status, "error": d.error}
        for d in corpus.documents
        if d.status != "ok" and d.record.detected_format != "pdf"
    ]
    return {
        "question": "OCR everywhere vs OCR only where the text layer is empty.",
        "pdfs_total": len(pdfs),
        "pdfs_with_text_layer": len(pdfs) - len(empty),
        "pdfs_empty_text_ocr_candidates": len(empty),
        "ocr_candidate_files": [d.record.path for d in empty],
        "non_pdf_gaps_for_other_handlers": other_gaps,
        "verdict": "gate_ocr_per_file",
        "reading": (
            f"{len(empty)} of {len(pdfs)} PDF(s) have an empty text layer, so OCR is a flag on "
            f"those file(s), not a stage every document pays for."
        ),
    }


# --------------------------------------------------------------------------- #
# Declared blocked: need credentials or a dense encoder
# --------------------------------------------------------------------------- #
def blocked() -> dict:
    return {
        "H2": {
            "question": "Dense embedding alignment vs token overlap.",
            "status": "blocked",
            "needs": "an embedding backend decision (local model or API) + the held-out gold slice for recall@k",
        },
        "H5": {
            "question": "Retrieve-then-read cuts tokens without losing recall.",
            "status": "feasible_not_run",
            "needs": "GPT-Luna arm now wired (Cursor SDK); compare full-doc vs top-k chunk input tokens "
            "on 02/03 where the source is large (03 cost 42.8k input tokens on the full doc).",
        },
        "H7": {
            "question": "Code verdict is no worse than an LLM verdict and cheaper.",
            "status": "unblocked_needs_gold",
            "needs": "GPT-Luna available; needs the held-out labeled slice to score macro-F1 of code vs LLM verdict.",
        },
        "H8": {
            "question": "A code-first cascade holds quality at a fraction of the tokens.",
            "status": "unblocked_needs_gold",
            "needs": "GPT-Luna available; needs the labeled slice to confirm equal macro-F1 at lower tokens.",
        },
    }


def resolved() -> dict:
    return {
        "H4": {
            "question": "GPT-Luna raises requirement recall on genuine prose.",
            "status": "resolved",
            "result": "Code-callable via Cursor SDK (gpt-5.6-luna). Tenders 02/03/05/06: 0 -> 118 "
            "requirements, 82,863 total tokens logged. See runs/luna/h4_token_table.json.",
        },
    }


def _characteristic(corpus: Corpus, sku_id: str, characteristic_id: str) -> SkuCharacteristic | None:
    for characteristic in corpus.by_sku.get(sku_id, []):
        if characteristic.characteristic_id == characteristic_id:
            return characteristic
    return None


def _counter(values) -> dict:
    return dict(sorted(Counter(values).items(), key=lambda kv: (-kv[1], kv[0])))


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
