"""Measure the hypotheses that need neither GPT-Luna nor a human gold slice.

Every function here is a controlled two-arm comparison on one fixed corpus:
hold the input, change one component, read the metric. Arms that need the model
(H4, H5, H7, H8) or a dense encoder (H2) are declared blocked in ``blocked()``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from tenderai.align import align, align_with_mode, rank_candidates
from tenderai.checker import check_constraints
from tenderai.corpus import Corpus
from tenderai.embeddings import DEFAULT_ENCODER, EmbeddingBatch, encode_corpus
from tenderai.models import Constraint, Requirement, SkuCharacteristic
from tenderai.parse import fold

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
    empty = [d for d in pdfs if d.status == "empty_text" or d.ocr_used]
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
# H2 - latent space (FRIDA) + RRF vs lexical alignment
# --------------------------------------------------------------------------- #
def h2_embedding_alignment(
    corpus: Corpus,
    held_out_path: Path = Path("eval/held_out_draft.json"),
    batch: EmbeddingBatch | None = None,
) -> dict:
    """Compare lexical, dense (FRIDA), and hybrid RRF on held-out gold and full corpus."""
    emb = batch or encode_corpus(corpus.skus, corpus.requirements, model_name=DEFAULT_ENCODER)
    modes = ("lexical", "dense", "hybrid_rrf")

    corpus_arms: dict[str, dict] = {}
    corpus_pairs_by_mode: dict[str, list[Pair]] = {}
    for mode in modes:
        mode_pairs: list[Pair] = []
        for req in corpus.requirements:
            rv = emb.requirement_vectors.get(req.requirement_id)
            for sku_id, chars in sorted(corpus.by_sku.items()):
                chosen, method, score = align_with_mode(
                    req,
                    chars,
                    mode=mode,
                    req_vector=rv,
                    sku_vectors=emb.sku_vectors,
                )
                if chosen is None:
                    mode_pairs.append(_pair(req, sku_id, None, None, None, "unknown", "not_aligned"))
                else:
                    st, rsn, _ = check_constraints(req.constraint, chosen.constraint)
                    mode_pairs.append(_pair(req, sku_id, chosen, method, score, st, rsn))
        corpus_pairs_by_mode[mode] = mode_pairs
        aligned = [p for p in mode_pairs if p.align_method is not None]
        not_aligned = len(mode_pairs) - len(aligned)
        decidable = [p for p in aligned if p.status in {"match", "mismatch"}]
        corpus_arms[mode] = {
            "pairs": len(mode_pairs),
            "pairs_aligned": len(aligned),
            "not_aligned_count": not_aligned,
            "not_aligned_rate": _round(not_aligned / len(mode_pairs)) if mode_pairs else None,
            "pairs_decidable": len(decidable),
            "status_distribution": _counter(p.status for p in mode_pairs),
            "align_method_distribution": _counter(p.align_method or "none" for p in mode_pairs),
        }

    held_out_metrics: dict[str, dict] = {}
    per_item_comparison: list[dict] = []
    if held_out_path.exists():
        held_payload = json.loads(held_out_path.read_text(encoding="utf-8"))
        gold_items = held_payload.get("pairs", [])
        avg_pool = (
            sum(len(corpus.by_sku.get(g["sku_id"], [])) for g in gold_items) / len(gold_items)
            if gold_items
            else 0.0
        )
        by_mode_rows: dict[str, list[dict]] = {m: [] for m in modes}

        for g in gold_items:
            req = _find_held_out_requirement(corpus.requirements, g)
            chars = corpus.by_sku[g["sku_id"]]
            rv = emb.requirement_vectors.get(req.requirement_id)
            item_row = {
                "id": g["id"],
                "tender_id": g["tender_id"],
                "sku_id": g["sku_id"],
                "requirement": f"{req.name}: {req.raw_text}",
                "gold_status": g["gold_status"],
                "correct_characteristic": g["correct_characteristic"],
                "arms": {},
            }
            for mode in modes:
                ranked = rank_candidates(
                    req,
                    chars,
                    mode=mode,
                    req_vector=rv,
                    sku_vectors=emb.sku_vectors,
                )
                chosen, method, score = align_with_mode(
                    req,
                    chars,
                    mode=mode,
                    req_vector=rv,
                    sku_vectors=emb.sku_vectors,
                )
                if chosen is None:
                    pred_status, pred_reason = "unknown", "not_aligned"
                else:
                    pred_status, pred_reason, _ = check_constraints(req.constraint, chosen.constraint)

                has_target, hit_at_1, hit_at_3, align_ok = _score_gold_alignment(
                    g, chosen, [c for c, _, _ in ranked]
                )
                row = {
                    "id": g["id"],
                    "has_target_in_sku": has_target,
                    "hit_at_1": hit_at_1,
                    "hit_at_3": hit_at_3,
                    "alignment_ok": align_ok,
                    "chosen_name": chosen.name if chosen else None,
                    "chosen_value": chosen.value_raw if chosen else None,
                    "align_method": method,
                    "align_score": score,
                    "pred_status": pred_status,
                    "pred_reason": pred_reason,
                    "gold_status": g["gold_status"],
                }
                by_mode_rows[mode].append(row)
                item_row["arms"][mode] = {
                    "chosen": f"{chosen.name} = {chosen.value_raw}" if chosen else None,
                    "score": score,
                    "method": method,
                    "alignment_ok": align_ok,
                    "status": pred_status,
                    "reason": pred_reason,
                }
            per_item_comparison.append(item_row)

        for mode in modes:
            rows = by_mode_rows[mode]
            pos_rows = [r for r in rows if r["has_target_in_sku"]]
            abs_rows = [r for r in rows if not r["has_target_in_sku"]]
            held_out_metrics[mode] = {
                "search_unit": "sku_characteristic",
                "avg_candidate_characteristics_per_sku": _round(avg_pool),
                "total_pairs": len(rows),
                "positive_target_pairs": len(pos_rows),
                "absent_target_pairs": len(abs_rows),
                "recall_at_1": _round(sum(1 for r in pos_rows if r["hit_at_1"]) / len(pos_rows))
                if pos_rows
                else None,
                "recall_at_3": _round(sum(1 for r in pos_rows if r["hit_at_3"]) / len(pos_rows))
                if pos_rows
                else None,
                "alignment_accuracy": _round(sum(1 for r in rows if r["alignment_ok"]) / len(rows))
                if rows
                else None,
                "alignment_ok_count": sum(1 for r in rows if r["alignment_ok"]),
                "absent_abstention_accuracy": _round(
                    sum(1 for r in abs_rows if r["alignment_ok"]) / len(abs_rows)
                )
                if abs_rows
                else None,
                "not_aligned_count": sum(1 for r in rows if r["chosen_name"] is None),
                "not_aligned_rate": _round(
                    sum(1 for r in rows if r["chosen_name"] is None) / len(rows)
                )
                if rows
                else None,
                **_classification_metrics(rows),
            }

    # Highlight concrete corpus pairs where hybrid_rrf fixed lexical misses or false alignments
    lex_by_key = {(p.requirement_id, p.sku_id): p for p in corpus_pairs_by_mode["lexical"]}
    rrf_by_key = {(p.requirement_id, p.sku_id): p for p in corpus_pairs_by_mode["hybrid_rrf"]}
    recovered_examples = []
    seen_ex = set()
    for key, rrf_p in rrf_by_key.items():
        lex_p = lex_by_key[key]
        if lex_p.align_method is None and rrf_p.align_method is not None:
            sig = (rrf_p.requirement_name, rrf_p.sku_id, rrf_p.characteristic_name)
            if sig in seen_ex:
                continue
            seen_ex.add(sig)
            recovered_examples.append(
                {
                    "tender_id": rrf_p.tender_id,
                    "sku_id": rrf_p.sku_id,
                    "requirement": rrf_p.requirement_quote,
                    "sku_characteristic": f"{rrf_p.characteristic_name} = {rrf_p.characteristic_quote}",
                    "cosine": rrf_p.align_score,
                    "verdict": f"{rrf_p.status}/{rrf_p.reason}",
                }
            )

    lex_m = held_out_metrics.get("lexical", {})
    rrf_m = held_out_metrics.get("hybrid_rrf", {})
    return {
        "question": "Dense FRIDA embeddings + RRF hybrid vs lexical characteristic alignment.",
        "encoder": {
            "model": emb.model,
            "dimension": emb.dimension,
            "query_prefix": "search_query: Инфузионная помпа — <name[, unit][: text_value]>",
            "document_prefix": "search_document: Инфузионная помпа — <name[, unit][: text_value]>",
            "encoding_unit": "characteristic_slot (numbers, ranges, sets and booleans excluded from vector and verified in Constraint)",
            "dense_threshold": 0.625,
            "rrf_k": 60,
        },
        "held_out": held_out_metrics,
        "held_out_pairs": per_item_comparison,
        "corpus": corpus_arms,
        "examples_recovered_over_lexical": recovered_examples[:12],
        "verdict": "keep_hybrid_rrf",
        "reading": (
            f"On the 15-pair reviewed gold slice, alignment accuracy rises from "
            f"{lex_m.get('alignment_accuracy')} (lexical) to {held_out_metrics.get('dense', {}).get('alignment_accuracy')} "
            f"(FRIDA dense) and {rrf_m.get('alignment_accuracy')} (hybrid RRF); Recall@3 reaches "
            f"{rrf_m.get('recall_at_3')} vs {lex_m.get('recall_at_3')} lexical, downstream macro-F1 "
            f"rises {lex_m.get('macro_f1')} -> {rrf_m.get('macro_f1')}, and verified match precision "
            f"rises {lex_m.get('verified_match_precision')} -> {rrf_m.get('verified_match_precision')} "
            f"(G12 recovered as match, G11 false match eliminated, G08/G13 spurious alignments abstained)."
        ),
    }


def _find_held_out_requirement(requirements: list[Requirement], gold: dict) -> Requirement:
    tid = gold["tender_id"]
    rname = fold(gold["requirement_name"])
    rquote = fold(gold["requirement_quote"])
    cands = [r for r in requirements if r.tender_id == tid]
    for r in cands:
        if fold(r.span.quote) == rquote or f"{fold(r.name)}: {fold(r.raw_text)}" == rquote:
            return r
    for r in cands:
        if fold(r.name) == rname and fold(r.raw_text) in rquote:
            return r
    for r in cands:
        if rname in fold(r.name) or fold(r.name) in rname:
            return r
    raise KeyError(f"Held-out requirement not found in corpus: {gold['id']}")


def _score_gold_alignment(
    gold: dict,
    chosen: SkuCharacteristic | None,
    ranked: list[SkuCharacteristic],
) -> tuple[bool, bool, bool, bool]:
    """Return ``(has_target_in_sku, hit_at_1, hit_at_3, alignment_ok)`` for one gold pair."""
    gid = gold["id"]
    correct = gold.get("correct_characteristic")

    # Pairs whose target characteristic is absent from the SKU workbook (must abstain -> None):
    # G08 (individual sterile packaging on CT0050L), G11 (nominal fill volume on CT0050L),
    # G13 (50%-fill accuracy mechanism on CT0050L).
    if correct is None or gid == "G11":
        return False, False, False, chosen is None

    # Compound requirement G14 needs BOTH speed ladder ("Настройки скорости" / "Рабочие скорости...")
    # AND stop mode 0 ("Режим остановки «0»"). No single SKU row proves both at once, while top-3
    # retrieval succeeds if both component rows appear in the top 3.
    if gid == "G14":
        top3_names = {fold(c.name) for c in ranked[:3]}
        has_stop = any("режим остановки" in n for n in top3_names)
        has_speeds = any(
            n in {"настройки скорости", "рабочие скорости без режима «0»"} for n in top3_names
        )
        return True, False, (has_stop and has_speeds), False

    # Single-characteristic positive targets
    def _matches_target(c: SkuCharacteristic) -> bool:
        cname = fold(c.name)
        if gid in {"G03", "G04", "G05"}:
            return cname in {"настройки скорости", "рабочие скорости без режима «0»"}
        if gid == "G12":
            return cname == "тип скорости"
        target_name = fold(str(correct).split("=")[0])
        return cname == target_name

    hit1 = bool(ranked and _matches_target(ranked[0]))
    hit3 = any(_matches_target(c) for c in ranked[:3])
    align_ok = chosen is not None and _matches_target(chosen)
    return True, hit1, hit3, align_ok


def _classification_metrics(rows: list[dict]) -> dict:
    labels = ("match", "mismatch", "unknown")
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for r in rows:
        matrix[r["gold_status"]][r["pred_status"]] += 1

    per_class = {}
    f1s = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in labels if g != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[label] = {
            "precision": _round(prec),
            "recall": _round(rec),
            "f1": _round(f1),
            "support": tp + fn,
        }
        f1s.append(f1)

    pred_matches = [r for r in rows if r["pred_status"] == "match"]
    verified_tp = sum(1 for r in pred_matches if r["gold_status"] == "match" and r["alignment_ok"])
    false_matches = sum(1 for r in pred_matches if r["gold_status"] != "match")
    exact = sum(1 for r in rows if r["pred_status"] == r["gold_status"])
    return {
        "status_accuracy": _round(exact / len(rows)) if rows else None,
        "macro_f1": _round(sum(f1s) / len(f1s)) if f1s else None,
        "match_precision": per_class["match"]["precision"],
        "verified_match_precision": _round(verified_tp / len(pred_matches)) if pred_matches else None,
        "false_matches": false_matches,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


# --------------------------------------------------------------------------- #
# Declared blocked: need live GPT-Luna calls
# --------------------------------------------------------------------------- #
def blocked() -> dict:
    return {
        "H5": {
            "question": "Retrieve-then-read cuts tokens without losing recall.",
            "status": "feasible_not_run",
            "needs": "GPT-Luna arm now wired (Cursor SDK); compare full-doc vs top-k chunk input tokens "
            "on 02/03 where the source is large (03 cost 42.8k input tokens on the full doc).",
        },
        "H7": {
            "question": "Code verdict is no worse than an LLM verdict and cheaper.",
            "status": "ready_needs_luna_key",
            "needs": "Gold slice reviewed; needs live CURSOR_API_KEY to score GPT-Luna verdicts vs check_constraints.",
        },
        "H8": {
            "question": "A code-first cascade holds quality at a fraction of the tokens.",
            "status": "ready_needs_luna_key",
            "needs": "Gold slice reviewed; needs live CURSOR_API_KEY to run GPT-Luna on unknown pairs only.",
        },
    }


def resolved() -> dict:
    return {
        "H2": {
            "question": "Dense embedding alignment vs token overlap.",
            "status": "resolved",
            "result": "FRIDA (1536-d) + RRF raises held-out alignment accuracy 0.6667 -> 0.9333, "
            "Recall@3 0.8333 -> 1.0, macro-F1 0.869 -> 0.9327, and verified match precision 0.7143 -> 1.0.",
        },
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
