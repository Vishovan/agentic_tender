"""Run every hypothesis that is resolvable today and save all metrics.

Usage:
    python scripts/experiments.py --dataset dataset_zakupki --out runs/experiments

Writes one JSON per hypothesis, a combined summary.json, and a report-ready
REPORT.md. Hypotheses that need GPT-Luna or a dense encoder are recorded as
blocked with their precondition, not silently skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tenderai import experiments as ex
from tenderai.corpus import prepare_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the testable hypotheses.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset_zakupki"))
    parser.add_argument("--out", type=Path, default=Path("runs/experiments"))
    parser.add_argument("--tenders", nargs="*", default=None)
    args = parser.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    corpus = prepare_corpus(args.dataset, args.tenders)
    pairs = ex.evaluate(corpus)

    results = {
        "baseline": ex.baseline(corpus, pairs),
        "H1": ex.h1_encoding_unit(corpus, pairs),
        "H3": ex.h3_hybrid_precision(pairs),
        "H6": ex.h6_reranker_ceiling(corpus),
        "H9": ex.h9_ocr_gating(corpus),
        "H10": ex.h10_unit_normalization(corpus, pairs),
    }
    blocked = ex.blocked()

    for name, payload in results.items():
        _write(out / f"{name}.json", payload)
    _write(out / "blocked.json", blocked)

    summary = {
        "dataset": str(args.dataset),
        "requirements": results["baseline"]["requirements"],
        "skus": results["baseline"]["skus"],
        "tenders_with_requirements": results["baseline"]["tenders_with_requirements"],
        "tenders_in_scope": results["baseline"]["tenders_in_scope"],
        "tested": {name: payload.get("verdict") for name, payload in results.items() if name != "baseline"},
        "blocked": {name: payload["needs"] for name, payload in blocked.items()},
    }
    _write(out / "summary.json", summary)
    (out / "REPORT.md").write_text(_report(results, blocked, args.dataset), encoding="utf-8")

    print(f"Wrote {len(results) + 2} artifacts to {out}")
    for name, payload in results.items():
        if name == "baseline":
            print(f"  baseline: {payload['requirements']} reqs, {payload['pairs']} pairs, "
                  f"{payload['tenders_with_requirements']}/{payload['tenders_in_scope']} tenders with reqs")
        else:
            print(f"  {name}: {payload.get('verdict')}")
    for name, payload in blocked.items():
        print(f"  {name}: blocked - {payload['needs']}")
    return 0


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _report(results: dict, blocked: dict, dataset: Path) -> str:
    base = results["baseline"]
    lines: list[str] = []
    lines.append("# Experiment log: architecture hypotheses")
    lines.append("")
    lines.append(
        f"Dataset `{dataset}`. Baseline is the deterministic pipeline "
        "(lexical align + code check, zero model tokens). Each row holds the input fixed, "
        "changes one component, and reads the metric."
    )
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- Requirements extracted: **{base['requirements']}** "
                 f"across **{base['tenders_with_requirements']}/{base['tenders_in_scope']}** tenders")
    lines.append(f"- SKUs: **{base['skus']}** | requirement x SKU pairs: **{base['pairs']}** "
                 f"(aligned {base['pairs_aligned']}, decidable {base['pairs_decidable']})")
    lines.append(f"- Status distribution: `{base['status_distribution']}`")
    lines.append(f"- Alignment method: `{base['align_method_distribution']}`")
    lines.append(f"- Model tokens (SKU / tenders / matching): **0 / 0 / 0** "
                 "- the cost floor H7/H8 must beat")
    lines.append("")
    lines.append("## Resolved today")
    lines.append("")
    lines.append("| ID | Hypothesis | Headline metric | Verdict |")
    lines.append("|----|------------|-----------------|---------|")
    lines.append(f"| H3 | Hybrid keeps precision where closeness lies | pure-similarity match precision "
                 f"**{results['H3']['pure_similarity_match_precision']}** vs hybrid "
                 f"**{results['H3']['hybrid_match_precision']}**; "
                 f"{results['H3']['false_matches_pure_would_emit']} false matches avoided | "
                 f"**{results['H3']['verdict']}** |")
    lines.append(f"| H10 | Unit/negation normalization is load-bearing | "
                 f"units: {results['H10']['unit_normalization']['false_matches_injected']} false "
                 f"match(es), {results['H10']['unit_normalization']['status_flips_when_units_stripped']} "
                 f"flip(s); negation: {results['H10']['negation_handling']['verdicts_depending_on_negation_or_bool']} "
                 f"verdict(s) depend on it | **{results['H10']['verdict']}** |")
    lines.append(f"| H1 | Encode per characteristic, not per SKU | "
                 f"{results['H1']['spurious_bag_matches']} spurious bag match(es) vs "
                 f"{results['H1']['per_characteristic_matches']} aligned match(es) | "
                 f"**{results['H1']['verdict']}** |")
    lines.append(f"| H6 | Reranker over {results['H6']['sku_count']} SKUs does not pay | "
                 f"recall@{results['H6']['sku_count']} = {results['H6']['recall_at_full_fan_out']}, "
                 f"headroom {results['H6']['recall_headroom_for_reranker']} | "
                 f"**{results['H6']['verdict']}** |")
    lines.append(f"| H9 | OCR only on empty-text PDFs | "
                 f"{results['H9']['pdfs_empty_text_ocr_candidates']}/{results['H9']['pdfs_total']} "
                 f"PDF(s) are OCR candidates | **{results['H9']['verdict']}** |")
    lines.append("")
    for name in ("H3", "H10", "H1", "H6", "H9"):
        payload = results[name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"> {payload['question']}")
        lines.append("")
        lines.append(payload["reading"])
        lines.append("")
    lines.append("## Blocked (need a precondition)")
    lines.append("")
    lines.append("| ID | Hypothesis | Needs |")
    lines.append("|----|------------|-------|")
    for name, payload in blocked.items():
        lines.append(f"| {name} | {payload['question']} | {payload['needs']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
