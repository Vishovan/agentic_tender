"""Aggregate the GPT-Luna extraction runs into the mandatory token/cost table.

Reads every runs/luna/*_sdk_usage.json (+ its _sdk_summary.json) and writes
runs/luna/h4_token_table.json: per-tender and total tokens by stage, with the
baseline (deterministic arm = 0 requirements, 0 model tokens on these prose
tenders) alongside the GPT-Luna result.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the H4 token table.")
    parser.add_argument("--dir", type=Path, default=Path("runs/luna"))
    parser.add_argument("--tariff-per-1k", type=float, default=None,
                        help="optional USD/1k total tokens to fill the cost column")
    args = parser.parse_args(argv)

    rows = []
    totals = {"input": 0, "output": 0, "cached": 0, "reasoning": 0, "total": 0, "requirements": 0}
    for usage_path in sorted(glob.glob(str(args.dir / "*_sdk_usage.json"))):
        usage = json.loads(Path(usage_path).read_text(encoding="utf-8"))
        summary_path = usage_path.replace("_usage.json", "_summary.json")
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8")) if Path(summary_path).exists() else {}
        inp = _int(usage.get("input_tokens"))
        out = _int(usage.get("output_tokens"))
        cached = _int(usage.get("cached_tokens"))
        reasoning = _int(usage.get("reasoning_tokens"))
        total = inp + out
        row = {
            "tender_id": summary.get("tender_id") or usage.get("run_id"),
            "baseline_requirements": 0,
            "luna_requirements": summary.get("luna_requirements"),
            "pairs_decidable": summary.get("pairs_decidable"),
            "status_distribution": summary.get("status_distribution"),
            "input_tokens": inp,
            "output_tokens": out,
            "cached_tokens": cached,
            "reasoning_tokens": reasoning,
            "total_tokens": total,
            "duration_ms": usage.get("duration_ms"),
        }
        if args.tariff_per_1k is not None:
            row["cost_usd"] = round(total / 1000 * args.tariff_per_1k, 4)
        rows.append(row)
        totals["input"] += inp
        totals["output"] += out
        totals["cached"] += cached
        totals["reasoning"] += reasoning
        totals["total"] += total
        totals["requirements"] += summary.get("luna_requirements") or 0

    table = {
        "experiment": "H4 - GPT-Luna requirement extraction on genuine prose tenders",
        "model": "gpt-5.6-luna",
        "access": "cursor-sdk",
        "stage_note": "All tokens are in the llm_extract stage. prepare_sku and matching are deterministic (0 model tokens).",
        "baseline": {
            "requirements": 0,
            "model_tokens": 0,
            "note": "The table/XML parser extracts 0 requirements from these prose tenders.",
        },
        "per_tender": rows,
        "totals": {
            "requirements_recovered": totals["requirements"],
            "input_tokens": totals["input"],
            "output_tokens": totals["output"],
            "cached_tokens": totals["cached"],
            "reasoning_tokens": totals["reasoning"],
            "total_tokens": totals["total"],
        },
        "cost": (
            {"tariff_usd_per_1k_total_tokens": args.tariff_per_1k,
             "cost_usd": round(totals["total"] / 1000 * args.tariff_per_1k, 4)}
            if args.tariff_per_1k is not None
            else {"status": "gap", "note": "gpt-5.6-luna tariff not provided; cost = total_tokens * tariff once known"}
        ),
    }

    out = args.dir / "h4_token_table.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tenders: {len(rows)} | requirements 0 -> {totals['requirements']} | "
          f"tokens input={totals['input']} output={totals['output']} "
          f"reasoning={totals['reasoning']} total={totals['total']}")
    print(f"wrote {out}")
    return 0


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
