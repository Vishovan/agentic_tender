"""Match GPT-Luna's extracted requirements for one tender against the SKUs.

    python scripts/luna_match.py --luna runs/luna/02_requirements.json \
        --dataset dataset_zakupki --out runs/luna

Baseline (table/XML parser) extracts 0 requirements from this prose tender, so
every requirement here is recall the deterministic arm could not reach.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tenderai.aggregate import position_status
from tenderai.align import align
from tenderai.checker import check_constraints
from tenderai.corpus import prepare_corpus
from tenderai.luna_arm import requirements_from_luna_json
from tenderai.models import Check, dump


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Match Luna requirements to SKUs.")
    parser.add_argument("--luna", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_zakupki"))
    parser.add_argument("--out", type=Path, default=Path("runs/luna"))
    args = parser.parse_args(argv)

    corpus = prepare_corpus(args.dataset)
    requirements = requirements_from_luna_json(args.luna)
    tender_id = requirements[0].tender_id if requirements else ""

    checks = []
    check_objects: dict[tuple[str, str], list[Check]] = defaultdict(list)
    for requirement in requirements:
        for sku_id, characteristics in sorted(corpus.by_sku.items()):
            chosen, method, score = align(requirement, characteristics)
            if chosen is None:
                status, reason, explanation = "unknown", "not_aligned", "Поиск не нашёл характеристику SKU."
                sku_span = None
            else:
                status, reason, explanation = check_constraints(requirement.constraint, chosen.constraint)
                sku_span = chosen.span
            check_objects[(requirement.position_id, sku_id)].append(
                Check(
                    requirement_id=requirement.requirement_id,
                    sku_id=sku_id,
                    characteristic_id=chosen.characteristic_id if chosen else None,
                    status=status,
                    reason=reason,
                    align_method=method,
                    align_score=score,
                    requirement_span=requirement.span,
                    sku_span=sku_span,
                    explanation=explanation,
                )
            )
            checks.append(
                {
                    "requirement_id": requirement.requirement_id,
                    "position_id": requirement.position_id,
                    "name": requirement.name,
                    "sku_id": sku_id,
                    "status": status,
                    "reason": reason,
                    "align_method": method,
                    "requirement_kind": requirement.constraint.kind,
                    "requirement_quote": requirement.span.quote,
                    "sku_characteristic": sku_span.quote if sku_span else None,
                }
            )

    position_rows = []
    for (position_id, sku_id), group in sorted(check_objects.items()):
        status, reason = position_status(group)
        position_rows.append({"position_id": position_id, "sku_id": sku_id, "status": status, "reason": reason})

    kinds = Counter(r.constraint.kind for r in requirements)
    parsed = sum(1 for r in requirements if r.constraint.kind not in {"unknown", "reference"})
    decidable = [c for c in checks if c["status"] in {"match", "mismatch"}]

    summary = {
        "tender_id": tender_id,
        "arm": "gpt-luna + code checker",
        "baseline_requirements_for_this_tender": 0,
        "luna_requirements": len(requirements),
        "positions": len({r.position_id for r in requirements}),
        "parsed_to_a_checkable_constraint": parsed,
        "requirement_kind_distribution": dict(kinds),
        "pairs": len(checks),
        "pairs_decidable": len(decidable),
        "status_distribution": dict(Counter(c["status"] for c in checks)),
        "reason_distribution": dict(Counter(c["reason"] for c in checks)),
        "match_examples": [c for c in checks if c["status"] == "match"][:8],
        "mismatch_examples": [c for c in checks if c["status"] == "mismatch"][:8],
        "note_usage": "GPT-Luna reached via Cursor subagent; per-call token/cost not exposed to the harness -> record as unavailable until a code-callable Luna endpoint is wired.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.luna.stem.replace("_requirements", "")
    _write(args.out / f"{stem}_requirements_parsed.json", [dump(r) for r in requirements])
    _write(args.out / f"{stem}_checks.json", checks)
    _write(args.out / f"{stem}_positions.json", position_rows)
    _write(args.out / f"{stem}_summary.json", summary)

    print(f"tender {tender_id}: {len(requirements)} requirements "
          f"(baseline 0), {parsed} parsed to a constraint")
    print(f"  kinds: {dict(kinds)}")
    print(f"  pairs {len(checks)}, decidable {len(decidable)}, "
          f"status {summary['status_distribution']}")
    return 0


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
