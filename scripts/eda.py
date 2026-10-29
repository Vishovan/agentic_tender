"""Exploratory pass over the dataset. Reads bytes, not extensions."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tenderai.extract import extract_document
from tenderai.inventory import inventory
from tenderai.requirements import requirements_from_document
from tenderai.sku import load_skus


def main(dataset: Path) -> None:
    records = inventory(dataset)
    report: dict = {}

    # 1. Files: extension vs real format.
    ext_fmt = Counter((r.extension, r.detected_format) for r in records if r.role == "tender")
    mismatched = {f"{ext}->{fmt}": n for (ext, fmt), n in ext_fmt.items() if ext != fmt}
    report["files"] = {
        "total": len(records),
        "tender_files": sum(1 for r in records if r.role == "tender"),
        "sku_files": sum(1 for r in records if r.role == "sku"),
        "by_detected_format": dict(Counter(r.detected_format for r in records if r.role == "tender")),
        "extension_lies": dict(sorted(mismatched.items(), key=lambda kv: -kv[1])),
        "error_json_files": [r.path for r in records if r.detected_format == "error_json"],
    }

    # 2. SKU characteristics.
    skus = load_skus(dataset, records)
    by_sku = defaultdict(list)
    for c in skus:
        by_sku[c.sku_id].append(c)
    report["sku"] = {
        sku_id: {
            "characteristics": len(items),
            "knowledge": dict(Counter(i.knowledge for i in items)),
            "constraint_kinds": dict(Counter(i.constraint.kind for i in items)),
            "units": dict(Counter(i.constraint.unit for i in items if i.constraint.unit)),
        }
        for sku_id, items in sorted(by_sku.items())
    }

    # 3. Per-tender extraction reach.
    per_tender: dict = {}
    doc_status = Counter()
    all_requirements = []
    for record in records:
        if record.role != "tender":
            continue
        document = extract_document(dataset, record)
        doc_status[document.status] += 1
        requirements = (
            requirements_from_document(record, document.tables)
            if document.status == "ok"
            else []
        )
        all_requirements.extend(requirements)
        bucket = per_tender.setdefault(
            record.tender_id,
            {"files": 0, "ok": 0, "gap": 0, "error": 0, "text_chars": 0, "table_requirements": 0},
        )
        bucket["files"] += 1
        bucket["text_chars"] += len(document.text)
        bucket["table_requirements"] += len(requirements)
        if document.status == "ok":
            bucket["ok"] += 1
        elif document.status in {"gap", "empty_text"}:
            bucket["gap"] += 1
        elif document.status == "error":
            bucket["error"] += 1
    report["documents_status"] = dict(doc_status)
    report["per_tender"] = {
        tender_id: {
            **stats,
            "requirements_found": stats["table_requirements"] > 0,
        }
        for tender_id, stats in sorted(per_tender.items())
    }

    # 4. Requirement shape across the whole set.
    report["requirements"] = {
        "total": len(all_requirements),
        "tenders_with_table_requirements": sum(
            1 for stats in per_tender.values() if stats["table_requirements"] > 0
        ),
        "tenders_without": sum(
            1 for stats in per_tender.values() if stats["table_requirements"] == 0
        ),
        "constraint_kinds": dict(Counter(r.constraint.kind for r in all_requirements)),
        "units": dict(Counter(r.constraint.unit for r in all_requirements if r.constraint.unit)),
        "mandatory": dict(Counter(r.mandatory for r in all_requirements)),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset_zakupki")
    main(dataset)
