"""Baseline harness: prepare once, check every position against every SKU."""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from pathlib import Path

from tenderai.aggregate import position_status, tender_status
from tenderai.align import align
from tenderai.checker import check_constraints
from tenderai.extract import extract_document
from tenderai.inventory import inventory
from tenderai.llm import LunaConfig
from tenderai.models import Check, UsageRecord, dump
from tenderai.requirements import requirements_from_document
from tenderai.sku import load_skus


def run(
    dataset: Path,
    out: Path,
    tender_ids: list[str] | None = None,
    llm: LunaConfig | None = None,
) -> dict:
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    out.mkdir(parents=True, exist_ok=True)
    records = inventory(dataset)
    if tender_ids:
        wanted = set(tender_ids)
        records = [
            record
            for record in records
            if record.role == "sku" or record.tender_id in wanted
        ]
    skus = load_skus(dataset, records)
    by_sku: dict[str, list] = defaultdict(list)
    for characteristic in skus:
        by_sku[characteristic.sku_id].append(characteristic)

    usage = [
        UsageRecord(
            run_id=run_id,
            stage="prepare_sku",
            model=None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=int((time.perf_counter() - started) * 1000),
            cache="miss",
            calls=0,
        )
    ]
    errors: list[dict] = []
    documents = []
    tender_started = time.perf_counter()
    for record in records:
        if record.role != "tender":
            continue
        document = extract_document(dataset, record)
        documents.append(document)
        if document.status != "ok":
            errors.append(
                {
                    "path": record.path,
                    "tender_id": record.tender_id,
                    "status": document.status,
                    "error": document.error,
                }
            )
    usage.append(
        UsageRecord(
            run_id=run_id,
            stage="prepare_tenders",
            model=None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=int((time.perf_counter() - tender_started) * 1000),
            cache="miss",
            calls=0,
        )
    )
    usage.append(
        UsageRecord(
            run_id=run_id,
            stage="llm_extract",
            model=llm.model if llm else None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=0,
            cache="skip",
            error=None,
            calls=0,
        )
    )

    requirements = []
    requirements_by_file: dict[str, int] = defaultdict(int)
    for document in documents:
        if document.status != "ok":
            continue
        extracted = requirements_from_document(document.record, document.tables)
        requirements.extend(extracted)
        requirements_by_file[document.record.path] = len(extracted)

    gap_tenders = {
        document.record.tender_id
        for document in documents
        if document.record.tender_id
        and (
            document.status != "ok"
            or (
                len(document.text) > 400
                and requirements_by_file.get(document.record.path, 0) == 0
            )
        )
    }
    checks: list[Check] = []
    match_started = time.perf_counter()
    for requirement in requirements:
        for sku_id, characteristics in sorted(by_sku.items()):
            chosen, method, score = align(requirement, characteristics)
            if chosen is None:
                checks.append(
                    Check(
                        requirement_id=requirement.requirement_id,
                        sku_id=sku_id,
                        characteristic_id=None,
                        status="unknown",
                        reason="not_aligned",
                        align_method=None,
                        align_score=None,
                        requirement_span=requirement.span,
                        sku_span=None,
                        explanation="Поиск не нашёл характеристику этого SKU. Это не несоответствие.",
                    )
                )
                continue
            status, reason, explanation = check_constraints(requirement.constraint, chosen.constraint)
            checks.append(
                Check(
                    requirement_id=requirement.requirement_id,
                    sku_id=sku_id,
                    characteristic_id=chosen.characteristic_id,
                    status=status,
                    reason=reason,
                    align_method=method,
                    align_score=score,
                    requirement_span=requirement.span,
                    sku_span=chosen.span,
                    explanation=explanation,
                )
            )
    usage.append(
        UsageRecord(
            run_id=run_id,
            stage="matching",
            model=None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=int((time.perf_counter() - match_started) * 1000),
            cache="miss",
            calls=0,
        )
    )

    requirement_by_id = {item.requirement_id: item for item in requirements}
    positions: list[dict] = []
    grouped: dict[tuple[str, str, str], list[Check]] = defaultdict(list)
    for check in checks:
        requirement = requirement_by_id[check.requirement_id]
        grouped[(requirement.tender_id, requirement.position_id, check.sku_id)].append(check)
    for (tender_id, position_id, sku_id), group in sorted(grouped.items()):
        status, reason = position_status(group)
        sample = requirement_by_id[group[0].requirement_id]
        positions.append(
            {
                "tender_id": tender_id,
                "position_id": position_id,
                "position_title": sample.position_title,
                "sku_id": sku_id,
                "status": status,
                "reason": reason,
                "checks": len(group),
            }
        )

    tender_rows = []
    by_tender_sku: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in positions:
        by_tender_sku[(row["tender_id"], row["sku_id"])].append(row)
    seen_tenders = sorted({record.tender_id for record in records if record.tender_id})
    for tender_id in seen_tenders:
        for sku_id in sorted(by_sku):
            rows = by_tender_sku.get((tender_id, sku_id), [])
            status, reason = tender_status(
                [row["status"] for row in rows],
                coverage_gap=tender_id in gap_tenders,
            )
            tender_rows.append(
                {
                    "tender_id": tender_id,
                    "sku_id": sku_id,
                    "status": status,
                    "reason": reason,
                    "positions": len(rows),
                }
            )

    representations = []
    for characteristic in skus:
        representations.append(
            {
                "object_type": "sku_characteristic",
                "object_id": characteristic.characteristic_id,
                "encoder_input": f"{characteristic.name}: {characteristic.value_raw} {characteristic.unit_raw}".strip(),
                "features": characteristic.constraint.feature_dict(),
                "vector": None,
                "span": dump(characteristic.span),
            }
        )
    for requirement in requirements:
        representations.append(
            {
                "object_type": "requirement",
                "object_id": requirement.requirement_id,
                "encoder_input": f"{requirement.name}: {requirement.raw_text}",
                "features": requirement.constraint.feature_dict(),
                "vector": None,
                "span": dump(requirement.span),
            }
        )

    _write(out, "run.json", {"run_id": run_id, "dataset": str(dataset), "llm_enabled": bool(llm and llm.enabled)})
    _write(out, "inventory.json", [dump(record) for record in records])
    _write(out, "prepared/skus.json", [dump(item) for item in skus])
    _write(out, "prepared/requirements.json", [dump(item) for item in requirements])
    _write(out, "representations/structured.json", representations)
    _write(out, "comparisons/checks.json", [dump(item) for item in checks])
    _write(out, "comparisons/positions.json", positions)
    _write(out, "comparisons/tenders.json", tender_rows)
    _write(out, "errors.json", errors)
    with (out / "usage.jsonl").open("w", encoding="utf-8") as handle:
        for record in usage:
            handle.write(json.dumps(dump(record), ensure_ascii=False) + "\n")

    summary = {
        "run_id": run_id,
        "skus": len(by_sku),
        "requirements": len(requirements),
        "checks": len(checks),
        "tender_rows": tender_rows,
        "errors": len(errors),
        "out": str(out),
    }
    _write(out, "summary.json", summary)
    return summary


def _write(out: Path, relative: str, payload: object) -> None:
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
