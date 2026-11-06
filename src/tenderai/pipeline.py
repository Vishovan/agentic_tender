"""Hybrid pipeline harness: prepare once, encode into FRIDA latent space, check every SKU."""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable

from tenderai.aggregate import position_status, tender_status
from tenderai.align import align_with_mode
from tenderai.checker import check_constraints
from tenderai.corpus import prepare_corpus
from tenderai.embeddings import DEFAULT_ENCODER, EmbeddingBatch, encode_corpus
from tenderai.llm import LunaConfig
from tenderai.models import Check, UsageRecord, dump


def run(
    dataset: Path,
    out: Path,
    tender_ids: list[str] | None = None,
    llm: LunaConfig | None = None,
    *,
    align_mode: str = "hybrid_rrf",
    embed_model: str = DEFAULT_ENCODER,
    luna_dir: Path | None = None,
    enable_ocr: bool = True,
    cache_root: Path | None = None,
    encode_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict:
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    out.mkdir(parents=True, exist_ok=True)

    corpus = prepare_corpus(
        dataset,
        tender_ids,
        enable_ocr=enable_ocr,
        cache_root=cache_root,
        luna_dir=luna_dir,
    )
    records = corpus.records
    skus = corpus.skus
    by_sku = corpus.by_sku
    documents = corpus.documents
    requirements = corpus.requirements
    no_requirement_tenders = corpus.no_requirement_tenders
    gap_tenders = corpus.gap_tenders
    prepare_ms = int((time.perf_counter() - started) * 1000)

    errors: list[dict] = []
    ocr_docs = [doc for doc in documents if doc.ocr_used]
    for document in documents:
        if document.status != "ok":
            errors.append(
                {
                    "path": document.record.path,
                    "tender_id": document.record.tender_id,
                    "status": document.status,
                    "error": document.error,
                }
            )

    usage = [
        UsageRecord(
            run_id=run_id,
            stage="prepare_sku",
            model=None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=0,
            cache="miss",
            calls=0,
        ),
        UsageRecord(
            run_id=run_id,
            stage="prepare_tenders",
            model=None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=prepare_ms,
            cache="miss",
            calls=0,
        ),
        UsageRecord(
            run_id=run_id,
            stage="ocr",
            model="apple-vision" if ocr_docs else None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=sum(doc.ocr_ms for doc in ocr_docs),
            cache=ocr_docs[0].ocr_cache if len(ocr_docs) == 1 else ("hit" if ocr_docs else "skip"),
            calls=len(ocr_docs),
        ),
    ]

    usage.extend(_luna_usage_records(run_id, luna_dir, corpus.tenders_in_scope, llm))

    batch: EmbeddingBatch | None = None
    if align_mode in {"dense", "hybrid_rrf"}:
        batch = encode_corpus(
            skus,
            requirements,
            model_name=embed_model,
            cache_root=cache_root,
            encode_fn=encode_fn,
        )
        usage.append(
            UsageRecord(
                run_id=run_id,
                stage="embed",
                model=batch.model,
                input_tokens=0,
                output_tokens=0,
                cached_tokens="unavailable",
                reasoning_tokens="unavailable",
                duration_ms=batch.duration_ms,
                cache=batch.cache,
                calls=len(batch.sku_inputs) + len(batch.requirement_inputs),
            )
        )
    else:
        usage.append(
            UsageRecord(
                run_id=run_id,
                stage="embed",
                model=None,
                input_tokens=0,
                output_tokens=0,
                cached_tokens="unavailable",
                reasoning_tokens="unavailable",
                duration_ms=0,
                cache="skip",
                calls=0,
            )
        )

    checks: list[Check] = []
    match_started = time.perf_counter()
    for requirement in requirements:
        req_vec = batch.requirement_vectors.get(requirement.requirement_id) if batch else None
        for sku_id, characteristics in sorted(by_sku.items()):
            chosen, method, score = align_with_mode(
                requirement,
                characteristics,
                mode=align_mode,
                req_vector=req_vec,
                sku_vectors=batch.sku_vectors if batch else None,
            )
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
            gap_reason = "no_requirements_found" if tender_id in no_requirement_tenders else "coverage_gap"
            status, reason = tender_status(
                [row["status"] for row in rows],
                coverage_gap=tender_id in gap_tenders,
                gap_reason=gap_reason,
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
        cid = characteristic.characteristic_id
        representations.append(
            {
                "object_type": "sku_characteristic",
                "object_id": cid,
                "sku_id": characteristic.sku_id,
                "encoder_model": batch.model if batch else None,
                "encoder_input": batch.sku_inputs[cid]
                if batch
                else f"{characteristic.name}: {characteristic.value_raw} {characteristic.unit_raw}".strip(),
                "features": characteristic.constraint.feature_dict(),
                "vector": batch.sku_vectors[cid] if batch else None,
                "span": dump(characteristic.span),
            }
        )
    for requirement in requirements:
        rid = requirement.requirement_id
        representations.append(
            {
                "object_type": "requirement",
                "object_id": rid,
                "tender_id": requirement.tender_id,
                "position_id": requirement.position_id,
                "encoder_model": batch.model if batch else None,
                "encoder_input": batch.requirement_inputs[rid]
                if batch
                else f"{requirement.name}: {requirement.raw_text}",
                "features": requirement.constraint.feature_dict(),
                "vector": batch.requirement_vectors[rid] if batch else None,
                "span": dump(requirement.span),
            }
        )

    _write(
        out,
        "run.json",
        {
            "run_id": run_id,
            "dataset": str(dataset),
            "align_mode": align_mode,
            "embed_model": batch.model if batch else None,
            "vector_dimension": batch.dimension if batch else None,
            "ocr_enabled": enable_ocr,
            "luna_dir": str(luna_dir) if luna_dir else None,
            "llm_enabled": bool((llm and llm.enabled) or (luna_dir and luna_dir.exists())),
        },
    )
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
        "align_mode": align_mode,
        "embed_model": batch.model if batch else None,
        "vector_dimension": batch.dimension if batch else None,
        "skus": len(by_sku),
        "requirements": len(requirements),
        "checks": len(checks),
        "positions": len(positions),
        "tender_rows": tender_rows,
        "errors": len(errors),
        "out": str(out),
    }
    _write(out, "summary.json", summary)
    return summary


def _luna_usage_records(
    run_id: str,
    luna_dir: Path | None,
    tenders_in_scope: set[str],
    llm: LunaConfig | None,
) -> list[UsageRecord]:
    if luna_dir is not None and luna_dir.exists():
        found: list[UsageRecord] = []
        for tender_id in sorted(tenders_in_scope):
            usage_file = luna_dir / f"{tender_id}_sdk_usage.json"
            if not usage_file.exists():
                continue
            payload = json.loads(usage_file.read_text(encoding="utf-8"))
            found.append(
                UsageRecord(
                    run_id=run_id,
                    stage="llm_extract",
                    model=payload.get("model") or "gpt-5.6-luna",
                    input_tokens=payload.get("input_tokens", "unavailable"),
                    output_tokens=payload.get("output_tokens", "unavailable"),
                    cached_tokens=payload.get("cached_tokens", "unavailable"),
                    reasoning_tokens=payload.get("reasoning_tokens", "unavailable"),
                    duration_ms=int(payload.get("duration_ms") or 0),
                    cache="hit",
                    error=payload.get("error"),
                    calls=int(payload.get("calls") or 1),
                )
            )
        if found:
            return found

    return [
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
    ]


def _write(out: Path, relative: str, payload: object) -> None:
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
