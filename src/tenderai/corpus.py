"""Shared preparation: one dataset in, SKUs and requirements out.

Both the baseline pipeline and the experiment harness read the corpus through
this function, so every hypothesis is measured on the exact same extraction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tenderai.eis_xml import has_characteristics, requirements_from_eis_xml
from tenderai.extract import Document, extract_document
from tenderai.inventory import FileRecord, inventory
from tenderai.luna_arm import requirements_from_luna_json
from tenderai.models import Requirement, SkuCharacteristic
from tenderai.parse import fold
from tenderai.requirements import requirements_from_document
from tenderai.sku import load_skus


@dataclass
class Corpus:
    dataset: Path
    records: list[FileRecord]
    skus: list[SkuCharacteristic]
    by_sku: dict[str, list[SkuCharacteristic]]
    documents: list[Document]
    requirements: list[Requirement]
    requirements_by_file: dict[str, int] = field(default_factory=dict)
    requirements_per_tender: dict[str, int] = field(default_factory=dict)
    tenders_in_scope: set[str] = field(default_factory=set)
    broken_tenders: set[str] = field(default_factory=set)
    no_requirement_tenders: set[str] = field(default_factory=set)
    gap_tenders: set[str] = field(default_factory=set)


def prepare_corpus(
    dataset: Path,
    tender_ids: list[str] | None = None,
    *,
    enable_ocr: bool = True,
    cache_root: Path | None = None,
    luna_dir: Path | None = None,
) -> Corpus:
    records = inventory(dataset)
    if tender_ids:
        wanted = set(tender_ids)
        records = [
            record
            for record in records
            if record.role == "sku" or record.tender_id in wanted
        ]

    skus = load_skus(dataset, records)
    by_sku: dict[str, list[SkuCharacteristic]] = defaultdict(list)
    for characteristic in skus:
        by_sku[characteristic.sku_id].append(characteristic)

    documents: list[Document] = []
    for record in records:
        if record.role != "tender":
            continue
        documents.append(
            extract_document(
                dataset,
                record,
                enable_ocr=enable_ocr,
                cache_root=cache_root,
            )
        )

    requirements: list[Requirement] = []
    requirements_by_file: dict[str, int] = defaultdict(int)
    seen_keys: set[tuple[str, str]] = set()
    for document in documents:
        if document.status != "ok":
            continue
        raw_xml = ""
        if document.record.detected_format == "xml":
            raw_xml = (dataset / document.record.path).read_text(
                encoding="utf-8", errors="replace"
            )
        extracted: list[Requirement] = []
        if raw_xml and has_characteristics(raw_xml):
            extracted.extend(requirements_from_eis_xml(document.record, raw_xml))
        else:
            extracted.extend(requirements_from_document(document.record, document.tables))
        for _, inner_xml in document.inner_xmls:
            if has_characteristics(inner_xml):
                extracted.extend(requirements_from_eis_xml(document.record, inner_xml))
        kept: list[Requirement] = []
        for requirement in extracted:
            key = (requirement.position_id, fold(requirement.name))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            kept.append(requirement)
        requirements.extend(kept)
        requirements_by_file[document.record.path] = len(kept)

    tenders_in_scope = {record.tender_id for record in records if record.tender_id}

    # Optional GPT-Luna prose fallback for tenders where table/XML extraction yielded 0.
    if luna_dir is not None and luna_dir.exists():
        current_counts: dict[str, int] = defaultdict(int)
        for req in requirements:
            current_counts[req.tender_id] += 1
        for tender_id in sorted(tenders_in_scope):
            if current_counts.get(tender_id, 0) > 0:
                continue
            luna_file = luna_dir / f"{tender_id}_sdk_requirements.json"
            if not luna_file.exists():
                continue
            luna_reqs = requirements_from_luna_json(luna_file)
            for req in luna_reqs:
                key = (req.position_id, fold(req.name))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                requirements.append(req)
                requirements_by_file[req.span.file] += 1

    requirements_per_tender: dict[str, int] = defaultdict(int)
    for requirement in requirements:
        requirements_per_tender[requirement.tender_id] += 1

    broken_tenders = {
        document.record.tender_id
        for document in documents
        if document.status != "ok" and document.record.tender_id
    }
    no_requirement_tenders = {
        tender_id
        for tender_id in tenders_in_scope
        if requirements_per_tender.get(tender_id, 0) == 0
    }
    gap_tenders = broken_tenders | no_requirement_tenders

    return Corpus(
        dataset=dataset,
        records=records,
        skus=skus,
        by_sku=dict(by_sku),
        documents=documents,
        requirements=requirements,
        requirements_by_file=dict(requirements_by_file),
        requirements_per_tender=dict(requirements_per_tender),
        tenders_in_scope=tenders_in_scope,
        broken_tenders=broken_tenders,
        no_requirement_tenders=no_requirement_tenders,
        gap_tenders=gap_tenders,
    )
