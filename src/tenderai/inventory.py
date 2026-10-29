"""Detect the real file type. Extensions in this dataset often lie."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileRecord:
    path: str
    sha256: str
    bytes: int
    extension: str
    detected_format: str
    tender_id: str | None
    sku_id: str | None
    role: str


def inventory(dataset: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in sorted(dataset.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {".DS_Store", "manifest.json", "README.md"}:
            continue
        data = path.read_bytes()
        relative = path.relative_to(dataset).as_posix()
        detected = detect_format(data)
        tender_id, sku_id, role = _identity(path, dataset)
        records.append(
            FileRecord(
                path=relative,
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
                extension=path.suffix.lower().lstrip("."),
                detected_format=detected,
                tender_id=tender_id,
                sku_id=sku_id,
                role=role,
            )
        )
    return records


def detect_format(data: bytes) -> str:
    if data.startswith(b"PK\x03\x04"):
        if b"word/" in data[:4000] or b"word/" in data:
            return "docx"
        if b"xl/" in data[:4000] or b"xl/" in data:
            return "xlsx"
        return "zip"
    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"Rar!"):
        return "rar"
    if data.startswith(b"\xd0\xcf\x11\xe0"):
        if _ole_name(data, "WordDocument"):
            return "doc"
        if _ole_name(data, "Workbook"):
            return "xls"
        return "ole"
    stripped = data.lstrip()
    if stripped.startswith(b"{") and len(data) < 2000:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and any(key in payload for key in ("error", "message", "status")):
            return "error_json"
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<"):
        return "xml"
    return "txt"


def _ole_name(data: bytes, name: str) -> bool:
    return name.encode() in data or name.encode("utf-16le") in data


def _identity(path: Path, dataset: Path) -> tuple[str | None, str | None, str]:
    relative = path.relative_to(dataset)
    parts = relative.parts
    if parts and parts[0] == "SKU":
        return None, path.stem, "sku"
    if len(parts) >= 2 and parts[0] == "tenders":
        folder = parts[1]
        tender_id = folder.split("_", 1)[1] if "_" in folder else folder
        return tender_id, None, "tender"
    return None, None, "other"
