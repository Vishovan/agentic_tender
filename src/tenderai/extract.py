"""Pull text and tables out of a file, with per-file gated OCR (H9) and RAR unpacking."""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from tenderai.inventory import FileRecord, detect_format
from tenderai.ocr import ocr_pdf


@dataclass
class Cell:
    a1: str
    value: str


@dataclass
class Table:
    sheet: str
    rows: list[list[Cell]]


@dataclass
class Document:
    record: FileRecord
    status: str
    error: str | None = None
    text: str = ""
    tables: list[Table] = field(default_factory=list)
    inner_xmls: list[tuple[str, str]] = field(default_factory=list)
    ocr_used: bool = False
    ocr_ms: int = 0
    ocr_cache: str = "skip"


def extract_document(
    dataset: Path,
    record: FileRecord,
    *,
    enable_ocr: bool = True,
    cache_root: Path | None = None,
) -> Document:
    path = dataset / record.path
    try:
        if record.detected_format == "xlsx":
            tables = _xlsx_tables(path)
            return Document(record, "ok", tables=tables, text=_tables_text(tables))
        if record.detected_format == "xls":
            tables = _xls_tables(path)
            return Document(record, "ok", tables=tables, text=_tables_text(tables))
        if record.detected_format == "docx":
            text, tables = _docx(path)
            return Document(record, "ok", text=text, tables=tables)
        if record.detected_format == "pdf":
            text = _pdf(path)
            if text.strip():
                return Document(record, "ok", text=text)
            if enable_ocr:
                ocr_text, ocr_ms, ocr_cache = ocr_pdf(path, record.sha256, cache_root=cache_root)
                if ocr_text.strip():
                    return Document(
                        record,
                        "ok",
                        text=ocr_text,
                        ocr_used=True,
                        ocr_ms=ocr_ms,
                        ocr_cache=ocr_cache,
                    )
            return Document(record, "empty_text", text="", error="pdf_text_layer_empty")
        if record.detected_format == "xml":
            return Document(record, "ok", text=_xml(path))
        if record.detected_format == "txt":
            return Document(record, "ok", text=path.read_text(encoding="utf-8", errors="replace"))
        if record.detected_format == "doc":
            text = _doc(path)
            return Document(record, "ok", text=text)
        if record.detected_format == "error_json":
            return Document(
                record,
                "gap",
                error="saved_error_response",
                text=path.read_text(encoding="utf-8", errors="replace"),
            )
        if record.detected_format == "rar":
            return _rar(path, record, enable_ocr=enable_ocr, cache_root=cache_root)
        return Document(record, "gap", error=f"unsupported:{record.detected_format}")
    except Exception as exc:  # a broken file must not drop the tender
        return Document(record, "error", error=f"{type(exc).__name__}: {exc}")


def _rar(
    path: Path,
    record: FileRecord,
    *,
    enable_ocr: bool,
    cache_root: Path | None,
) -> Document:
    if shutil.which("bsdtar") is None:
        return Document(record, "gap", error="rar_not_unpacked")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        completed = subprocess.run(
            ["bsdtar", "-xf", str(path), "-C", str(tmp_dir)],
            check=False,
            capture_output=True,
            timeout=60,
        )
        if completed.returncode != 0:
            err = completed.stderr.decode("utf-8", errors="replace").strip()[:300]
            return Document(record, "gap", error=f"rar_unpack_failed:{err}")

        texts: list[str] = []
        tables: list[Table] = []
        inner_xmls: list[tuple[str, str]] = []
        ocr_used = False
        ocr_ms = 0
        for inner_path in sorted(tmp_dir.rglob("*")):
            if not inner_path.is_file() or inner_path.name == ".DS_Store":
                continue
            data = inner_path.read_bytes()
            rel = inner_path.relative_to(tmp_dir).as_posix()
            fmt = detect_format(data)
            inner_rec = FileRecord(
                path=rel,
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
                extension=inner_path.suffix.lower().lstrip("."),
                detected_format=fmt,
                tender_id=record.tender_id,
                sku_id=record.sku_id,
                role=record.role,
            )
            inner_doc = extract_document(
                tmp_dir, inner_rec, enable_ocr=enable_ocr, cache_root=cache_root
            )
            if inner_doc.text.strip():
                texts.append(f"[archive:{rel}]\n{inner_doc.text}")
            for table in inner_doc.tables:
                tables.append(Table(sheet=f"{inner_path.name}:{table.sheet}", rows=table.rows))
            if fmt == "xml":
                inner_xmls.append((rel, data.decode("utf-8", errors="replace")))
            if inner_doc.ocr_used:
                ocr_used = True
                ocr_ms += inner_doc.ocr_ms
        return Document(
            record,
            "ok",
            text="\n\n".join(texts),
            tables=tables,
            inner_xmls=inner_xmls,
            ocr_used=ocr_used,
            ocr_ms=ocr_ms,
        )


def _tables_text(tables: list[Table]) -> str:
    lines: list[str] = []
    for table in tables:
        for row in table.rows:
            lines.append(" | ".join(cell.value for cell in row if cell.value))
    return "\n".join(lines)


def _xlsx_tables(path: Path) -> list[Table]:
    import openpyxl

    # Pass BytesIO so openpyxl does not reject genuine XLSX archives whose
    # filename ends with .xml or .pdf (e.g. tender 09's 02_Извещение.xml).
    workbook = openpyxl.load_workbook(io.BytesIO(path.read_bytes()), data_only=True, read_only=True)
    tables: list[Table] = []
    try:
        for sheet in workbook.worksheets:
            rows: list[list[Cell]] = []
            for row in sheet.iter_rows():
                cells = [
                    Cell(a1=cell.coordinate, value=_text(cell.value))
                    for cell in row
                    if _text(cell.value)
                ]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(Table(sheet=sheet.title, rows=rows))
    finally:
        workbook.close()
    return tables


def _xls_tables(path: Path) -> list[Table]:
    import xlrd

    book = xlrd.open_workbook(path)
    tables: list[Table] = []
    for sheet in book.sheets():
        rows: list[list[Cell]] = []
        for row_index in range(sheet.nrows):
            cells: list[Cell] = []
            for col_index in range(sheet.ncols):
                value = _text(sheet.cell_value(row_index, col_index))
                if not value:
                    continue
                cells.append(Cell(a1=_a1(row_index + 1, col_index + 1), value=value))
            if cells:
                rows.append(cells)
        if rows:
            tables.append(Table(sheet=sheet.name, rows=rows))
    return tables


def _docx(path: Path) -> tuple[str, list[Table]]:
    import docx

    document = docx.Document(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    tables: list[Table] = []
    for index, table in enumerate(document.tables, start=1):
        rows: list[list[Cell]] = []
        for row_index, row in enumerate(table.rows, start=1):
            cells = []
            for col_index, cell in enumerate(row.cells, start=1):
                value = " ".join(cell.text.split())
                if value:
                    cells.append(Cell(a1=_a1(row_index, col_index), value=value))
            if cells:
                rows.append(cells)
        if rows:
            tables.append(Table(sheet=f"table-{index}", rows=rows))
    return "\n".join(paragraphs + [_tables_text(tables)]), tables


def _pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(f"[page {index}]\n{text}" for index, text in enumerate(pages, start=1) if text)


def _xml(path: Path) -> str:
    from xml.etree import ElementTree

    tree = ElementTree.parse(path)
    return " ".join(piece.strip() for piece in tree.getroot().itertext() if piece.strip())


def _doc(path: Path) -> str:
    completed = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(path)],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[:500] or "textutil failed")
    return completed.stdout.decode("utf-8", errors="replace")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace("\u00a0", " ").strip()


def _a1(row: int, col: int) -> str:
    letters = ""
    index = col
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return f"{letters}{row}"
