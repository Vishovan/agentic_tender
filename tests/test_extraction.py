from dataclasses import dataclass

from tenderai.eis_xml import has_characteristics, requirements_from_eis_xml
from tenderai.parse import parse_constraint


@dataclass
class _Rec:
    tender_id: str
    path: str = "tenders/07/04_Извещение.xml"


EIS_XML = """
<ns4:characteristicsUsingReferenceInfo>
  <ns4:code>abc</ns4:code>
  <ns4:name>Номинальный объем резервуара</ns4:name>
  <ns4:values><ns4:value>
    <ns4:OKEI><ns2:nationalCode>см[3*]; мл</ns2:nationalCode></ns4:OKEI>
    <ns4:rangeSet><ns4:valueRange>
      <ns4:minMathNotation>greaterOrEqual</ns4:minMathNotation><ns4:min>250</ns4:min>
      <ns4:maxMathNotation>lessOrEqual</ns4:maxMathNotation><ns4:max>299</ns4:max>
    </ns4:valueRange></ns4:rangeSet>
  </ns4:value></ns4:values>
</ns4:characteristicsUsingReferenceInfo>
<ns4:characteristicsUsingReferenceInfo>
  <ns4:code>def</ns4:code>
  <ns4:name>Болюс</ns4:name>
  <ns4:values><ns4:value><ns4:qualityDescription>Нет</ns4:qualityDescription></ns4:value></ns4:values>
</ns4:characteristicsUsingReferenceInfo>
<ns4:characteristicsUsingReferenceInfo>
  <ns4:code>ghi</ns4:code>
  <ns4:name>Скорость инфузии, мл/час</ns4:name>
  <ns4:values><ns4:value><ns4:valueSet><ns4:concreteValue>5</ns4:concreteValue></ns4:valueSet></ns4:value></ns4:values>
</ns4:characteristicsUsingReferenceInfo>
"""


def test_detects_eis_characteristics():
    assert has_characteristics(EIS_XML)
    assert not has_characteristics("<xml><foo/></xml>")


def test_eis_xml_range_quality_and_concrete():
    reqs = requirements_from_eis_xml(_Rec("0372100048826000117"), EIS_XML)
    by_name = {r.name: r.constraint for r in reqs}
    assert len(reqs) == 3

    volume = by_name["Номинальный объем резервуара"]
    assert volume.kind == "range"
    assert (volume.low, volume.high) == (250.0, 299.0)
    assert volume.unit == "ml"

    bolus = by_name["Болюс"]
    assert bolus.kind == "bool" and bolus.value == 0

    speed = by_name["Скорость инфузии, мл/час"]
    assert speed.kind == "number" and speed.value == 5.0 and speed.unit == "ml/h"


def test_eis_span_points_at_characteristic_code():
    reqs = requirements_from_eis_xml(_Rec("0372100048826000117"), EIS_XML)
    assert reqs[0].span.locator == "characteristic/abc"
    assert reqs[0].span.quote.startswith("Номинальный объем")


def test_russian_operator_phrases():
    assert parse_constraint("не более 60", name="Время").operator == "lte"
    assert parse_constraint("не менее 250", unit="мл", name="Объём").operator == "gte"
    assert parse_constraint("не более 2", unit="мл", name="Остаток").value == 2.0


def test_single_inequality_symbols_and_colon_sets():
    upper = parse_constraint("≤ 2", unit="мл", name="Остаточный объём")
    assert upper.kind == "number" and upper.operator == "lte" and upper.value == 2.0
    strict_upper = parse_constraint("< 5", unit="мл")
    assert strict_upper.operator == "lt" and strict_upper.value == 5.0
    lower = parse_constraint("≥ 100", unit="мл")
    assert lower.operator == "gte" and lower.value == 100.0
    ladder = parse_constraint("2:4:6:8:10:12:14", unit="мл/час", name="Скорость инфузии")
    assert ladder.kind == "set"
    assert ladder.values == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    assert ladder.unit == "ml/h"


def test_xlsx_disguised_as_xml_extracts_cleanly(tmp_path):
    import openpyxl

    from tenderai.extract import extract_document
    from tenderai.inventory import inventory

    tdir = tmp_path / "tenders" / "09_0337"
    tdir.mkdir(parents=True)
    xlsx_buf = tmp_path / "temp.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "Наименование показателя"
    ws["B1"] = "Значение"
    ws["A2"] = "Болюс"
    ws["B2"] = "Нет"
    wb.save(xlsx_buf)
    wb.close()

    disguised = tdir / "02_Извещение.xml"
    disguised.write_bytes(xlsx_buf.read_bytes())
    xlsx_buf.unlink()

    records = inventory(tmp_path)
    assert len(records) == 1
    assert records[0].detected_format == "xlsx"
    doc = extract_document(tmp_path, records[0])
    assert doc.status == "ok"
    assert len(doc.tables) == 1


def test_gated_ocr_runs_only_on_empty_pdf(tmp_path, monkeypatch):
    from pypdf import PdfWriter

    from tenderai import extract as ext
    from tenderai.inventory import inventory

    tdir = tmp_path / "tenders" / "04_0128"
    tdir.mkdir(parents=True)
    pdf_path = tdir / "06_empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    calls = []

    def fake_ocr(path, sha, cache_root=None):
        calls.append(sha)
        return "[page 1]\nОбоснование НМЦК", 12, "miss"

    monkeypatch.setattr(ext, "ocr_pdf", fake_ocr)
    rec = inventory(tmp_path)[0]

    disabled = ext.extract_document(tmp_path, rec, enable_ocr=False)
    assert disabled.status == "empty_text"
    assert calls == []

    enabled = ext.extract_document(tmp_path, rec, enable_ocr=True)
    assert enabled.status == "ok"
    assert enabled.ocr_used is True
    assert "Обоснование" in enabled.text
    assert len(calls) == 1
