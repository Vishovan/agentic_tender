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
