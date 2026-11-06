"""The experiment harness must measure what the hypotheses claim."""

from pathlib import Path

from tenderai import experiments as ex
from tenderai.corpus import Corpus
from tenderai.models import Requirement, SkuCharacteristic, Span
from tenderai.parse import parse_constraint


def _sku(name: str, value: str, unit: str = "") -> SkuCharacteristic:
    return SkuCharacteristic(
        sku_id="S1",
        characteristic_id=f"S1:{name}:{value}",
        name=name,
        value_raw=value,
        unit_raw=unit,
        knowledge="stated",
        constraint=parse_constraint(value, unit=unit, name=name),
        span=Span("sku.xlsx", "A1", f"{name}={value}"),
    )


def _req(rid: str, name: str, raw: str, unit: str = "") -> Requirement:
    return Requirement(
        requirement_id=rid,
        tender_id="T1",
        position_id="T1:1:pump",
        position_title="Насос",
        name=name,
        raw_text=raw,
        mandatory=True,
        constraint=parse_constraint(raw, unit=unit, name=name),
        span=Span("ooz.xlsx", "A1", f"{name}: {raw}"),
    )


def _corpus() -> Corpus:
    skus = [
        _sku("Болюс", "Да"),
        _sku("Номинальный объем резервуара", "250", "мл"),
        _sku("Защита от УФ-лучей", "Нет"),
    ]
    requirements = [
        _req("r_bolus", "Болюс", "Нет"),  # negation: req Нет vs sku Да -> mismatch
        _req("r_vol_hi", "Номинальный объем резервуара", "≥ 400", "мл"),  # numeric: 250 fails
        _req("r_vol_lo", "Номинальный объем резервуара", "≥ 200", "мл"),  # numeric: 250 passes
    ]
    return Corpus(
        dataset=Path("mem"),
        records=[],
        skus=skus,
        by_sku={"S1": skus},
        documents=[],
        requirements=requirements,
    )


def test_evaluate_produces_canon_aligned_verdicts():
    corpus = _corpus()
    pairs = ex.evaluate(corpus)
    by_req = {p.requirement_id: p for p in pairs}
    assert by_req["r_bolus"].status == "mismatch"
    assert by_req["r_bolus"].reason == "bool_conflict"
    assert by_req["r_vol_hi"].status == "mismatch"
    assert by_req["r_vol_lo"].status == "match"


def test_h3_pure_similarity_loses_precision():
    corpus = _corpus()
    result = ex.h3_hybrid_precision(ex.evaluate(corpus))
    # one match, two close-but-incompatible mismatches
    assert result["false_matches_pure_would_emit"] == 2
    assert result["false_matches_numeric"] == 1
    assert result["false_matches_negation"] == 1
    assert result["pure_similarity_match_precision"] < 1.0
    assert result["hybrid_match_precision"] == 1.0
    assert result["verdict"] == "keep_hybrid"


def test_h1_bag_match_fires_on_wrong_characteristic():
    corpus = _corpus()
    result = ex.h1_encoding_unit(corpus, ex.evaluate(corpus))
    # "Болюс: Нет" (bool 0) should wrongly bag-match the UV characteristic ("Нет").
    assert result["spurious_bag_matches"] >= 1
    assert result["verdict"] == "keep_per_characteristic"


def test_h10_counts_negation_dependence():
    corpus = _corpus()
    result = ex.h10_unit_normalization(corpus, ex.evaluate(corpus))
    assert result["negation_handling"]["verdicts_depending_on_negation_or_bool"] >= 1
    assert result["verdict"] == "keep_normalization"


def test_h6_rejects_reranker_for_small_sku_set():
    result = ex.h6_reranker_ceiling(_corpus())
    assert result["sku_count"] == 1
    assert result["recall_headroom_for_reranker"] == 0.0
    assert result["verdict"] == "reject_reranker"


def test_blocked_hypotheses_name_their_precondition():
    blocked = ex.blocked()
    assert set(blocked) == {"H5", "H7", "H8"}
    assert all("needs" in payload for payload in blocked.values())


def test_h2_and_h4_are_resolved():
    resolved = ex.resolved()
    assert {"H2", "H4"} <= set(resolved)
    assert resolved["H2"]["status"] == "resolved"
    assert resolved["H4"]["status"] == "resolved"


def test_h2_embedding_alignment_recovers_semantic_synonym(tmp_path: Path):
    from tenderai.embeddings import EmbeddingBatch

    skus = [
        _sku("Тип скорости", "Постоянная"),
        _sku("Максимальный объём заполнения", "300", "мл"),
    ]
    reqs = [
        _req("r_speed_ctrl", "Регулирование скорости инфузии", "нет"),
        _req("r_nom_fill", "Номинальный объем заполнения", "≥ 300 и ≤ 399", "мл"),
    ]
    corpus = Corpus(
        dataset=Path("mem"),
        records=[],
        skus=skus,
        by_sku={"S1": skus},
        documents=[],
        requirements=reqs,
    )
    batch = EmbeddingBatch(
        model="mock-frida",
        dimension=2,
        sku_vectors={
            skus[0].characteristic_id: [1.0, 0.0],
            skus[1].characteristic_id: [0.0, 1.0],
        },
        requirement_vectors={
            "r_speed_ctrl": [0.9, 0.4359],
            "r_nom_fill": [0.1, 0.995],
        },
        sku_inputs={c.characteristic_id: c.name for c in skus},
        requirement_inputs={r.requirement_id: r.name for r in reqs},
        duration_ms=1,
        cache="hit",
    )
    result = ex.h2_embedding_alignment(corpus, held_out_path=tmp_path / "none.json", batch=batch)
    assert result["corpus"]["lexical"]["status_distribution"].get("match", 0) == 1  # false match on max fill!
    assert result["corpus"]["hybrid_rrf"]["status_distribution"] == {"match": 1, "unknown": 1}
    assert result["verdict"] == "keep_hybrid_rrf"
