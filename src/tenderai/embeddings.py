"""Latent-space encoder for SKU characteristics and tender requirements (H2).

Default encoder: ``ai-forever/FRIDA`` with asymmetric retrieval prefixes
(``search_query:`` for tender requirements and ``search_document:`` for SKU
characteristics) and L2-normalized vectors so inner product equals cosine
similarity. Numbers, units, operators, negations and unknown flags remain in
``Constraint`` features for deterministic checking — similarity only ranks and
gates candidate characteristics.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tenderai.models import Requirement, SkuCharacteristic

DEFAULT_ENCODER = "ai-forever/FRIDA"
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "
DOMAIN_FRAME = "Инфузионная помпа — "

_UNIT_DISPLAY = {
    "ml": "мл",
    "ml/h": "мл/ч",
    "um": "мкм",
    "cm": "см",
    "pct": "%",
    "min": "мин",
}


@dataclass
class EmbeddingBatch:
    model: str
    dimension: int
    sku_vectors: dict[str, list[float]]
    requirement_vectors: dict[str, list[float]]
    sku_inputs: dict[str, str]
    requirement_inputs: dict[str, str]
    duration_ms: int
    cache: str


def sku_encoder_text(characteristic: SkuCharacteristic) -> str:
    """Build the semantic slot string sent to FRIDA for one SKU characteristic.

    Quantitative constraints (numbers, ranges, sets), booleans ("Да"/"Нет") and
    missing markers ("Не указано") are kept strictly in ``Constraint`` so they
    do not cause cross-attribute number/boolean collisions in cosine space.
    Categorical text values (e.g. "Постоянная", "Оксид этилена") refine the slot.
    """
    unit_part = f", {characteristic.unit_raw}" if characteristic.unit_raw else ""
    value_part = (
        f": {characteristic.value_raw}"
        if characteristic.constraint.kind == "text" and characteristic.value_raw
        else ""
    )
    return f"{DOCUMENT_PREFIX}{DOMAIN_FRAME}{characteristic.name}{unit_part}{value_part}".strip()


def requirement_encoder_text(requirement: Requirement) -> str:
    """Build the semantic slot string sent to FRIDA for one tender requirement."""
    unit_ru = _UNIT_DISPLAY.get(requirement.constraint.unit or "", "")
    unit_part = f", {unit_ru}" if unit_ru and unit_ru not in requirement.name.lower() else ""
    value_part = (
        f": {requirement.raw_text}"
        if requirement.constraint.kind == "text"
        and requirement.constraint.text
        and "изменяем" not in requirement.constraint.text
        else ""
    )
    return f"{QUERY_PREFIX}{DOMAIN_FRAME}{requirement.name}{unit_part}{value_part}".strip()


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity for L2-normalized (or arbitrary) dense vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    return dot / (norm_l * norm_r)


_MODEL_CACHE: dict[str, object] = {}


def _load_sentence_transformer(model_name: str, cache_root: Path | None = None):
    if model_name not in _MODEL_CACHE:
        import os

        from sentence_transformers import SentenceTransformer

        hf_dir = (cache_root or Path(".cache")) / "huggingface"
        hf_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_dir.resolve()))
        try:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name, cache_folder=str(hf_dir.resolve()), local_files_only=True)
        except Exception:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name, cache_folder=str(hf_dir.resolve()))
    return _MODEL_CACHE[model_name]


def encode_corpus(
    skus: list[SkuCharacteristic],
    requirements: list[Requirement],
    *,
    model_name: str = DEFAULT_ENCODER,
    cache_root: Path | None = None,
    encode_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> EmbeddingBatch:
    """Encode SKU characteristics and tender requirements into one latent space."""
    sku_inputs = {item.characteristic_id: sku_encoder_text(item) for item in skus}
    req_inputs = {item.requirement_id: requirement_encoder_text(item) for item in requirements}

    all_texts = sorted(set(sku_inputs.values()) | set(req_inputs.values()))
    root = cache_root or Path(".cache")
    slug = "".join(ch if ch.isalnum() else "_" for ch in model_name)
    digest = hashlib.sha256(
        json.dumps({"model": model_name, "texts": all_texts}, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = root / "embeddings" / slug / f"{digest}.json"

    if encode_fn is None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        by_text: dict[str, list[float]] = payload["vectors"]
        dim = int(payload["dimension"])
        return EmbeddingBatch(
            model=model_name,
            dimension=dim,
            sku_vectors={cid: by_text[text] for cid, text in sku_inputs.items()},
            requirement_vectors={rid: by_text[text] for rid, text in req_inputs.items()},
            sku_inputs=sku_inputs,
            requirement_inputs=req_inputs,
            duration_ms=0,
            cache="hit",
        )

    started = time.perf_counter()
    if encode_fn is not None:
        raw_vectors = encode_fn(all_texts)
    else:
        encoder = _load_sentence_transformer(model_name, cache_root=root)
        encoded = encoder.encode(
            all_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        raw_vectors = [[round(float(x), 6) for x in row] for row in encoded]
    duration_ms = int((time.perf_counter() - started) * 1000)

    by_text = dict(zip(all_texts, raw_vectors))
    dim = len(raw_vectors[0]) if raw_vectors else 0

    if encode_fn is None and all_texts:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"model": model_name, "dimension": dim, "vectors": by_text},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return EmbeddingBatch(
        model=model_name,
        dimension=dim,
        sku_vectors={cid: by_text[text] for cid, text in sku_inputs.items()},
        requirement_vectors={rid: by_text[text] for rid, text in req_inputs.items()},
        sku_inputs=sku_inputs,
        requirement_inputs=req_inputs,
        duration_ms=duration_ms,
        cache="miss",
    )


def encode_texts(
    texts: list[str],
    *,
    model_name: str = DEFAULT_ENCODER,
    cache_root: Path | None = None,
    encode_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> tuple[list[list[float]], int, str]:
    """Encode arbitrary prefixed strings (e.g. H5 document chunks) with SHA-256 caching."""
    if not texts:
        return [], 0, "skip"
    unique = sorted(set(texts))
    root = cache_root or Path(".cache")
    slug = "".join(ch if ch.isalnum() else "_" for ch in model_name)
    digest = hashlib.sha256(
        json.dumps({"model": model_name, "texts": unique}, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = root / "embeddings" / slug / f"texts_{digest}.json"

    if encode_fn is None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        by_text = payload["vectors"]
        return [by_text[t] for t in texts], 0, "hit"

    started = time.perf_counter()
    if encode_fn is not None:
        raw_vectors = encode_fn(unique)
    else:
        encoder = _load_sentence_transformer(model_name, cache_root=root)
        encoded = encoder.encode(unique, normalize_embeddings=True, show_progress_bar=False)
        raw_vectors = [[round(float(x), 6) for x in row] for row in encoded]
    duration_ms = int((time.perf_counter() - started) * 1000)

    by_text = dict(zip(unique, raw_vectors))
    if encode_fn is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"model": model_name, "vectors": by_text}, ensure_ascii=False),
            encoding="utf-8",
        )
    return [by_text[t] for t in texts], duration_ms, "miss"
