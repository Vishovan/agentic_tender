"""Encode all SKU characteristics and tender requirements with FRIDA (H2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tenderai.corpus import prepare_corpus
from tenderai.embeddings import DEFAULT_ENCODER, encode_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode corpus into the FRIDA latent space.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset_zakupki"))
    parser.add_argument("--luna-dir", type=Path, default=Path("runs/luna"))
    parser.add_argument("--model", default=DEFAULT_ENCODER)
    args = parser.parse_args(argv)

    corpus = prepare_corpus(args.dataset, luna_dir=args.luna_dir)
    batch = encode_corpus(corpus.skus, corpus.requirements, model_name=args.model)
    print(
        f"model={batch.model} dim={batch.dimension} "
        f"skus={len(batch.sku_vectors)} reqs={len(batch.requirement_vectors)} "
        f"cache={batch.cache} duration_ms={batch.duration_ms}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
