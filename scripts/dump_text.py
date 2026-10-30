"""Dump the extracted text of one dataset file (the text a model arm would read).

    python scripts/dump_text.py --dataset dataset_zakupki \
        --path "tenders/02_0373200087824002359/05_Техническое_задание_16568494-1.pdf" \
        --out runs/prose/02_tz.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tenderai.inventory import inventory
from tenderai.extract import extract_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract one file's text.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset_zakupki"))
    parser.add_argument("--path", required=True, help="path relative to the dataset root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    record = next((r for r in inventory(args.dataset) if r.path == args.path), None)
    if record is None:
        print(f"not found: {args.path}", file=sys.stderr)
        return 1
    document = extract_document(args.dataset, record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document.text, encoding="utf-8")
    print(f"format={record.detected_format} status={document.status} "
          f"chars={len(document.text)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
