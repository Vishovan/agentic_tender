"""Call GPT-Luna from code (Cursor SDK) to extract one tender's requirements.

This is the harness-grade path: no manual chat, and the model's token usage is
captured for the mandatory token/cost table.

    export CURSOR_API_KEY=cursor_...        # mint at Cursor dashboard -> Integrations
    python scripts/luna_sdk_extract.py \
        --dataset dataset_zakupki \
        --path "tenders/02_0373200087824002359/05_Техническое_задание_16568494-1.pdf" \
        --tender 0373200087824002359 --out runs/luna
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tenderai.extract import extract_document
from tenderai.inventory import inventory
from tenderai.luna_arm import build_extraction_prompt, parse_luna_payload
from tenderai.luna_client import DEFAULT_MODEL, call_luna, usage_record
from tenderai.models import dump


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract requirements with GPT-Luna via the Cursor SDK.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset_zakupki"))
    parser.add_argument("--path", required=True, help="file path relative to the dataset root")
    parser.add_argument("--tender", required=True, help="tender id, e.g. 0373200087824002359")
    parser.add_argument("--out", type=Path, default=Path("runs/luna"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    record = next((r for r in inventory(args.dataset) if r.path == args.path), None)
    if record is None:
        print(f"not found in dataset: {args.path}", file=sys.stderr)
        return 1
    document = extract_document(args.dataset, record)
    if not document.text.strip():
        print(f"no extractable text ({document.status}); OCR may be needed", file=sys.stderr)
        return 1

    prompt = build_extraction_prompt(args.tender, args.path, document.text)
    try:
        result = call_luna(prompt, model=args.model, cwd=str(Path.cwd()))
    except RuntimeError as exc:
        print(f"GPT-Luna call could not start: {exc}", file=sys.stderr)
        return 2

    try:
        payload = parse_luna_payload(result.text)
    except json.JSONDecodeError:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.tender}_sdk_raw.txt").write_text(result.text, encoding="utf-8")
        print("model did not return JSON; raw text saved for inspection", file=sys.stderr)
        return 3

    positions = payload.get("positions", [])
    n_req = sum(len(p.get("requirements", [])) for p in positions)
    record_usage = usage_record(result, run_id=result.run_id or "luna", stage="llm_extract")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tender}_sdk_requirements.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / f"{args.tender}_sdk_usage.json").write_text(
        json.dumps(dump(record_usage), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"status={result.status} model={result.model} run={result.run_id}")
    print(f"positions={len(positions)} requirements={n_req}")
    print(f"tokens: input={result.input_tokens} output={result.output_tokens} "
          f"cached={result.cached_tokens} reasoning={result.reasoning_tokens} "
          f"total={result.total_tokens} duration_ms={result.duration_ms}")
    print(f"artifacts in {args.out}/ (now run: python scripts/luna_match.py "
          f"--luna {args.out}/{args.tender}_sdk_requirements.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
