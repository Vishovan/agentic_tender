"""Command line for the baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tenderai.llm import LunaConfig
from tenderai.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сопоставить требования закупок с четырьмя SKU.")
    parser.add_argument("command", nargs="?", default="run", choices=["run"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--tenders", default="", help="Номера закупок через запятую. Пусто — весь набор.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--llm", action="store_true", help="Включить извлечение требований через GPT-Luna.")
    args = parser.parse_args(argv)

    llm_cfg = {"enabled": False, "model": "", "api_base": "", "api_key_env": "TENDERAI_API_KEY"}
    if args.config:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        llm_cfg.update(loaded.get("llm") or {})
    if args.llm:
        llm_cfg["enabled"] = True
    config = LunaConfig(
        enabled=bool(llm_cfg["enabled"]),
        model=str(llm_cfg.get("model") or ""),
        api_base=str(llm_cfg.get("api_base") or ""),
        api_key_env=str(llm_cfg.get("api_key_env") or "TENDERAI_API_KEY"),
    )
    if not args.dataset.exists():
        print(f"Нет папки датасета: {args.dataset}")
        return 1
    tender_ids = [item.strip() for item in args.tenders.split(",") if item.strip()]
    summary = run(args.dataset, args.out, tender_ids or None, config)
    print(f"run {summary['run_id']}  requirements {summary['requirements']}  errors {summary['errors']}")
    print(f"{'tender':<22} {'sku':<10} {'status':<10} reason")
    for row in summary["tender_rows"]:
        print(f"{row['tender_id']:<22} {row['sku_id']:<10} {row['status']:<10} {row['reason']}")
    print(f"артефакты: {summary['out']}")
    return 0
