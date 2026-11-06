"""Command line for the TenderAI matching pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tenderai.embeddings import DEFAULT_ENCODER
from tenderai.llm import LunaConfig
from tenderai.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сопоставить требования закупок с четырьмя SKU.")
    parser.add_argument("command", nargs="?", default="run", choices=["run"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/full"))
    parser.add_argument("--tenders", default="", help="Номера закупок через запятую. Пусто — весь набор.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--llm", action="store_true", help="Включить извлечение требований через GPT-Luna.")
    parser.add_argument(
        "--luna-dir",
        type=Path,
        default=Path("runs/luna"),
        help="Папка с кешированными результатами GPT-Luna для прозовых закупок.",
    )
    parser.add_argument(
        "--no-luna-cache",
        action="store_true",
        help="Игнорировать кешированные требования GPT-Luna (только таблицы/XML).",
    )
    parser.add_argument(
        "--align-mode",
        choices=["hybrid_rrf", "dense", "lexical"],
        default="hybrid_rrf",
        help="Режим выравнивания характеристик (по умолчанию hybrid_rrf на FRIDA).",
    )
    parser.add_argument("--embed-model", default=DEFAULT_ENCODER)
    parser.add_argument("--no-ocr", action="store_true", help="Отключить точечный OCR для пустых PDF.")
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
    luna_dir = None if args.no_luna_cache else args.luna_dir
    summary = run(
        args.dataset,
        args.out,
        tender_ids or None,
        config,
        align_mode=args.align_mode,
        embed_model=args.embed_model,
        luna_dir=luna_dir,
        enable_ocr=not args.no_ocr,
    )
    print(
        f"run {summary['run_id']}  mode {summary['align_mode']}  "
        f"requirements {summary['requirements']}  errors {summary['errors']}"
    )
    print(f"{'tender':<22} {'sku':<10} {'status':<10} reason")
    for row in summary["tender_rows"]:
        print(f"{row['tender_id']:<22} {row['sku_id']:<10} {row['status']:<10} {row['reason']}")
    print(f"артефакты: {summary['out']}")
    return 0
