"""Optional GPT-Luna extraction. The model does not assign match status."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from tenderai.models import UsageRecord

_PROMPT = """Извлеки технические требования к изделию из фрагмента закупки.
Верни JSON-массив объектов с полями name, raw_text, quote, operator, value, unit.
quote — дословный кусок фрагмента. Не оценивай соответствие товару.
Не включай цену, срок поставки, адрес, срок годности партии и документы участника.
Фрагмент:
"""


class LunaConfig:
    def __init__(self, enabled: bool, model: str, api_base: str, api_key_env: str) -> None:
        self.enabled = enabled
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env


def extract_quotes(
    config: LunaConfig,
    chunk: str,
    run_id: str,
) -> tuple[list[dict], UsageRecord]:
    if not config.enabled:
        return [], UsageRecord(
            run_id=run_id,
            stage="llm_extract",
            model=config.model or None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=0,
            cache="skip",
            calls=0,
        )
    key = os.environ.get(config.api_key_env, "")
    if not config.model or not config.api_base or not key:
        return [], UsageRecord(
            run_id=run_id,
            stage="llm_extract",
            model=config.model or None,
            input_tokens="unavailable",
            output_tokens="unavailable",
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=0,
            cache="miss",
            error="llm_not_configured",
            calls=0,
        )
    body = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Ты извлекай только то, что есть в тексте."},
            {"role": "user", "content": _PROMPT + chunk[:12000]},
        ],
    }
    request = urllib.request.Request(
        f"{config.api_base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return [], UsageRecord(
            run_id=run_id,
            stage="llm_extract",
            model=config.model,
            input_tokens="unavailable",
            output_tokens="unavailable",
            cached_tokens="unavailable",
            reasoning_tokens="unavailable",
            duration_ms=0,
            cache="miss",
            error=str(exc),
            calls=1,
        )
    usage = payload.get("usage") or {}
    content = payload["choices"][0]["message"]["content"]
    quotes = [item for item in _parse_array(content) if item.get("quote") and item["quote"] in chunk]
    return quotes, UsageRecord(
        run_id=run_id,
        stage="llm_extract",
        model=config.model,
        input_tokens=usage.get("prompt_tokens", "unavailable"),
        output_tokens=usage.get("completion_tokens", "unavailable"),
        cached_tokens=_nested(usage, "prompt_tokens_details", "cached_tokens"),
        reasoning_tokens=_nested(usage, "completion_tokens_details", "reasoning_tokens"),
        duration_ms=0,
        cache="miss",
        calls=1,
    )


def _nested(usage: dict, parent: str, child: str) -> int | str:
    details = usage.get(parent)
    if not isinstance(details, dict) or child not in details:
        return "unavailable"
    return details[child]


def _parse_array(content: str) -> list[dict]:
    start = content.find("[")
    end = content.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]
