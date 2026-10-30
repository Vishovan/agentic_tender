"""Code-callable GPT-Luna through the Cursor SDK, with token accounting.

The Cursor SDK runs the model from code (no manual chat) and returns a
``RunResult`` whose ``usage`` carries input / output / cache / reasoning tokens.
That is exactly the per-stage token journal TASK.md requires, so this is the
harness-grade path to GPT-Luna. Access needs one credential: a Cursor API key
(``CURSOR_API_KEY``), mintable by the caller at the Cursor dashboard.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from tenderai.models import UsageRecord

# The Cursor SDK model id is "gpt-5.6-luna"; "gpt-5.6-luna-medium" is the
# subagent reasoning-effort variant of the same model.
DEFAULT_MODEL = "gpt-5.6-luna"


@dataclass
class LunaResult:
    text: str
    status: str
    model: str | None
    run_id: str
    duration_ms: int
    input_tokens: int | str
    output_tokens: int | str
    cached_tokens: int | str
    reasoning_tokens: int | str
    total_tokens: int | str


def call_luna(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    cwd: str | None = None,
) -> LunaResult:
    """One-shot GPT-Luna call. Raises if the SDK or key is unavailable."""
    api_key = api_key or os.environ.get("CURSOR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CURSOR_API_KEY is not set. Mint a key at Cursor dashboard -> Integrations "
            "and `export CURSOR_API_KEY=...` before calling GPT-Luna from code."
        )
    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("cursor-sdk is not installed (`pip install cursor-sdk`).") from exc

    started = time.perf_counter()
    result = Agent.prompt(
        prompt,
        AgentOptions(
            model=model,
            api_key=api_key,
            local=LocalAgentOptions(cwd=cwd or os.getcwd()),
        ),
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    usage = getattr(result, "usage", None)
    return LunaResult(
        text=getattr(result, "result", ""),
        status=str(getattr(result, "status", "")),
        model=model,
        run_id=getattr(result, "id", ""),
        duration_ms=getattr(result, "duration_ms", duration_ms),
        input_tokens=_field(usage, "input_tokens"),
        output_tokens=_field(usage, "output_tokens"),
        cached_tokens=_field(usage, "cache_read_tokens"),
        reasoning_tokens=_field(usage, "reasoning_tokens"),
        total_tokens=_field(usage, "total_tokens"),
    )


def usage_record(result: LunaResult, run_id: str, stage: str) -> UsageRecord:
    """Map a Luna call into the harness usage journal (one call)."""
    return UsageRecord(
        run_id=run_id,
        stage=stage,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_tokens=result.cached_tokens,
        reasoning_tokens=result.reasoning_tokens,
        duration_ms=result.duration_ms,
        cache="miss",
        error=None if result.status in {"finished", "completed", "success"} else result.status,
        calls=1,
    )


def _field(usage: object, name: str) -> int | str:
    if usage is None:
        return "unavailable"
    value = getattr(usage, name, None)
    return value if value is not None else "unavailable"
