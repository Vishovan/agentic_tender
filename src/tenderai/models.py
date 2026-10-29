"""Records that every later stage reads. Numbers stay here, not in a vector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def dump(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {key: dump(value) for key, value in asdict(obj).items()}
    if isinstance(obj, list):
        return [dump(item) for item in obj]
    if isinstance(obj, dict):
        return {key: dump(value) for key, value in obj.items()}
    return obj


@dataclass
class Span:
    file: str
    locator: str
    quote: str


@dataclass
class Constraint:
    """Parsed value used by the deterministic checker.

    kind:
      unknown — no usable value
      bool — yes/no
      number — one number and a comparison operator
      range — inclusive or exclusive bounds
      set — every listed number must be supported
      text — a short categorical phrase
      reference — "соответствие" without a local value
    """

    kind: str
    operator: str | None = None
    value: float | None = None
    low: float | None = None
    high: float | None = None
    low_inclusive: bool = True
    high_inclusive: bool = True
    values: list[float] = field(default_factory=list)
    text: str | None = None
    unit: str | None = None
    knowledge: str = "stated"

    def feature_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "operator": self.operator,
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "values": self.values,
            "text": self.text,
            "unit": self.unit,
            "knowledge": self.knowledge,
        }


@dataclass
class SkuCharacteristic:
    sku_id: str
    characteristic_id: str
    name: str
    value_raw: str
    unit_raw: str
    knowledge: str
    constraint: Constraint
    span: Span
    product_name: str = ""


@dataclass
class Requirement:
    requirement_id: str
    tender_id: str
    position_id: str
    position_title: str
    name: str
    raw_text: str
    mandatory: bool
    constraint: Constraint
    span: Span


@dataclass
class Check:
    requirement_id: str
    sku_id: str
    characteristic_id: str | None
    status: str
    reason: str
    align_method: str | None
    align_score: float | None
    requirement_span: Span
    sku_span: Span | None
    explanation: str


@dataclass
class UsageRecord:
    run_id: str
    stage: str
    model: str | None
    input_tokens: int | str
    output_tokens: int | str
    cached_tokens: int | str
    reasoning_tokens: int | str
    duration_ms: int
    cache: str
    error: str | None = None
    calls: int = 0
