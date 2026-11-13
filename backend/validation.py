"""Validation of incoming readings.

The rule for bad input: decide per reading, never silently.

* **Rejected**: the reading cannot be placed (no sensor id, no usable timestamp) or
  carries no usable signal at all. It is not stored; the reason is logged.
* **Flagged**: the reading can be placed and at least one signal is usable. The bad
  signals are stored as NULL, the rest are kept, and each problem is logged.

A value is never "repaired" here. A NaN, an impossible value or text in a numeric field
becomes NULL, and the analysis code decides how to fill the gap (see ml/preprocessing.py).
The single exception is text that is a plain number ("45.2"): the number is used and the
wrong type is still logged.

This module has no database access so the same checks can be reused on CSV data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.config import SIGNAL_NAMES, SIGNALS


@dataclass
class Issue:
    reason: str
    field: str = ""
    detail: str = ""

    def flag(self) -> str:
        return f"{self.reason}:{self.field}" if self.field else self.reason


@dataclass
class CleanReading:
    sensor_id: str
    station_km: float | None
    timestamp: datetime  # naive UTC
    values: dict[str, float | None]
    issues: list[Issue] = field(default_factory=list)


def to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def parse_timestamp(raw: Any) -> datetime | None:
    """Parse an ISO 8601 string. A timestamp without an offset is taken as UTC."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return to_naive_utc(datetime.fromisoformat(raw.strip()))
    except ValueError:
        return None


def _short(raw: Any) -> str:
    text = repr(raw)
    return text if len(text) <= 60 else text[:57] + "..."


def clean_number(raw: Any) -> tuple[float | None, str, str]:
    """Turn a raw JSON value into a finite float, or explain why not.

    Returns (value, reason, detail); reason is "" when the value is fine.
    """
    if raw is None:
        return None, "missing_field", "value is null"
    if isinstance(raw, bool):
        return None, "wrong_type", f"boolean {raw!r} where a number is expected"
    if isinstance(raw, (int, float)):
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            return None, "nan", f"value is {_short(raw)}"
        return value, "", ""
    if isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return None, "wrong_type", f"text {_short(raw)} where a number is expected"
        if math.isnan(value) or math.isinf(value):
            return None, "nan", f"text {_short(raw)}"
        return value, "wrong_type", f"number sent as text {_short(raw)}; value used"
    return None, "wrong_type", f"{type(raw).__name__} where a number is expected"


def clean_signal(name: str, item: dict[str, Any]) -> tuple[float | None, Issue | None]:
    """Validate one signal of a reading against its type and physical range."""
    if name not in item:
        return None, Issue("missing_field", name, "field is absent")
    value, reason, detail = clean_number(item[name])
    if value is not None:
        spec = SIGNALS[name]
        if not spec.hard_min <= value <= spec.hard_max:
            return None, Issue(
                "out_of_range", name, f"{value:g} is outside the possible range {spec.hard_min:g} to {spec.hard_max:g} {spec.unit}"
            )
    return value, (Issue(reason, name, detail) if reason else None)


def validate_item(item: Any) -> tuple[CleanReading | None, list[Issue]]:
    """Validate one reading.

    Returns (reading, issues). `reading` is None when the item is rejected, and then
    `issues` holds the reason. Otherwise `issues` lists what was flagged (possibly nothing).
    """
    if not isinstance(item, dict):
        return None, [Issue("not_an_object", "", f"expected a JSON object, got {type(item).__name__}")]

    sensor_id = item.get("sensor_id")
    if not isinstance(sensor_id, str) or not sensor_id.strip():
        return None, [Issue("missing_sensor_id", "sensor_id", f"got {_short(sensor_id)}")]
    sensor_id = sensor_id.strip()
    if len(sensor_id) > 32:
        return None, [Issue("bad_sensor_id", "sensor_id", "longer than 32 characters")]

    if "timestamp" not in item or item["timestamp"] is None:
        return None, [Issue("missing_timestamp", "timestamp", "a reading without a time cannot be placed in the history")]
    timestamp = parse_timestamp(item["timestamp"])
    if timestamp is None:
        return None, [Issue("bad_timestamp", "timestamp", f"not an ISO 8601 time: {_short(item['timestamp'])}")]

    issues: list[Issue] = []

    station_km: float | None = None
    if "station_km" in item:
        station_km, reason, detail = clean_number(item["station_km"])
        if station_km is not None and station_km < 0:
            station_km, reason, detail = None, "out_of_range", "station_km cannot be negative"
        if station_km is None:
            issues.append(Issue(reason, "station_km", detail))
    # An absent station_km is not a fault for a known sensor; crud decides.

    values: dict[str, float | None] = {}
    for name in SIGNAL_NAMES:
        values[name], issue = clean_signal(name, item)
        if issue is not None:
            issues.append(issue)

    if all(v is None for v in values.values()):
        return None, [Issue("no_usable_signals", "", "; ".join(f"{i.field}: {i.reason}" for i in issues))]

    return CleanReading(sensor_id, station_km, timestamp, values, issues), issues
