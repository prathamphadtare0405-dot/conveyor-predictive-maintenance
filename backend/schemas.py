"""Response models for the API.

Incoming readings are deliberately *not* described by a strict model: a strict model
would turn one bad field into a rejected request, and the requirement is to keep what
is usable. Incoming data is checked item by item in backend/validation.py instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, PlainSerializer

T = TypeVar("T")


def _iso_z(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds") + "Z"


# Datetimes are stored as naive UTC and always sent with an explicit Z.
UtcDatetime = Annotated[datetime, PlainSerializer(_iso_z, return_type=str)]


class Page(BaseModel, Generic[T]):
    total: int
    limit: int
    offset: int
    items: list[T]


# --- ingest ----------------------------------------------------------------------------


class IssueOut(BaseModel):
    reason: str
    field: str = ""
    detail: str = ""


class ItemProblemOut(BaseModel):
    index: int
    outcome: str
    sensor_id: str | None = None
    timestamp: UtcDatetime | None = None
    issues: list[IssueOut]


class IngestResponse(BaseModel):
    received: int
    accepted: int
    flagged: int
    rejected: int
    duplicates: int
    late: int
    problems: list[ItemProblemOut]
    problems_truncated: bool = False


# --- sensors ---------------------------------------------------------------------------


class ReadingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sensor_id: str
    timestamp: UtcDatetime
    temperature: float | None
    vibration: float | None
    speed: float | None
    load: float | None
    current: float | None
    flags: str
    arrived_late: bool


class SensorOut(BaseModel):
    sensor_id: str
    station_km: float
    status: str
    last_seen_at: UtcDatetime
    seconds_since_last_message: float
    latest_reading: ReadingOut | None


class SensorDetail(SensorOut):
    first_seen_at: UtcDatetime
    readings_stored: int


# --- system ----------------------------------------------------------------------------


class QualityEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sensor_id: str
    timestamp: UtcDatetime | None
    received_at: UtcDatetime
    outcome: str
    reason: str
    field: str
    detail: str
    raw: str


class SensorQuality(BaseModel):
    sensor_id: str
    readings_stored: int
    flagged_readings: int
    rejected: int
    duplicates: int
    late: int
    by_reason: dict[str, int]


class QualityTotals(BaseModel):
    readings_stored: int
    flagged_readings: int
    rejected: int
    duplicates: int
    late: int


class DataQualityOut(BaseModel):
    totals: QualityTotals
    by_reason: dict[str, int]
    per_sensor: list[SensorQuality]
    recent_events: list[QualityEventOut]


class HealthOut(BaseModel):
    status: str
    database: str
    server_time: UtcDatetime
    sensors: int
    readings_stored: int
