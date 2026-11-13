"""All database work. Route handlers call these functions and nothing else touches the tables."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import case, func, select, update
from sqlalchemy.orm import Session

from backend.config import SIGNAL_NAMES, settings
from backend.models import Anomaly, Prediction, QualityEvent, Reading, Sensor
from backend.validation import CleanReading, Issue, validate_item

STATUS_NORMAL = "normal"
STATUS_ABNORMAL = "abnormal"
STATUS_PREDICTED_FAILURE = "predicted failure"
STATUS_OFFLINE = "offline"


def utcnow() -> datetime:
    """Current time as naive UTC, the form used in every table."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------------------


@dataclass
class ItemProblem:
    index: int
    outcome: str  # rejected | flagged | duplicate
    sensor_id: str | None
    timestamp: datetime | None
    issues: list[Issue]


@dataclass
class IngestResult:
    received: int = 0
    accepted: int = 0  # stored, nothing wrong
    flagged: int = 0  # stored, with at least one problem logged
    rejected: int = 0  # not stored
    duplicates: int = 0  # already present, not stored again
    late: int = 0  # stored, but older than a reading already held (also counted in accepted/flagged)
    problems: list[ItemProblem] = field(default_factory=list)
    # sensor id -> data timestamps stored by this call; the analysis step works from this
    stored: dict[str, list[datetime]] = field(default_factory=dict)

    @property
    def nothing_usable(self) -> bool:
        return self.received > 0 and self.rejected == self.received


def _raw_text(raw: Any) -> str:
    try:
        text = json.dumps(raw, default=str)
    except (TypeError, ValueError):
        text = repr(raw)
    return text[:500]


def _log(
    db: Session, now: datetime, outcome: str, issue: Issue, raw: Any, sensor_id: str | None, timestamp: datetime | None
) -> None:
    db.add(
        QualityEvent(
            sensor_id=sensor_id or "unknown",
            timestamp=timestamp,
            received_at=now,
            outcome=outcome,
            reason=issue.reason,
            field=issue.field,
            detail=issue.detail,
            raw=_raw_text(raw),
        )
    )


def ingest_readings(db: Session, items: list[Any], now: datetime | None = None) -> IngestResult:
    """Validate, de-duplicate and store a batch of raw readings.

    Every item ends in exactly one of four outcomes: accepted, flagged, rejected or
    duplicate. All but "accepted" leave a row in the quality log.
    """
    now = now or utcnow()
    result = IngestResult(received=len(items))
    by_sensor: dict[str, list[tuple[int, CleanReading, Any]]] = defaultdict(list)

    for index, raw in enumerate(items):
        reading, issues = validate_item(raw)
        if reading is None:
            sensor_id = raw.get("sensor_id") if isinstance(raw, dict) and isinstance(raw.get("sensor_id"), str) else None
            _reject(db, result, now, index, raw, issues[0], sensor_id, None)
        else:
            by_sensor[reading.sensor_id].append((index, reading, raw))

    for sensor_id, entries in by_sensor.items():
        _ingest_sensor(db, result, now, sensor_id, entries)

    db.commit()
    result.problems.sort(key=lambda p: p.index)
    return result


def _reject(
    db: Session,
    result: IngestResult,
    now: datetime,
    index: int,
    raw: Any,
    issue: Issue,
    sensor_id: str | None,
    timestamp: datetime | None,
) -> None:
    result.rejected += 1
    result.problems.append(ItemProblem(index, "rejected", sensor_id, timestamp, [issue]))
    _log(db, now, "rejected", issue, raw, sensor_id, timestamp)


def _ingest_sensor(
    db: Session, result: IngestResult, now: datetime, sensor_id: str, entries: list[tuple[int, CleanReading, Any]]
) -> None:
    sensor = db.get(Sensor, sensor_id)
    if sensor is None:
        # A new station registers itself with its first reading, which must say where it is.
        km = next((r.station_km for _, r, _ in entries if r.station_km is not None), None)
        if km is None:
            issue = Issue("unknown_sensor", "station_km", "first reading from this sensor must include station_km")
            for index, reading, raw in entries:
                _reject(db, result, now, index, raw, issue, sensor_id, reading.timestamp)
            return
        sensor = Sensor(id=sensor_id, station_km=km, first_seen_at=now, last_seen_at=now)
        db.add(sensor)
        db.flush()  # the station row must exist before readings that refer to it

    sensor.last_seen_at = now  # any well-formed message proves the station is alive
    entries.sort(key=lambda e: e[1].timestamp)  # place out-of-order items correctly within the batch
    lo, hi = entries[0][1].timestamp, entries[-1][1].timestamp
    existing = set(
        db.scalars(
            select(Reading.timestamp).where(Reading.sensor_id == sensor_id, Reading.timestamp >= lo, Reading.timestamp <= hi)
        )
    )
    newest_before = sensor.last_reading_ts

    # The stuck check compares each new value with the readings just before it.
    run = settings.stuck_repeat_count
    recent: list[dict[str, float | None]] = []
    if run > 1:
        rows = db.execute(
            select(*(getattr(Reading, s) for s in SIGNAL_NAMES))
            .where(Reading.sensor_id == sensor_id)
            .order_by(Reading.timestamp.desc())
            .limit(run - 1)
        ).all()
        recent = [dict(zip(SIGNAL_NAMES, row)) for row in reversed(rows)]

    stored: list[datetime] = []
    for index, reading, raw in entries:
        ts = reading.timestamp
        if ts in existing:
            issue = Issue("duplicate", "", "a reading for this sensor and timestamp is already stored; the first one is kept")
            result.duplicates += 1
            result.problems.append(ItemProblem(index, "duplicate", sensor_id, ts, [issue]))
            _log(db, now, "duplicate", issue, raw, sensor_id, ts)
            continue
        existing.add(ts)

        issues = list(reading.issues)
        if reading.station_km is not None and abs(reading.station_km - sensor.station_km) > 1e-6:
            issues.append(
                Issue(
                    "station_km_mismatch",
                    "station_km",
                    f"reading says {reading.station_km:g} km, station is registered at {sensor.station_km:g} km; registered value kept",
                )
            )

        late = newest_before is not None and ts < newest_before
        if not late and run > 1:
            for name in SIGNAL_NAMES:
                value = reading.values[name]
                if value is not None and len(recent) >= run - 1 and all(r[name] == value for r in recent[-(run - 1) :]):
                    issues.append(Issue("stuck", name, f"same value {value:g} for {run} or more readings in a row"))
            recent.append(reading.values)

        db.add(
            Reading(
                sensor_id=sensor_id,
                timestamp=ts,
                flags=",".join(i.flag() for i in issues),
                arrived_late=late,
                received_at=now,
                **reading.values,
            )
        )
        stored.append(ts)
        if late:
            result.late += 1
            _log(
                db, now, "late", Issue("late_arrival", "", f"arrived after a reading taken at {newest_before.isoformat()}Z"),
                raw, sensor_id, ts,
            )
        if issues:
            result.flagged += 1
            result.problems.append(ItemProblem(index, "flagged", sensor_id, ts, issues))
            for issue in issues:
                _log(db, now, "flagged", issue, raw, sensor_id, ts)
        else:
            result.accepted += 1

    if stored:
        result.stored[sensor_id] = stored
        newest = max(stored)
        if sensor.last_reading_ts is None or newest > sensor.last_reading_ts:
            sensor.last_reading_ts = newest


# --------------------------------------------------------------------------------------
# Sensors and history
# --------------------------------------------------------------------------------------


def base_status(sensor: Sensor, now: datetime) -> str:
    """`offline` when nothing has arrived within the timeout, otherwise `normal`.

    This is judged on arrival time, not on reading timestamps: a silent station is one
    the backend has not heard from, whatever its last reading said.
    """
    silent_for = (now - sensor.last_seen_at).total_seconds()
    return STATUS_OFFLINE if silent_for > settings.sensor_offline_timeout_s else STATUS_NORMAL


def list_sensors(db: Session) -> list[Sensor]:
    return list(db.scalars(select(Sensor).order_by(Sensor.station_km, Sensor.id)))


def get_sensor(db: Session, sensor_id: str) -> Sensor | None:
    return db.get(Sensor, sensor_id)


def latest_reading(db: Session, sensor_id: str) -> Reading | None:
    return db.scalars(
        select(Reading).where(Reading.sensor_id == sensor_id).order_by(Reading.timestamp.desc()).limit(1)
    ).first()


def reading_count(db: Session, sensor_id: str | None = None) -> int:
    stmt = select(func.count(Reading.id))
    if sensor_id is not None:
        stmt = stmt.where(Reading.sensor_id == sensor_id)
    return db.scalar(stmt) or 0


def get_history(
    db: Session,
    sensor_id: str,
    start: datetime | None,
    end: datetime | None,
    limit: int,
    offset: int,
    newest_first: bool,
) -> tuple[int, list[Reading]]:
    """Readings of one sensor ordered by data time, whatever order they arrived in."""
    conditions = [Reading.sensor_id == sensor_id]
    if start is not None:
        conditions.append(Reading.timestamp >= start)
    if end is not None:
        conditions.append(Reading.timestamp <= end)
    total = db.scalar(select(func.count(Reading.id)).where(*conditions)) or 0
    order = Reading.timestamp.desc() if newest_first else Reading.timestamp.asc()
    rows = db.scalars(select(Reading).where(*conditions).order_by(order).limit(limit).offset(offset))
    return total, list(rows)


# --------------------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------------------


def quality_summary(db: Session, recent_limit: int = 50) -> dict[str, Any]:
    """Counts of rejected, flagged, duplicate and late readings, in total and per sensor."""
    per_sensor: dict[str, dict[str, Any]] = {}

    def entry(sensor_id: str) -> dict[str, Any]:
        return per_sensor.setdefault(
            sensor_id,
            {
                "sensor_id": sensor_id,
                "readings_stored": 0,
                "flagged_readings": 0,
                "rejected": 0,
                "duplicates": 0,
                "late": 0,
                "by_reason": {},
            },
        )

    for sensor_id, total, flagged in db.execute(
        select(Reading.sensor_id, func.count(Reading.id), func.sum(case((Reading.flags != "", 1), else_=0))).group_by(
            Reading.sensor_id
        )
    ):
        e = entry(sensor_id)
        e["readings_stored"] = total
        e["flagged_readings"] = int(flagged or 0)

    column = {"rejected": "rejected", "duplicate": "duplicates", "late": "late"}
    by_reason: dict[str, int] = defaultdict(int)
    for sensor_id, outcome, reason, count in db.execute(
        select(QualityEvent.sensor_id, QualityEvent.outcome, QualityEvent.reason, func.count(QualityEvent.id)).group_by(
            QualityEvent.sensor_id, QualityEvent.outcome, QualityEvent.reason
        )
    ):
        e = entry(sensor_id)
        e["by_reason"][reason] = e["by_reason"].get(reason, 0) + count
        by_reason[reason] += count
        if outcome in column:
            e[column[outcome]] += count

    sensors = sorted(per_sensor.values(), key=lambda e: e["sensor_id"])
    totals = {
        key: sum(e[key] for e in sensors) for key in ("readings_stored", "flagged_readings", "rejected", "duplicates", "late")
    }
    recent = list(db.scalars(select(QualityEvent).order_by(QualityEvent.id.desc()).limit(recent_limit)))
    return {"totals": totals, "by_reason": dict(sorted(by_reason.items())), "per_sensor": sensors, "recent_events": recent}


# --------------------------------------------------------------------------------------
# Analysis support: reading frames, scores, anomalies, predictions
# --------------------------------------------------------------------------------------

RISK_ORDER = ["Low", "Medium", "High", "Critical"]


def reading_rows(db: Session, sensor_ids: list[str], since: datetime) -> list[tuple]:
    """Readings of the given sensors from `since` on, ordered by sensor and data time.

    Returns plain tuples: (id, sensor_id, timestamp, temperature, vibration, speed, load,
    current, flags, is_anomaly, severity).
    """
    stmt = (
        select(
            Reading.id,
            Reading.sensor_id,
            Reading.timestamp,
            *(getattr(Reading, s) for s in SIGNAL_NAMES),
            Reading.flags,
            Reading.is_anomaly,
            Reading.severity,
        )
        .where(Reading.sensor_id.in_(sensor_ids), Reading.timestamp >= since)
        .order_by(Reading.sensor_id, Reading.timestamp)
    )
    # Run on the connection: this is read thousands of rows at a time and needs no ORM objects.
    return db.connection().execute(stmt).all()


def save_scores(db: Session, rows: list[dict[str, Any]]) -> None:
    """rows: [{"id":, "anomaly_score":, "is_anomaly":, "severity":}]"""
    if rows:
        db.execute(update(Reading), rows)


def active_anomaly(db: Session, sensor_id: str) -> Anomaly | None:
    return db.scalars(
        select(Anomaly).where(Anomaly.sensor_id == sensor_id, Anomaly.status == "active").order_by(Anomaly.id.desc()).limit(1)
    ).first()


def open_anomaly(db: Session, sensor: Sensor, ts: datetime, anomaly_type: str, severity: str, score: float) -> Anomaly:
    anomaly = Anomaly(
        sensor_id=sensor.id,
        station_km=sensor.station_km,
        started_at=ts,
        last_anomalous_at=ts,
        status="active",
        anomaly_type=anomaly_type,
        severity=severity,
        worst_score=score,
        reading_count=1,
    )
    db.add(anomaly)
    return anomaly


def extend_anomaly(anomaly: Anomaly, ts: datetime, anomaly_type: str, severity: str, score: float) -> None:
    anomaly.last_anomalous_at = max(anomaly.last_anomalous_at, ts)
    anomaly.reading_count += 1
    if RISK_ORDER.index(severity) > RISK_ORDER.index(anomaly.severity):
        anomaly.severity, anomaly.anomaly_type = severity, anomaly_type
    anomaly.worst_score = min(anomaly.worst_score, score)


def clear_anomaly(anomaly: Anomaly) -> None:
    anomaly.status = "cleared"
    anomaly.ended_at = anomaly.last_anomalous_at


def list_anomalies(
    db: Session,
    sensor_id: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[Anomaly]]:
    """Anomalies, newest first. `start` / `end` select anomalies that overlap that period."""
    conditions = []
    if sensor_id:
        conditions.append(Anomaly.sensor_id == sensor_id)
    if severity:
        conditions.append(Anomaly.severity == severity)
    if status:
        conditions.append(Anomaly.status == status)
    if start is not None:
        conditions.append(Anomaly.last_anomalous_at >= start)
    if end is not None:
        conditions.append(Anomaly.started_at <= end)
    total = db.scalar(select(func.count(Anomaly.id)).where(*conditions)) or 0
    rows = db.scalars(
        select(Anomaly).where(*conditions).order_by(Anomaly.started_at.desc(), Anomaly.id.desc()).limit(limit).offset(offset)
    )
    return total, list(rows)


def add_prediction(db: Session, sensor: Sensor, created_at: datetime, assessment: Any) -> Prediction:
    prediction = Prediction(
        sensor_id=sensor.id,
        station_km=sensor.station_km,
        created_at=created_at,
        failure_mode=assessment.failure_mode,
        failure_mode_label=assessment.failure_mode_label,
        risk_level=assessment.risk_level,
        eta_low_h=assessment.eta_low_h,
        eta_high_h=assessment.eta_high_h,
        confidence=assessment.confidence,
        reason=assessment.reason,
        details=json.dumps(assessment.details, default=str),
    )
    db.add(prediction)
    return prediction


def latest_predictions(db: Session) -> dict[str, Prediction]:
    """The newest prediction of every sensor, keyed by sensor id."""
    newest = select(func.max(Prediction.id)).group_by(Prediction.sensor_id)
    return {p.sensor_id: p for p in db.scalars(select(Prediction).where(Prediction.id.in_(newest)))}


def prediction_history(db: Session, sensor_id: str, limit: int, offset: int) -> tuple[int, list[Prediction]]:
    total = db.scalar(select(func.count(Prediction.id)).where(Prediction.sensor_id == sensor_id)) or 0
    rows = db.scalars(
        select(Prediction)
        .where(Prediction.sensor_id == sensor_id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return total, list(rows)


def station_statuses(db: Session, sensors: list[Sensor], now: datetime) -> dict[str, str]:
    """Status of each station, most urgent first:

    offline            nothing received within the timeout (nothing else can be judged)
    predicted failure  the latest prediction has risk High or Critical
    abnormal           an anomaly is active now
    normal             none of the above
    """
    active = set(db.scalars(select(Anomaly.sensor_id).where(Anomaly.status == "active")))
    predictions = latest_predictions(db)
    statuses = {}
    for sensor in sensors:
        if base_status(sensor, now) == STATUS_OFFLINE:
            statuses[sensor.id] = STATUS_OFFLINE
        elif sensor.id in predictions and predictions[sensor.id].risk_level in ("High", "Critical"):
            statuses[sensor.id] = STATUS_PREDICTED_FAILURE
        elif sensor.id in active:
            statuses[sensor.id] = STATUS_ABNORMAL
        else:
            statuses[sensor.id] = STATUS_NORMAL
    return statuses
