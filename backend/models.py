"""Database tables.

All datetimes are stored as naive UTC. Two clocks are kept apart on purpose:

* `timestamp` columns are *data time*: when the sensor says the reading was taken.
* `received_at` / `last_seen_at` are *arrival time*: the backend's own clock.

Data time orders the history and drives analysis. Arrival time decides whether a
station is offline. They differ whenever readings arrive late or the simulator runs
with time acceleration.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class Sensor(Base):
    """A sensor station at a fixed kilometre position on the conveyor structure."""

    __tablename__ = "sensors"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    station_km: Mapped[float] = mapped_column(Float)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime)  # arrival time of the latest message
    last_reading_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # newest data time stored


class Reading(Base):
    """One stored reading. A signal is NULL when it was missing or unusable."""

    __tablename__ = "readings"
    __table_args__ = (
        UniqueConstraint("sensor_id", "timestamp", name="uq_reading_sensor_time"),
        Index("ix_reading_sensor_time", "sensor_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(ForeignKey("sensors.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    vibration: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    load: Mapped[float | None] = mapped_column(Float, nullable=True)
    current: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Comma-separated "reason:field" entries, e.g. "nan:current,stuck:vibration". Empty when clean.
    flags: Mapped[str] = mapped_column(String(255), default="")
    arrived_late: Mapped[bool] = mapped_column(Boolean, default=False)
    received_at: Mapped[datetime] = mapped_column(DateTime)
    # Filled by the analysis step. NULL means the reading was not scored (no model, or it arrived too late).
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_anomaly: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    severity: Mapped[str] = mapped_column(String(10), default="")


class QualityEvent(Base):
    """Log of everything that was wrong with incoming data. Nothing is dropped silently.

    outcome:
      rejected   the item could not be stored at all
      flagged    the item was stored, with one signal blanked or marked (one row per problem)
      duplicate  a reading for this sensor and timestamp already existed; not stored again
      late       the item was stored but arrived after newer readings of the same sensor
    """

    __tablename__ = "quality_events"
    __table_args__ = (Index("ix_quality_sensor_outcome", "sensor_id", "outcome"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(String(32), default="unknown")
    timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime)
    outcome: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(String(48))
    field: Mapped[str] = mapped_column(String(32), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    raw: Mapped[str] = mapped_column(Text, default="")  # the offending payload, truncated


class Anomaly(Base):
    """A period during which a station's readings were anomalous.

    It is `active` from the first confirmed anomalous reading until the station has been
    back to normal for the configured clear period; then it is `cleared` and `ended_at`
    is set to the last anomalous reading. Severity and type are those of the worst reading.
    """

    __tablename__ = "anomalies"
    __table_args__ = (Index("ix_anomaly_sensor_status", "sensor_id", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(ForeignKey("sensors.id"))
    station_km: Mapped[float] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime)  # data time
    last_anomalous_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(10), default="active")  # active | cleared
    anomaly_type: Mapped[str] = mapped_column(String(160))
    severity: Mapped[str] = mapped_column(String(10))
    worst_score: Mapped[float] = mapped_column(Float)
    reading_count: Mapped[int] = mapped_column(Integer, default=1)


class Prediction(Base):
    """One run of the failure predictor for one station: what, where, when, and why."""

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_prediction_sensor_time", "sensor_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(ForeignKey("sensors.id"))
    station_km: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime)  # data time of the newest reading used
    failure_mode: Mapped[str | None] = mapped_column(String(24), nullable=True)
    failure_mode_label: Mapped[str] = mapped_column(String(48))
    risk_level: Mapped[str] = mapped_column(String(10))
    eta_low_h: Mapped[float | None] = mapped_column(Float, nullable=True)
    eta_high_h: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(10), nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    details: Mapped[str] = mapped_column(Text, default="{}")  # JSON: levels, trend fits, risk parts
