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
