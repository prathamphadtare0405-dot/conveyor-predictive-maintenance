"""Health, data quality and the dashboard summary."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.config import settings
from backend.database import get_db

router = APIRouter(tags=["system"])


@router.get("/health", response_model=schemas.HealthOut)
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("select 1"))
        database, sensors, readings = "ok", len(crud.list_sensors(db)), crud.reading_count(db)
    except SQLAlchemyError as exc:
        database, sensors, readings = f"error: {type(exc).__name__}", 0, 0
    return {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "server_time": crud.utcnow(),
        "sensors": sensors,
        "readings_stored": readings,
    }


@router.get("/api/data-quality", response_model=schemas.DataQualityOut, summary="Rejected, flagged, duplicate and late readings")
def data_quality(recent: int = Query(50, ge=0, le=500, description="How many recent log entries to include"), db: Session = Depends(get_db)):
    return crud.quality_summary(db, recent_limit=recent)


@router.get("/api/dashboard/summary", summary="Everything the dashboard header needs in one call")
def dashboard_summary(db: Session = Depends(get_db)) -> dict[str, Any]:
    now = crud.utcnow()
    sensors = crud.list_sensors(db)
    status_counts: dict[str, int] = {}
    for sensor in sensors:
        status = crud.base_status(sensor, now)
        status_counts[status] = status_counts.get(status, 0) + 1
    newest = max((s.last_reading_ts for s in sensors if s.last_reading_ts), default=None)
    quality = crud.quality_summary(db, recent_limit=0)
    return {
        "server_time": now.isoformat(timespec="milliseconds") + "Z",
        "conveyor_length_km": settings.conveyor_length_km,
        "sensors_total": len(sensors),
        "status_counts": status_counts,
        "latest_reading_time": newest.isoformat(timespec="milliseconds") + "Z" if newest else None,
        "readings_stored": quality["totals"]["readings_stored"],
        "data_quality": quality["totals"],
    }
