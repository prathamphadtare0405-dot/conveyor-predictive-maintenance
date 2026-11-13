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
from ml.train import model_path

router = APIRouter(tags=["system"])


def model_state() -> str:
    return "loaded" if model_path().exists() else "not trained (run: python -m ml.train)"


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
        "anomaly_model": model_state(),
    }


@router.get("/api/data-quality", response_model=schemas.DataQualityOut, summary="Rejected, flagged, duplicate and late readings")
def data_quality(recent: int = Query(50, ge=0, le=500, description="How many recent log entries to include"), db: Session = Depends(get_db)):
    return crud.quality_summary(db, recent_limit=recent)


@router.get("/api/dashboard/summary", summary="Everything the dashboard header needs in one call")
def dashboard_summary(db: Session = Depends(get_db)) -> dict[str, Any]:
    now = crud.utcnow()
    sensors = crud.list_sensors(db)
    statuses = crud.station_statuses(db, sensors, now)
    status_counts: dict[str, int] = {}
    for status in statuses.values():
        status_counts[status] = status_counts.get(status, 0) + 1

    _, active = crud.list_anomalies(db, status="active", limit=1000)
    severity_counts: dict[str, int] = {}
    for anomaly in active:
        severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1
    risk_counts: dict[str, int] = {}
    for prediction in crud.latest_predictions(db).values():
        risk_counts[prediction.risk_level] = risk_counts.get(prediction.risk_level, 0) + 1

    newest = max((s.last_reading_ts for s in sensors if s.last_reading_ts), default=None)
    quality = crud.quality_summary(db, recent_limit=0)
    return {
        "server_time": now.isoformat(timespec="milliseconds") + "Z",
        "conveyor_length_km": settings.conveyor_length_km,
        "anomaly_model": model_state(),
        "sensors_total": len(sensors),
        "status_counts": status_counts,
        "active_anomalies": len(active),
        "active_anomalies_by_severity": severity_counts,
        "predictions_by_risk": risk_counts,
        "latest_reading_time": newest.isoformat(timespec="milliseconds") + "Z" if newest else None,
        "readings_stored": quality["totals"]["readings_stored"],
        "data_quality": quality["totals"],
        "vision_model": "available" if settings.vision_weights.exists() else "not trained (run: python -m vision.train)",
        "vision": crud.vision_summary(db),
    }
