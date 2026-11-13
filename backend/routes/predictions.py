"""Failure predictions: what might fail, where, and when."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.database import get_db

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("", response_model=list[schemas.PredictionOut], summary="Latest prediction of every station, highest risk first")
def latest_predictions(
    min_risk: str = Query("Low", pattern="^(Low|Medium|High|Critical)$", description="Leave out stations below this risk level"),
    db: Session = Depends(get_db),
):
    floor = crud.RISK_ORDER.index(min_risk)
    rows = [p for p in crud.latest_predictions(db).values() if crud.RISK_ORDER.index(p.risk_level) >= floor]
    rows.sort(key=lambda p: (-crud.RISK_ORDER.index(p.risk_level), p.eta_low_h if p.eta_low_h is not None else 1e9, p.station_km))
    return rows


@router.get("/{sensor_id}", response_model=schemas.SensorPredictions, summary="Latest prediction and history for one station")
def sensor_predictions(
    sensor_id: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    if crud.get_sensor(db, sensor_id) is None:
        raise HTTPException(404, f"Sensor {sensor_id!r} is not known.")
    total, rows = crud.prediction_history(db, sensor_id, limit, offset)
    latest = crud.latest_predictions(db).get(sensor_id)
    return {"sensor_id": sensor_id, "latest": latest, "history": {"total": total, "limit": limit, "offset": offset, "items": rows}}
