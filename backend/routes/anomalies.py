"""Anomalies raised by the detector."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.database import get_db
from backend.validation import to_naive_utc

router = APIRouter(prefix="/api/anomalies", tags=["anomalies"])

SEVERITY_PATTERN = "^(Low|Medium|High|Critical)$"


@router.get("", response_model=schemas.Page[schemas.AnomalyOut], summary="Anomalies, newest first")
def list_anomalies(
    sensor_id: str | None = Query(None),
    severity: str | None = Query(None, pattern=SEVERITY_PATTERN),
    status: str | None = Query(None, pattern="^(active|cleared)$"),
    start: datetime | None = Query(None, description="Only anomalies still going on at or after this time"),
    end: datetime | None = Query(None, description="Only anomalies that started at or before this time"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    start = to_naive_utc(start) if start else None
    end = to_naive_utc(end) if end else None
    if start and end and start > end:
        raise HTTPException(422, "start is after end.")
    total, rows = crud.list_anomalies(db, sensor_id, severity, status, start, end, limit, offset)
    return {"total": total, "limit": limit, "offset": offset, "items": rows}


@router.get("/active", response_model=list[schemas.AnomalyOut], summary="Anomalies that are active now")
def active_anomalies(db: Session = Depends(get_db)):
    """An anomaly stays active until the station has been back to normal for the clear period."""
    _, rows = crud.list_anomalies(db, status="active", limit=1000)
    return rows
