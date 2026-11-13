"""Sensor readings: ingest, station list, history."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.config import settings
from backend.database import get_db
from backend.models import Sensor
from backend.validation import to_naive_utc
from ml.live import analyser

log = logging.getLogger("backend.sensors")

router = APIRouter(prefix="/api/sensors", tags=["sensors"])

MAX_PROBLEMS_RETURNED = 100


def sensor_out(db: Session, sensor: Sensor, now: datetime, status: str, prediction=None) -> dict[str, Any]:
    latest = crud.latest_reading(db, sensor.id)
    return {
        "sensor_id": sensor.id,
        "station_km": sensor.station_km,
        "status": status,
        "active_anomaly": crud.active_anomaly(db, sensor.id),
        "prediction": prediction,
        "last_seen_at": sensor.last_seen_at,
        "first_seen_at": sensor.first_seen_at,
        "seconds_since_last_message": round((now - sensor.last_seen_at).total_seconds(), 1),
        "latest_reading": latest,
    }


@router.post(
    "/readings",
    response_model=schemas.IngestResponse,
    responses={422: {"description": "Nothing in the request could be stored; the body says why, item by item."}},
    summary="Send one reading (JSON object) or a batch (JSON array)",
)
def post_readings(payload: Any = Body(...), db: Session = Depends(get_db)):
    """Each reading ends as accepted, flagged, rejected or duplicate, and the response
    says which. The request as a whole fails with 422 only when no item was usable."""
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise HTTPException(422, "Body must be a reading object or an array of reading objects.")
    if not items:
        raise HTTPException(422, "Empty array: no readings to store.")
    if len(items) > settings.max_batch_size:
        raise HTTPException(422, f"Batch of {len(items)} readings exceeds the limit of {settings.max_batch_size}.")

    result = crud.ingest_readings(db, items)
    # The readings are committed at this point. If analysis fails, the data is still safe
    # and the failure is logged; ingest never fails because of the model.
    try:
        analyser.run(db, result.stored)
    except Exception:
        db.rollback()
        log.exception("Analysis failed for a batch of %d readings; the readings themselves are stored.", result.received)
    body = schemas.IngestResponse(
        received=result.received,
        accepted=result.accepted,
        flagged=result.flagged,
        rejected=result.rejected,
        duplicates=result.duplicates,
        late=result.late,
        problems=[
            schemas.ItemProblemOut(
                index=p.index,
                outcome=p.outcome,
                sensor_id=p.sensor_id,
                timestamp=p.timestamp,
                issues=[schemas.IssueOut(reason=i.reason, field=i.field, detail=i.detail) for i in p.issues],
            )
            for p in result.problems[:MAX_PROBLEMS_RETURNED]
        ],
        problems_truncated=len(result.problems) > MAX_PROBLEMS_RETURNED,
    )
    if result.nothing_usable:
        return JSONResponse(status_code=422, content=jsonable_encoder(body))
    return body


@router.get("", response_model=list[schemas.SensorOut], summary="All sensor stations with status and latest reading")
def list_sensors(db: Session = Depends(get_db)):
    now = crud.utcnow()
    sensors = crud.list_sensors(db)
    statuses = crud.station_statuses(db, sensors, now)
    predictions = crud.latest_predictions(db)
    return [sensor_out(db, s, now, statuses[s.id], predictions.get(s.id)) for s in sensors]


@router.get("/{sensor_id}", response_model=schemas.SensorDetail)
def get_sensor(sensor_id: str, db: Session = Depends(get_db)):
    sensor = crud.get_sensor(db, sensor_id)
    if sensor is None:
        raise HTTPException(404, f"Sensor {sensor_id!r} is not known.")
    now = crud.utcnow()
    status = crud.station_statuses(db, [sensor], now)[sensor.id]
    out = sensor_out(db, sensor, now, status, crud.latest_predictions(db).get(sensor.id))
    out["readings_stored"] = crud.reading_count(db, sensor_id)
    return out


@router.get("/{sensor_id}/history", response_model=schemas.Page[schemas.ReadingOut])
def get_history(
    sensor_id: str,
    start: datetime | None = Query(None, description="Only readings at or after this time (ISO 8601)"),
    end: datetime | None = Query(None, description="Only readings at or before this time (ISO 8601)"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    order: str = Query("desc", pattern="^(asc|desc)$", description="desc = newest first"),
    db: Session = Depends(get_db),
):
    if crud.get_sensor(db, sensor_id) is None:
        raise HTTPException(404, f"Sensor {sensor_id!r} is not known.")
    start = to_naive_utc(start) if start else None
    end = to_naive_utc(end) if end else None
    if start and end and start > end:
        raise HTTPException(422, "start is after end.")
    total, rows = crud.get_history(db, sensor_id, start, end, limit, offset, newest_first=order == "desc")
    return {"total": total, "limit": limit, "offset": offset, "items": rows}
