"""Belt images: detection, raw detections, and the registry of unique damages."""

from __future__ import annotations

import logging
from datetime import datetime

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.config import settings
from backend.database import get_db
from backend.validation import to_naive_utc
from vision import appearance, belt_registry
from vision.detection import VisionModelMissing, annotate, crop, get_detector
from vision.preprocessing import InvalidImage, decode_image, difference_hash, hash_to_hex

log = logging.getLogger("backend.vision")

router = APIRouter(prefix="/api/vision", tags=["vision"])

IMAGE_URL_PREFIX = "/vision-images"


def image_url(path: str) -> str | None:
    return f"{IMAGE_URL_PREFIX}/{path}" if path else None


def dead_reckon(db: Session, captured_at: datetime) -> tuple[float, int]:
    """Estimate belt position from time alone: nominal speed since the first frame.

    This is the fallback when the client sends no position. It assumes constant speed and
    no marker correction, so it drifts; the response says `position_source: dead_reckoning`.
    """
    first = crud.first_frame(db)
    if first is None:
        return 0.0, 0
    elapsed = (captured_at - first.captured_at).total_seconds()
    if elapsed < 0:
        raise HTTPException(422, "captured_at is earlier than the first frame; send belt_position_m and lap with the image.")
    travelled = first.belt_position_m + elapsed * settings.nominal_belt_speed_mps
    loop = settings.belt_loop_length_m
    return travelled % loop, first.lap + int(travelled // loop)


@router.post(
    "/detect",
    response_model=schemas.DetectResponse,
    summary="Analyse one belt image",
    responses={422: {"description": "Not a usable image, or inconsistent position fields."}, 503: {"description": "No trained vision model."}},
)
def detect(
    file: UploadFile = File(..., description="The image (JPEG or PNG)"),
    belt_position_m: float | None = Form(None, description="Position of this frame along the belt loop, in metres from the reference marker"),
    lap: int | None = Form(None, description="Lap number of the belt at this frame"),
    captured_at: datetime | None = Form(None, description="When the frame was taken (ISO 8601); defaults to now"),
    db: Session = Depends(get_db),
):
    """Runs the detector, then puts each detection through the damage registry.

    Send `belt_position_m` and `lap` from the camera's position tracker. Without them the
    backend estimates the position from the capture time (dead reckoning).
    """
    if file.content_type and not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):
        raise HTTPException(422, f"Expected an image, got content type {file.content_type!r}.")
    try:
        image = decode_image(file.file.read())
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc

    if (belt_position_m is None) != (lap is None):
        raise HTTPException(422, "Send belt_position_m and lap together, or neither.")
    loop = settings.belt_loop_length_m
    if belt_position_m is not None and not 0.0 <= belt_position_m < loop:
        raise HTTPException(422, f"belt_position_m must be between 0 and the loop length ({loop:g} m).")
    if lap is not None and lap < 0:
        raise HTTPException(422, "lap cannot be negative.")
    when = to_naive_utc(captured_at) if captured_at else crud.utcnow()

    try:
        detections, analysed = get_detector().detect(image)
    except VisionModelMissing as exc:
        raise HTTPException(503, str(exc)) from exc

    if belt_position_m is None:
        position, lap_number, source = *dead_reckon(db, when), "dead_reckoning"
    else:
        position, lap_number, source = belt_position_m, lap, "client"

    hashes = [hash_to_hex(difference_hash(crop(analysed, d))) for d in detections]
    looks = [appearance.extract(analysed, d.x1, d.y1, d.x2, d.y2) for d in detections]
    frame, rows = crud.register_frame(db, when, position, lap_number, source, detections, hashes, looks)

    if rows:
        # Only frames that show damage are kept as images.
        relative = f"frames/frame_{frame.id:06d}.jpg"
        target = settings.vision_output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(target), annotate(analysed, detections, [f"#{damage.id}" for _, damage in rows])):
            frame.image_path = relative
            for _, damage in rows:
                damage.image_path = relative
        else:
            log.error("Could not write the annotated image to %s", target)
    db.commit()

    return {
        "frame_id": frame.id,
        "captured_at": frame.captured_at,
        "belt_position_m": round(position, 2),
        "lap": lap_number,
        "position_source": source,
        "image_width": analysed.shape[1],
        "image_height": analysed.shape[0],
        "detections": [
            {
                "damage_class": row.damage_class,
                "confidence": round(row.confidence, 4),
                "box": {"x1": round(row.x1, 1), "y1": round(row.y1, 1), "x2": round(row.x2, 1), "y2": round(row.y2, 1)},
                "area_fraction": round(row.area_fraction, 5),
                "damage_id": damage.id,
                "is_new_damage": row.is_new_damage,
            }
            for row, damage in rows
        ],
        "annotated_image_url": image_url(frame.image_path),
    }


@router.post("/reset", summary="Empty the damage registry (demo helper)")
def reset(db: Session = Depends(get_db)) -> dict[str, int]:
    """Deletes all frames, detections, damages and saved images. Sensor data is not touched.

    This exists so that `scripts/seed_demo.py` can be run repeatedly. A real deployment
    would not expose it without access control.
    """
    counts = crud.reset_vision(db)
    frames_dir = settings.vision_output_dir / "frames"
    if frames_dir.is_dir():
        for path in frames_dir.glob("frame_*.jpg"):
            path.unlink(missing_ok=True)
    return counts


@router.get("/detections", response_model=schemas.Page[schemas.DetectionOut], summary="Raw detections, newest first")
def list_detections(
    damage_id: int | None = Query(None),
    lap: int | None = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Every box the detector produced. The same damage appears here once per sighting; use /damages for unique damages."""
    total, rows = crud.list_detections(db, damage_id, lap, limit, offset)
    return {"total": total, "limit": limit, "offset": offset, "items": rows}


@router.get("/damages", response_model=list[schemas.DamageOut], summary="Unique damages on the belt, by position along the loop")
def list_damages(
    status: str | None = Query(None, pattern="^(active|not re-observed)$"),
    confirmed_only: bool = Query(False, description="Leave out damages seen on one lap only"),
    db: Session = Depends(get_db),
):
    """A damage is `confirmed` once it has been seen on two different laps. A single
    sighting may be dirt, a shadow or a detector error, and is listed as unconfirmed."""
    history = crud.damage_size_history(db)
    out = []
    for damage in crud.list_damages(db, status):
        if confirmed_only and not belt_registry.is_confirmed(damage):
            continue
        points = history.get(damage.id, [])
        growth = belt_registry.growth_trend([(p["lap"], p["area_fraction"]) for p in points])
        out.append(
            {
                "id": damage.id,
                "damage_class": damage.damage_class,
                "status": damage.status,
                "confirmed": belt_registry.is_confirmed(damage),
                "belt_position_m": round(damage.position_m, 1),
                "lateral": round(damage.lateral, 3),
                "first_seen_at": damage.first_seen_at,
                "last_seen_at": damage.last_seen_at,
                "first_lap": damage.first_lap,
                "last_lap": damage.last_lap,
                "laps_seen": damage.laps_seen,
                "sightings": damage.sightings,
                "best_confidence": round(damage.best_confidence, 3),
                "last_confidence": round(damage.last_confidence, 3),
                "growth_trend": growth["trend"],
                "growth_pct_per_lap": growth["growth_pct_per_lap"],
                "size_history": points,
                "annotated_image_url": image_url(damage.image_path),
            }
        )
    return out
