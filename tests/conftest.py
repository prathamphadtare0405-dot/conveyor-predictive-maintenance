"""Shared test setup.

The environment is pointed at a temporary directory *before* any project module is
imported, so tests never touch the real database or data folder.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="conveyor-tests-"))
os.environ["DATA_DIR"] = str(_TMP)
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["ML_MODEL_DIR"] = str(_TMP / "ml_models")
os.environ["VISION_OUTPUT_DIR"] = str(_TMP / "vision")
os.environ["VISION_WEIGHTS"] = str(_TMP / "missing_weights.pt")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture()
def db():
    """An empty database for each test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db):
    return TestClient(app)


def reading(sensor_id: str = "S01", ts: str = "2026-11-13T08:00:00Z", **overrides) -> dict:
    """A valid reading; pass keyword arguments to change or break individual fields."""
    item = {
        "sensor_id": sensor_id,
        "station_km": 0.25,
        "timestamp": ts,
        "temperature": 52.0,
        "vibration": 2.5,
        "speed": 4.0,
        "load": 70.0,
        "current": 260.0,
    }
    item.update(overrides)
    return item
