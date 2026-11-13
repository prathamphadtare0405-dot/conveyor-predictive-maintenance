"""FastAPI application.

Run from the project root:

    uvicorn backend.main:app --reload

Interactive API docs are at http://127.0.0.1:8000/docs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.config import settings
from backend.database import init_db
from backend.routes import sensors, system

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Conveyor predictive maintenance",
    description="Receives sensor readings and belt images, stores them, and reports anomalies, failure predictions and belt damage.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(sensors.router)
app.include_router(system.router)
