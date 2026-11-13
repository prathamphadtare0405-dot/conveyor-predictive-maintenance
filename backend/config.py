"""Central configuration.

Every setting comes from an environment variable (optionally loaded from a `.env`
file in the project root) and has a default, so the project runs with no setup.
The simulator, ML code, backend and dashboard all import `settings` from here so
that there is one definition of the conveyor and its operating limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} is not a number") from exc


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} is not an integer") from exc


def _path(name: str, default: str) -> Path:
    """Relative paths are resolved against the project root, not the current directory."""
    p = Path(_str(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class SignalSpec:
    """Operating envelope of one measured signal.

    `hard_min` / `hard_max` bound what the instrument can physically report; a value
    outside them is a sensor or transmission fault, not a process condition.
    `warn` and `fail` are process limits: beyond `warn` the reading is outside normal
    operation, and `fail` is the level at which the simulator declares the component
    failed and to which the predictor extrapolates trends. `direction` is +1 when the
    signal fails by rising (temperature, vibration, load, current) and -1 when it
    fails by falling (belt speed dropping below drive speed means slip).
    """

    name: str
    unit: str
    hard_min: float
    hard_max: float
    warn: float
    fail: float
    direction: int = 1

    def beyond_warn(self, value: float) -> bool:
        return (value - self.warn) * self.direction > 0

    def beyond_fail(self, value: float) -> bool:
        return (value - self.fail) * self.direction >= 0


# One table of limits shared by the simulator (to generate), the backend (to validate
# and grade severity) and the predictor (to extrapolate to). The reasoning behind the
# numbers is in docs/DESIGN.md.
SIGNALS: dict[str, SignalSpec] = {
    "temperature": SignalSpec("temperature", "degC", -40.0, 200.0, 75.0, 95.0),
    "vibration": SignalSpec("vibration", "mm/s", 0.0, 100.0, 7.1, 11.0),
    "speed": SignalSpec("speed", "m/s", 0.0, 10.0, 3.6, 3.2, direction=-1),
    "load": SignalSpec("load", "%", 0.0, 200.0, 100.0, 120.0),
    "current": SignalSpec("current", "A", 0.0, 1000.0, 340.0, 400.0),
}
SIGNAL_NAMES: tuple[str, ...] = tuple(SIGNALS)


@dataclass(frozen=True)
class Settings:
    # Conveyor geometry
    conveyor_length_km: float = field(default_factory=lambda: _float("CONVEYOR_LENGTH_KM", 5.0))
    sensor_count: int = field(default_factory=lambda: _int("SENSOR_COUNT", 10))
    nominal_belt_speed_mps: float = field(default_factory=lambda: _float("NOMINAL_BELT_SPEED_MPS", 4.0))

    # Storage
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", "data"))
    database_url: str = field(default_factory=lambda: _str("DATABASE_URL", ""))

    # Backend
    backend_host: str = field(default_factory=lambda: _str("BACKEND_HOST", "127.0.0.1"))
    backend_port: int = field(default_factory=lambda: _int("BACKEND_PORT", 8000))
    backend_url: str = field(default_factory=lambda: _str("BACKEND_URL", "http://127.0.0.1:8000"))
    sensor_offline_timeout_s: float = field(default_factory=lambda: _float("SENSOR_OFFLINE_TIMEOUT_S", 60.0))
    anomaly_clear_after_s: float = field(default_factory=lambda: _float("ANOMALY_CLEAR_AFTER_S", 120.0))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO"))

    # Simulator
    sim_interval_s: float = field(default_factory=lambda: _float("SIM_INTERVAL_S", 5.0))
    sim_speedup: float = field(default_factory=lambda: _float("SIM_SPEEDUP", 1.0))
    sim_fault_rate: float = field(default_factory=lambda: _float("SIM_FAULT_RATE", 0.5))
    sim_messy_rate: float = field(default_factory=lambda: _float("SIM_MESSY_RATE", 0.03))
    sim_seed: int = field(default_factory=lambda: _int("SIM_SEED", 42))

    # Anomaly detection and prediction
    ml_model_dir: Path = field(default_factory=lambda: _path("ML_MODEL_DIR", "ml/models"))
    rolling_window: int = field(default_factory=lambda: _int("ROLLING_WINDOW", 24))
    iforest_contamination: float = field(default_factory=lambda: _float("IFOREST_CONTAMINATION", 0.02))
    trend_lookback_min: float = field(default_factory=lambda: _float("TREND_LOOKBACK_MIN", 180.0))

    # Vision
    vision_weights: Path = field(default_factory=lambda: _path("VISION_WEIGHTS", "vision/models/belt_damage.pt"))
    vision_conf_threshold: float = field(default_factory=lambda: _float("VISION_CONF_THRESHOLD", 0.25))
    vision_output_dir: Path = field(default_factory=lambda: _path("VISION_OUTPUT_DIR", "data/vision"))
    registry_position_tolerance_m: float = field(default_factory=lambda: _float("REGISTRY_POSITION_TOLERANCE_M", 5.0))
    registry_missed_laps: int = field(default_factory=lambda: _int("REGISTRY_MISSED_LAPS", 3))

    # Dashboard
    dashboard_refresh_s: int = field(default_factory=lambda: _int("DASHBOARD_REFRESH_S", 5))

    def __post_init__(self) -> None:
        if self.conveyor_length_km <= 0:
            raise ValueError("CONVEYOR_LENGTH_KM must be positive")
        if self.sensor_count < 1:
            raise ValueError("SENSOR_COUNT must be at least 1")
        if not self.database_url:
            # Absolute path so the same file is used whatever the working directory is.
            db_file = (self.data_dir / "conveyor.db").as_posix()
            object.__setattr__(self, "database_url", f"sqlite:///{db_file}")

    @property
    def belt_loop_length_m(self) -> float:
        """Length of the closed belt: carry side + return side + wrap around the pulleys."""
        override = os.getenv("BELT_LOOP_LENGTH_M")
        if override not in (None, ""):
            return float(override)
        return 2.0 * self.conveyor_length_km * 1000.0 + 30.0

    def station_positions_km(self, count: int | None = None) -> list[float]:
        """Fixed kilometre position of each sensor station.

        Stations sit at the centre of equal segments, so with the defaults (5 km, 10
        stations) they are at 0.25, 0.75, ... 4.75 km.
        """
        n = count or self.sensor_count
        segment = self.conveyor_length_km / n
        return [round(segment * (i + 0.5), 3) for i in range(n)]

    @staticmethod
    def sensor_id(index: int) -> str:
        """Station ids are S01, S02, ... counted from the tail (loading) end."""
        return f"S{index + 1:02d}"


settings = Settings()
