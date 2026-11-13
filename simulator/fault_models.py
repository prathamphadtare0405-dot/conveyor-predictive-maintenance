"""Failure modes used by the sensor simulator.

Two kinds of fault are modelled:

* **Gradual faults** develop over hours and end in a simulated failure. Each one has a
  known failure time, so every reading produced while it is active can be labelled with
  the true time remaining. This is what makes the predictor testable.
* **Sudden events** (blockage, spike) appear without warning and are over in seconds or
  minutes. They exist so the system also sees faults that cannot be predicted.

A fault never produces readings itself. It adjusts the healthy values that the station
computed, in two steps: first the load (because current and temperature are derived from
load), then the remaining signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from backend.config import SIGNALS


class FaultMode(str, Enum):
    NONE = "none"
    BEARING_WEAR = "bearing_wear"
    BELT_SLIP = "belt_slip"
    OVERLOAD = "overload"
    BLOCKAGE = "blockage"
    SPIKE = "spike"


GRADUAL_MODES: tuple[FaultMode, ...] = (FaultMode.BEARING_WEAR, FaultMode.BELT_SLIP, FaultMode.OVERLOAD)
SUDDEN_MODES: tuple[FaultMode, ...] = (FaultMode.BLOCKAGE, FaultMode.SPIKE)

STAGE_HEALTHY = "healthy"
STAGE_EARLY = "early"
STAGE_DEVELOPING = "developing"
STAGE_ADVANCED = "advanced"
STAGE_FAILED = "failed"
STAGE_SUDDEN = "sudden"

# Load level an overloaded section settles at, in % of rated load.
OVERLOAD_LEVEL = 115.0


def stage_for_progress(progress: float) -> str:
    """Name the degradation stage from the fraction of the fault's life already used."""
    if progress >= 1.0:
        return STAGE_FAILED
    if progress >= 0.66:
        return STAGE_ADVANCED
    if progress >= 0.33:
        return STAGE_DEVELOPING
    return STAGE_EARLY


@dataclass
class GradualFault:
    """A fault that grows from `start_s` until failure at `start_s + duration_s`.

    `severity = progress ** shape` drives the signals. With shape 1 the drift is linear;
    larger shapes start slowly and accelerate, which is closer to real wear and is harder
    for a linear trend fit to predict. After failure the signals stay at failure level for
    `repair_s`, then the fault is removed (the component is treated as replaced).
    """

    mode: FaultMode
    start_s: float
    duration_s: float
    repair_s: float
    shape: float = 1.5

    @property
    def failure_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def end_s(self) -> float:
        return self.failure_s + self.repair_s

    def progress(self, t: float) -> float:
        return float(np.clip((t - self.start_s) / self.duration_s, 0.0, 1.0))

    def severity(self, t: float) -> float:
        return self.progress(t) ** self.shape

    def stage(self, t: float) -> str:
        return stage_for_progress(self.progress(t))

    def time_to_failure_h(self, t: float) -> float | None:
        """True hours until failure, or None once the failure has happened."""
        if t >= self.failure_s:
            return None
        return (self.failure_s - t) / 3600.0

    def is_over(self, t: float) -> bool:
        return t >= self.end_s

    def adjust_load(self, load: float, t: float) -> float:
        if self.mode is not FaultMode.OVERLOAD:
            return load
        # Overload arrives early in the fault's life and then stays: the damage comes from
        # how long it lasts, which shows up as slowly climbing temperature.
        ramp = min(1.0, self.progress(t) / 0.15)
        return load + max(0.0, OVERLOAD_LEVEL - load) * ramp

    def adjust_signals(self, values: dict[str, float], t: float, rng: np.random.Generator) -> None:
        p = self.progress(t)
        sev = self.severity(t)
        temp_fail = SIGNALS["temperature"].fail

        if self.mode is FaultMode.BEARING_WEAR:
            # Vibration leads and reaches its failure limit at the failure time.
            values["vibration"] += max(0.0, SIGNALS["vibration"].fail - values["vibration"]) * sev
            # Temperature follows only after about a third of the fault's life, and does
            # not have to reach its own limit before the bearing fails.
            late = float(np.clip((p - 0.35) / 0.65, 0.0, 1.0)) ** self.shape
            values["temperature"] += max(0.0, temp_fail - values["temperature"]) * 0.6 * late
            values["current"] += 8.0 * sev  # extra rolling resistance

        elif self.mode is FaultMode.BELT_SLIP:
            # Mean belt speed sags towards the failure limit and gets noisier on the way.
            values["speed"] -= max(0.0, values["speed"] - SIGNALS["speed"].fail) * sev
            values["speed"] += rng.normal(0.0, 0.08 * p)
            values["current"] += 70.0 * sev  # drive works harder against the slip
            values["temperature"] += 6.0 * sev  # friction heat at the pulley

        elif self.mode is FaultMode.OVERLOAD:
            # Load and current are already high through adjust_load; heat builds up until
            # the temperature limit is reached at the failure time.
            values["temperature"] += max(0.0, temp_fail - values["temperature"]) * sev


@dataclass
class SuddenEvent:
    """A short event with no lead-up: a blockage (minutes) or a one-reading spike."""

    mode: FaultMode
    start_s: float
    duration_s: float
    signal: str = ""
    magnitude: float = 1.0
    speed_drop: float = 0.0

    def is_active(self, t: float) -> bool:
        return self.start_s <= t < self.start_s + self.duration_s

    def is_over(self, t: float) -> bool:
        return t >= self.start_s + self.duration_s

    def adjust_load(self, load: float, t: float) -> float:
        if self.mode is FaultMode.BLOCKAGE and self.is_active(t):
            return load + self.magnitude  # material piles up at the station
        return load

    def adjust_signals(self, values: dict[str, float], t: float) -> None:
        if not self.is_active(t):
            return
        if self.mode is FaultMode.BLOCKAGE:
            values["speed"] = max(0.0, values["speed"] - self.speed_drop)
        elif self.mode is FaultMode.SPIKE:
            values[self.signal] *= self.magnitude


def random_gradual_fault(
    rng: np.random.Generator,
    t: float,
    mode: FaultMode | None = None,
    hours_to_failure: float | None = None,
    min_hours: float = 4.0,
    max_hours: float = 24.0,
) -> GradualFault:
    """Create a gradual fault starting at `t` with a random mode, length and shape."""
    if mode is None:
        mode = GRADUAL_MODES[int(rng.integers(len(GRADUAL_MODES)))]
    if mode not in GRADUAL_MODES:
        raise ValueError(f"{mode.value} is not a gradual fault mode")
    hours = hours_to_failure if hours_to_failure is not None else float(rng.uniform(min_hours, max_hours))
    return GradualFault(
        mode=mode,
        start_s=t,
        duration_s=hours * 3600.0,
        repair_s=float(rng.uniform(20.0, 60.0)) * 60.0,
        shape=float(rng.uniform(1.0, 2.2)),
    )


def random_sudden_event(rng: np.random.Generator, t: float, interval_s: float) -> SuddenEvent:
    """Create a blockage or a spike starting at `t`."""
    if rng.random() < 0.5:
        return SuddenEvent(
            mode=FaultMode.BLOCKAGE,
            start_s=t,
            duration_s=float(rng.uniform(60.0, 300.0)),
            magnitude=float(rng.uniform(35.0, 60.0)),
            speed_drop=float(rng.uniform(0.6, 1.4)),
        )
    signal = ("vibration", "temperature", "current")[int(rng.integers(3))]
    return SuddenEvent(
        mode=FaultMode.SPIKE,
        start_s=t,
        duration_s=interval_s,  # exactly one reading
        signal=signal,
        magnitude=float(rng.uniform(1.6, 2.5)),
    )
