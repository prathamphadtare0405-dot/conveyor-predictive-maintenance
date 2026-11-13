"""Sensor simulator for the conveyor: the only source of non-vision data.

Examples (from the project root):

    python -m simulator.sensor_simulator --backfill 48          # 48 h of history to CSV
    python -m simulator.sensor_simulator --speedup 60           # live, 1 simulated hour per minute
    python -m simulator.sensor_simulator --speedup 60 --inject bearing_wear:S04:3

How it is put together:

* `Station` produces the true readings of one sensor station, including any fault.
* `MessInjector` then damages a configurable share of readings the way field data gets
  damaged (missing fields, NaN, wrong types, duplicates, late arrival, stuck values,
  silent stations).
* What is left is written to CSV together with hidden ground-truth columns (`gt_*`) and
  POSTed to the backend without them.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from backend.config import SIGNAL_NAMES, settings
from simulator.fault_models import (
    GRADUAL_MODES,
    STAGE_HEALTHY,
    STAGE_SUDDEN,
    FaultMode,
    GradualFault,
    SuddenEvent,
    random_gradual_fault,
    random_sudden_event,
)

log = logging.getLogger("simulator")

PAYLOAD_FIELDS = ["sensor_id", "station_km", "timestamp", *SIGNAL_NAMES]
TRUTH_FIELDS = ["gt_fault_mode", "gt_stage", "gt_time_to_failure_h", "gt_messy"]
CSV_FIELDS = PAYLOAD_FIELDS + TRUTH_FIELDS

POINT_MESS_KINDS = ("missing_field", "nan", "out_of_range", "wrong_type", "duplicate", "out_of_order")
EPISODE_MESS_KINDS = ("stuck", "silent")
ALL_MESS_KINDS = POINT_MESS_KINDS + EPISODE_MESS_KINDS


def iso_utc(epoch_s: float) -> str:
    """ISO 8601 UTC timestamp with millisecond precision, e.g. 2026-11-13T08:00:05.000Z."""
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class SimConfig:
    sensors: int = settings.sensor_count
    length_km: float = settings.conveyor_length_km
    interval_s: float = settings.sim_interval_s
    belt_speed_mps: float = settings.nominal_belt_speed_mps
    # Expected number of gradual faults per station per simulated day. Sudden events use
    # the same rate.
    fault_rate: float = settings.sim_fault_rate
    # Probability that a reading gets one point corruption. Stuck and silent episodes
    # start with 1/100 of this probability per reading and then last many readings.
    messy_rate: float = settings.sim_messy_rate
    messy_kinds: tuple[str, ...] = ALL_MESS_KINDS
    seed: int = settings.sim_seed

    def __post_init__(self) -> None:
        if self.sensors < 1:
            raise ValueError("sensors must be at least 1")
        if self.interval_s <= 0:
            raise ValueError("interval must be positive")
        if self.fault_rate < 0 or self.messy_rate < 0:
            raise ValueError("fault rate and messy rate cannot be negative")
        unknown = set(self.messy_kinds) - set(ALL_MESS_KINDS)
        if unknown:
            raise ValueError(f"unknown messy kinds: {sorted(unknown)}")


@dataclass
class Emission:
    """One reading as it leaves the simulator: what is sent, plus what only the CSV keeps."""

    payload: dict[str, Any]
    truth: dict[str, Any]
    messy: str = ""

    def csv_row(self) -> dict[str, Any]:
        row = dict(self.payload)
        row.update(self.truth)
        row["gt_messy"] = self.messy
        return row


# --------------------------------------------------------------------------------------
# Healthy process + faults for one station
# --------------------------------------------------------------------------------------


class FeedProfile:
    """Material fed onto the belt at the tail end, in % of rated load, as a function of time.

    It is a deterministic function so that a station at kilometre x can look up what was
    loaded x / belt_speed seconds earlier: the same material passes every station in turn.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.phase_slow = float(rng.uniform(0.0, 2.0 * math.pi))
        self.phase_fast = float(rng.uniform(0.0, 2.0 * math.pi))

    def at(self, t: float) -> float:
        slow = 12.0 * math.sin(2.0 * math.pi * t / (3.1 * 3600.0) + self.phase_slow)
        fast = 5.0 * math.sin(2.0 * math.pi * t / (47.0 * 60.0) + self.phase_fast)
        return 70.0 + slow + fast


def ambient_temperature(t: float) -> float:
    """Daily ambient cycle in degC: coolest around 03:00, warmest around 15:00."""
    hour = (t % 86400.0) / 3600.0
    return 25.0 + 8.0 * math.sin(2.0 * math.pi * (hour - 9.0) / 24.0)


class Station:
    """One sensor station: healthy behaviour, at most one gradual fault, at most one sudden event."""

    THERMAL_TIME_CONSTANT_S = 600.0  # housing temperature lags load and ambient changes
    LOAD_NOISE_TIME_CONSTANT_S = 600.0
    NOISE_SD = {"temperature": 0.5, "vibration": 0.12, "speed": 0.015, "load": 0.8, "current": 3.0}

    def __init__(self, index: int, km: float, cfg: SimConfig, feed: FeedProfile, rng: np.random.Generator) -> None:
        self.sensor_id = settings.sensor_id(index)
        self.km = km
        self.cfg = cfg
        self.feed = feed
        self.rng = rng
        # Small fixed differences between stations, so no two sensors read identically.
        self.temp_offset = float(rng.normal(0.0, 1.5))
        self.vib_offset = float(rng.normal(0.0, 0.2))
        self.current_offset = float(rng.normal(0.0, 4.0))
        self.load_noise = 0.0
        self.temp_state: float | None = None
        self.fault: GradualFault | None = None
        self.event: SuddenEvent | None = None
        self.fault_log: list[GradualFault] = []
        self.event_count = 0

    def start_fault(self, fault: GradualFault) -> None:
        self.fault = fault
        self.fault_log.append(fault)

    def _update_faults(self, t: float) -> None:
        dt = self.cfg.interval_s
        if self.fault is not None and self.fault.is_over(t):
            self.fault = None
        if self.event is not None and self.event.is_over(t):
            self.event = None
        chance = self.cfg.fault_rate * dt / 86400.0
        if self.fault is None and self.rng.random() < chance:
            self.start_fault(random_gradual_fault(self.rng, t))
        if self.event is None and self.rng.random() < chance:
            self.event = random_sudden_event(self.rng, t, dt)
            self.event_count += 1

    def read(self, t: float) -> tuple[dict[str, float], dict[str, Any]]:
        """Return the true signal values at time `t` and the ground-truth labels."""
        self._update_faults(t)
        dt = self.cfg.interval_s
        rng = self.rng

        # Load: what was fed at the tail, delayed by the travel time to this station.
        travel_s = self.km * 1000.0 / self.cfg.belt_speed_mps
        k = dt / self.LOAD_NOISE_TIME_CONSTANT_S
        self.load_noise += -self.load_noise * k + 2.0 * math.sqrt(2.0 * k) * rng.normal()
        load = self.feed.at(t - travel_s) + self.load_noise
        if self.event is not None:
            load = self.event.adjust_load(load, t)
        if self.fault is not None:
            load = self.fault.adjust_load(load, t)
        load = max(0.0, load)

        # Temperature follows ambient and load with a lag.
        target = ambient_temperature(t) + 12.0 + 0.22 * load + self.temp_offset
        if self.temp_state is None:
            self.temp_state = target
        self.temp_state += (target - self.temp_state) * min(1.0, dt / self.THERMAL_TIME_CONSTANT_S)

        values = {
            "temperature": self.temp_state,
            "vibration": 2.0 + 0.008 * load + self.vib_offset,
            "speed": self.cfg.belt_speed_mps - 0.002 * (load - 70.0),
            "load": load,
            "current": 120.0 + 2.0 * load + self.current_offset,
        }
        if self.fault is not None:
            self.fault.adjust_signals(values, t, rng)
        if self.event is not None:
            self.event.adjust_signals(values, t)
        # Measurement noise goes on last, so a failed component still gives a noisy
        # reading and is not mistaken for a stuck sensor.
        for name, sd in self.NOISE_SD.items():
            values[name] += rng.normal(0.0, sd)

        values = {
            "temperature": round(values["temperature"], 2),
            "vibration": round(max(0.0, values["vibration"]), 3),
            "speed": round(max(0.0, values["speed"]), 3),
            "load": round(max(0.0, values["load"]), 2),
            "current": round(max(0.0, values["current"]), 2),
        }
        return values, self._truth(t)

    def _truth(self, t: float) -> dict[str, Any]:
        if self.fault is not None:
            ttf = self.fault.time_to_failure_h(t)
            return {
                "gt_fault_mode": self.fault.mode.value,
                "gt_stage": self.fault.stage(t),
                "gt_time_to_failure_h": "" if ttf is None else round(ttf, 4),
            }
        if self.event is not None and self.event.is_active(t):
            return {"gt_fault_mode": self.event.mode.value, "gt_stage": STAGE_SUDDEN, "gt_time_to_failure_h": ""}
        return {"gt_fault_mode": FaultMode.NONE.value, "gt_stage": STAGE_HEALTHY, "gt_time_to_failure_h": ""}


# --------------------------------------------------------------------------------------
# Messy data
# --------------------------------------------------------------------------------------


class MessInjector:
    """Damages readings on purpose. Every damaged reading is labelled in `gt_messy`.

    Point corruptions (one reading):
      missing_field  a field is absent (usually a signal, sometimes the timestamp)
      nan            a signal is NaN
      out_of_range   a signal has a physically impossible value
      wrong_type     a signal is a string such as "ERR" or "45.2C"
      duplicate      the same reading is sent again one tick later
      out_of_order   the reading is held back and sent 2-6 ticks late

    Episodes (many readings):
      stuck          one signal of a station repeats the same value
      silent         a station sends nothing at all
    """

    EPISODE_START_FACTOR = 0.01

    def __init__(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        self.rate = cfg.messy_rate
        self.rng = rng
        self.point_kinds = [k for k in POINT_MESS_KINDS if k in cfg.messy_kinds]
        self.episode_kinds = [k for k in EPISODE_MESS_KINDS if k in cfg.messy_kinds]
        self.stuck: dict[str, tuple[str, Any, int]] = {}  # sensor -> (signal, value, readings left)
        self.silent: dict[str, int] = {}  # sensor -> ticks left
        self.held: list[tuple[int, Emission]] = []  # (tick at which to release, emission)
        self.counts: Counter[str] = Counter()

    def process(self, readings: list[Emission], tick: int) -> list[Emission]:
        out: list[Emission] = [e for due, e in self.held if due <= tick]
        self.held = [(due, e) for due, e in self.held if due > tick]
        if self.rate <= 0:
            return out + readings

        for em in readings:
            sensor = em.payload["sensor_id"]

            # --- episodes -------------------------------------------------------------
            if sensor in self.silent:
                self.silent[sensor] -= 1
                if self.silent[sensor] <= 0:
                    del self.silent[sensor]
                self.counts["silent_readings"] += 1
                continue
            if sensor in self.stuck:
                signal, value, left = self.stuck[sensor]
                em.payload[signal] = value
                em.messy = "stuck"
                self.counts["stuck_readings"] += 1
                if left <= 1:
                    del self.stuck[sensor]
                else:
                    self.stuck[sensor] = (signal, value, left - 1)
                out.append(em)
                continue
            if self.episode_kinds and self.rng.random() < self.rate * self.EPISODE_START_FACTOR * len(self.episode_kinds):
                kind = self.episode_kinds[int(self.rng.integers(len(self.episode_kinds)))]
                self.counts[f"{kind}_episodes"] += 1
                if kind == "silent":
                    self.silent[sensor] = int(self.rng.integers(24, 121))  # 2-10 min at 5 s
                    self.counts["silent_readings"] += 1
                    continue
                signal = SIGNAL_NAMES[int(self.rng.integers(len(SIGNAL_NAMES)))]
                self.stuck[sensor] = (signal, em.payload[signal], int(self.rng.integers(10, 41)))
                em.messy = "stuck"
                self.counts["stuck_readings"] += 1
                out.append(em)
                continue

            # --- point corruptions ----------------------------------------------------
            if self.point_kinds and self.rng.random() < self.rate:
                kind = self.point_kinds[int(self.rng.integers(len(self.point_kinds)))]
                self.counts[kind] += 1
                em.messy = kind
                if kind == "out_of_order":
                    self.held.append((tick + int(self.rng.integers(2, 7)), em))
                    continue
                if kind == "duplicate":
                    copy = Emission(dict(em.payload), dict(em.truth), "duplicate")
                    self.held.append((tick + 1, copy))
                    em.messy = ""  # the first copy is a normal reading; the repeat is the duplicate
                else:
                    self._corrupt(em.payload, kind)
            out.append(em)
        return out

    def _corrupt(self, payload: dict[str, Any], kind: str) -> None:
        signal = SIGNAL_NAMES[int(self.rng.integers(len(SIGNAL_NAMES)))]
        if kind == "missing_field":
            # Mostly a signal goes missing (reading still partly usable); sometimes the
            # timestamp does (reading unusable, the backend has to reject it).
            target = "timestamp" if self.rng.random() < 0.15 else signal
            payload.pop(target, None)
        elif kind == "nan":
            payload[signal] = float("nan")
        elif kind == "out_of_range":
            payload[signal] = (-999.0, 9999.0, -1.0 * abs(payload[signal]) - 50.0, 1.0e6)[int(self.rng.integers(4))]
        elif kind == "wrong_type":
            value = payload[signal]
            payload[signal] = ("ERR", "N/A", f"{value}C", "")[int(self.rng.integers(4))]

    def flush(self) -> list[Emission]:
        """Release readings still held back when the run ends."""
        out = [e for _, e in self.held]
        self.held = []
        return out


# --------------------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------------------


class ConveyorSimulator:
    """Generates one batch of readings per tick for all stations."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.sensors + 2)
        self.feed = FeedProfile(np.random.default_rng(seeds[0]))
        self.mess = MessInjector(cfg, np.random.default_rng(seeds[1]))
        positions = [round(cfg.length_km / cfg.sensors * (i + 0.5), 3) for i in range(cfg.sensors)]
        self.stations = [
            Station(i, positions[i], cfg, self.feed, np.random.default_rng(seeds[i + 2])) for i in range(cfg.sensors)
        ]
        self.tick_index = 0

    def station(self, sensor_id: str) -> Station:
        for st in self.stations:
            if st.sensor_id == sensor_id:
                return st
        raise KeyError(f"no station {sensor_id}; stations are {self.stations[0].sensor_id}..{self.stations[-1].sensor_id}")

    def inject(self, mode: FaultMode, sensor_id: str, hours_to_failure: float, t: float) -> GradualFault:
        """Start a chosen gradual fault now, replacing whatever the station had."""
        st = self.station(sensor_id)
        fault = random_gradual_fault(st.rng, t, mode=mode, hours_to_failure=hours_to_failure)
        st.start_fault(fault)
        return fault

    def tick(self, t: float) -> list[Emission]:
        """Produce the readings for simulated time `t` (epoch seconds), after damage."""
        clean = []
        stamp = iso_utc(t)
        for st in self.stations:
            values, truth = st.read(t)
            payload = {"sensor_id": st.sensor_id, "station_km": st.km, "timestamp": stamp, **values}
            clean.append(Emission(payload, truth))
        out = self.mess.process(clean, self.tick_index)
        self.tick_index += 1
        return out

    def fault_summary(self) -> list[dict[str, Any]]:
        rows = []
        for st in self.stations:
            for f in st.fault_log:
                rows.append(
                    {
                        "sensor_id": st.sensor_id,
                        "mode": f.mode.value,
                        "start": iso_utc(f.start_s),
                        "failure": iso_utc(f.failure_s),
                        "hours_to_failure": round(f.duration_s / 3600.0, 2),
                    }
                )
        return sorted(rows, key=lambda r: r["start"])


# --------------------------------------------------------------------------------------
# Output: CSV and backend
# --------------------------------------------------------------------------------------


class CsvSink:
    def __init__(self, path: Path, append: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not (append and path.exists() and path.stat().st_size > 0)
        self.path = path
        self._fh = path.open("a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS, restval="", extrasaction="ignore")
        if new_file:
            self._writer.writeheader()
        self.rows = 0

    def write(self, emissions: list[Emission]) -> None:
        for em in emissions:
            self._writer.writerow(em.csv_row())
        self.rows += len(emissions)

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


class BackendPoster:
    """Sends readings to the backend and survives the backend being down.

    Readings wait in a bounded buffer. A failed send starts an exponential backoff
    (1 s, 2 s, 4 s ... capped at 30 s); generation carries on meanwhile and the buffer is
    sent in order once the backend answers again. If the buffer overflows, the oldest
    readings are discarded and the loss is counted and logged.
    """

    CHUNK = 500
    MAX_BACKOFF_S = 30.0

    def __init__(self, base_url: str, max_buffer: int = 50_000, timeout_s: tuple[float, float] = (1.0, 15.0)) -> None:
        # timeout_s is (connect, read): a short connect timeout keeps a dead backend from
        # stalling generation; the read timeout allows for the backend analysing a batch.
        self.url = base_url.rstrip("/") + "/api/sensors/readings"
        self.buffer: deque[dict[str, Any]] = deque()
        self.max_buffer = max_buffer
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.failures = 0
        self.next_attempt = 0.0
        self.stats: Counter[str] = Counter()

    def submit(self, payloads: list[dict[str, Any]]) -> None:
        self.buffer.extend(payloads)
        overflow = len(self.buffer) - self.max_buffer
        if overflow > 0:
            for _ in range(overflow):
                self.buffer.popleft()
            self.stats["discarded_buffer_full"] += overflow
            log.error("Send buffer full: discarded the %d oldest unsent readings", overflow)
        self.flush()

    def flush(self, force: bool = False) -> None:
        if not force and time.monotonic() < self.next_attempt:
            return
        while self.buffer:
            chunk = [self.buffer[i] for i in range(min(self.CHUNK, len(self.buffer)))]
            try:
                # json.dumps is used directly because `requests` refuses to encode NaN,
                # and sending NaN is one of the things this simulator does on purpose.
                resp = self.session.post(
                    self.url,
                    data=json.dumps(chunk),
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                self._back_off(f"{type(exc).__name__}")
                return
            if resp.status_code >= 500:
                self._back_off(f"HTTP {resp.status_code}")
                return
            for _ in chunk:
                self.buffer.popleft()
            if self.failures:
                log.info("Backend is reachable again; %d readings still buffered", len(self.buffer))
            self.failures = 0
            self.next_attempt = 0.0
            if resp.status_code >= 400:
                # The backend understood the request and refused it. Retrying the same
                # bytes cannot help, so the chunk is counted as refused and not resent.
                self.stats["refused_by_backend"] += len(chunk)
                log.error("Backend refused %d readings: HTTP %d %s", len(chunk), resp.status_code, resp.text[:200])
                continue
            self.stats["sent"] += len(chunk)
            try:
                body = resp.json()
            except ValueError:
                continue
            if isinstance(body, dict):
                for key in ("accepted", "flagged", "rejected", "duplicates"):
                    if isinstance(body.get(key), int):
                        self.stats[f"backend_{key}"] += body[key]

    def _back_off(self, reason: str) -> None:
        self.failures += 1
        delay = min(self.MAX_BACKOFF_S, 2.0 ** (self.failures - 1))
        self.next_attempt = time.monotonic() + delay
        log.warning(
            "Backend not reachable (%s). Retry in %.0f s; %d readings buffered; still generating.",
            reason,
            delay,
            len(self.buffer),
        )


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


@dataclass
class RunStats:
    ticks: int = 0
    emitted: int = 0
    by_mode: Counter = field(default_factory=Counter)


def _apply_injections(sim: ConveyorSimulator, specs: list[str], t: float) -> None:
    for spec in specs:
        fault = sim.inject(*parse_injection(spec), t)
        log.info("Injected %s, failure due at %s", spec, iso_utc(fault.failure_s))


def parse_injection(spec: str) -> tuple[FaultMode, str, float]:
    """Parse MODE:SENSOR:HOURS, for example bearing_wear:S04:3."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--inject expects MODE:SENSOR:HOURS, got {spec!r}")
    try:
        mode = FaultMode(parts[0])
        hours = float(parts[2])
    except ValueError as exc:
        raise ValueError(f"--inject {spec!r}: {exc}") from exc
    if mode not in GRADUAL_MODES:
        raise ValueError(f"--inject mode must be one of {[m.value for m in GRADUAL_MODES]}")
    if hours <= 0:
        raise ValueError("--inject hours must be positive")
    return mode, parts[1], hours


def run_backfill(
    sim: ConveyorSimulator,
    hours: float,
    sink: CsvSink,
    poster: BackendPoster | None,
    end_s: float | None = None,
    injections: list[str] | None = None,
) -> RunStats:
    """Generate `hours` of history ending now (or at `end_s`) as fast as possible."""
    interval = sim.cfg.interval_s
    end = end_s if end_s is not None else time.time()
    end = math.floor(end / interval) * interval
    ticks = int(round(hours * 3600.0 / interval))
    start = end - ticks * interval
    _apply_injections(sim, injections or [], start)
    stats = RunStats()
    pending: list[dict[str, Any]] = []
    try:
        for i in range(ticks):
            emissions = sim.tick(start + i * interval)
            _record(emissions, sink, stats)
            if poster is not None:
                pending.extend(e.payload for e in emissions)
                if len(pending) >= BackendPoster.CHUNK:
                    poster.submit(pending)
                    pending = []
            stats.ticks += 1
    except KeyboardInterrupt:
        log.info("Interrupted; stopping backfill early.")
    tail = sim.mess.flush()
    _record(tail, sink, stats)
    if poster is not None:
        poster.submit(pending + [e.payload for e in tail])
        _drain(poster)
    return stats


def run_live(
    sim: ConveyorSimulator,
    speedup: float,
    duration_s: float | None,
    sink: CsvSink,
    poster: BackendPoster | None,
    injections: list[str] | None = None,
) -> RunStats:
    """Generate readings in step with the clock; `speedup` makes simulated time run faster.

    Simulated time starts at the current wall-clock time. One tick is always one sampling
    interval of simulated time; with speedup N the wait between ticks is interval / N.
    """
    interval = sim.cfg.interval_s
    t = math.floor(time.time() / interval) * interval
    _apply_injections(sim, injections or [], t)
    stats = RunStats()
    wall_start = time.monotonic()
    next_tick = wall_start
    next_report = wall_start + 10.0
    try:
        while duration_s is None or time.monotonic() - wall_start < duration_s:
            emissions = sim.tick(t)
            _record(emissions, sink, stats)
            if poster is not None:
                poster.submit([e.payload for e in emissions])
            stats.ticks += 1
            t += interval
            now = time.monotonic()
            if now >= next_report:
                active = [f"{s.sensor_id}:{s.fault.mode.value}" for s in sim.stations if s.fault is not None]
                log.info(
                    "sim time %s | %d readings | sent %d, buffered %d | active faults: %s",
                    iso_utc(t),
                    stats.emitted,
                    poster.stats["sent"] if poster else 0,
                    len(poster.buffer) if poster else 0,
                    ", ".join(active) or "none",
                )
                next_report = now + 10.0
            next_tick += interval / speedup
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # running behind; do not try to catch up in a burst
    except KeyboardInterrupt:
        log.info("Ctrl+C received; stopping.")
    tail = sim.mess.flush()
    _record(tail, sink, stats)
    if poster is not None:
        poster.submit([e.payload for e in tail])
        _drain(poster, max_wait_s=5.0)
    return stats


def _record(emissions: list[Emission], sink: CsvSink, stats: RunStats) -> None:
    sink.write(emissions)
    stats.emitted += len(emissions)
    for em in emissions:
        stats.by_mode[em.truth["gt_fault_mode"]] += 1


def _drain(poster: BackendPoster, max_wait_s: float = 60.0) -> None:
    """Try to empty the send buffer before exiting, without hanging if the backend is down."""
    deadline = time.monotonic() + max_wait_s
    try:
        while poster.buffer and time.monotonic() < deadline:
            poster.flush(force=True)
            if poster.buffer:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    if poster.buffer:
        log.warning("Exiting with %d readings never delivered to the backend (they are in the CSV).", len(poster.buffer))


def print_summary(sim: ConveyorSimulator, stats: RunStats, sink: CsvSink, poster: BackendPoster | None) -> None:
    print("\n=== Simulator summary ===")
    print(f"ticks: {stats.ticks}   readings emitted: {stats.emitted}   CSV: {sink.path} ({sink.rows} rows written)")
    print("readings by true fault mode: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.by_mode.items())))
    mess = sim.mess.counts
    print("messy data injected: " + (", ".join(f"{k}={v}" for k, v in sorted(mess.items())) or "none"))
    faults = sim.fault_summary()
    print(f"gradual faults started: {len(faults)}   sudden events: {sum(s.event_count for s in sim.stations)}")
    for f in faults:
        print(f"  {f['sensor_id']} {f['mode']:<13} start {f['start']}  failure {f['failure']}  ({f['hours_to_failure']} h)")
    if poster is not None:
        print("backend: " + (", ".join(f"{k}={v}" for k, v in sorted(poster.stats.items())) or "nothing sent"))
        print(f"backend: {len(poster.buffer)} readings left unsent")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simulator.sensor_simulator",
        description="Simulate conveyor sensor stations, including gradual faults and messy data.",
    )
    p.add_argument("--sensors", type=int, default=settings.sensor_count, help="number of sensor stations (default %(default)s)")
    p.add_argument("--interval", type=float, default=settings.sim_interval_s, help="seconds of simulated time between readings (default %(default)s)")
    p.add_argument("--speedup", type=float, default=settings.sim_speedup, help="time acceleration in live mode; 60 = one simulated hour per minute (default %(default)s)")
    p.add_argument("--fault-rate", type=float, default=settings.sim_fault_rate, help="expected gradual faults per station per simulated day (default %(default)s)")
    p.add_argument("--messy-rate", type=float, default=settings.sim_messy_rate, help="share of readings corrupted on purpose; 0 turns it off (default %(default)s)")
    p.add_argument("--messy-kinds", default=",".join(ALL_MESS_KINDS), help="comma-separated kinds of corruption to use (default: all)")
    p.add_argument("--seed", type=int, default=settings.sim_seed, help="random seed (default %(default)s)")
    p.add_argument("--duration", type=float, default=None, help="live mode: stop after this many real seconds (default: run until Ctrl+C)")
    p.add_argument("--backfill", type=float, default=None, metavar="HOURS", help="generate HOURS of history ending now, as fast as possible, then exit")
    p.add_argument("--inject", action="append", default=[], metavar="MODE:SENSOR:HOURS", help="start a gradual fault at the beginning of the run, e.g. bearing_wear:S04:3 (repeatable)")
    p.add_argument("--csv", type=Path, default=None, help="CSV output path (default data/backfill.csv or data/live.csv)")
    p.add_argument("--backend-url", default=settings.backend_url, help="backend base URL (default %(default)s)")
    p.add_argument("--post", action="store_true", help="backfill mode: also send the history to the backend (off by default)")
    p.add_argument("--no-post", action="store_true", help="live mode: write the CSV only, do not contact the backend")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.speedup <= 0:
        raise SystemExit("--speedup must be positive")
    try:
        cfg = SimConfig(
            sensors=args.sensors,
            interval_s=args.interval,
            fault_rate=args.fault_rate,
            messy_rate=args.messy_rate,
            messy_kinds=tuple(k.strip() for k in args.messy_kinds.split(",") if k.strip()),
            seed=args.seed,
        )
        sim = ConveyorSimulator(cfg)
        for spec in args.inject:
            sim.station(parse_injection(spec)[1])  # fail early on a bad spec
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc.args[0] if exc.args else exc}") from exc

    backfill = args.backfill is not None
    csv_path = args.csv or settings.data_dir / ("backfill.csv" if backfill else "live.csv")
    if backfill and csv_path.exists():
        log.info("Replacing existing %s", csv_path)
    sink = CsvSink(csv_path, append=not backfill)
    post = args.post if backfill else not args.no_post
    poster = BackendPoster(args.backend_url) if post else None

    try:
        if backfill:
            log.info("Backfill: %.1f h, %d stations, every %.0f s", args.backfill, cfg.sensors, cfg.interval_s)
            stats = run_backfill(sim, args.backfill, sink, poster, injections=args.inject)
        else:
            log.info(
                "Live: %d stations, every %.0f s of simulated time, speedup %.0fx, %s. Ctrl+C to stop.",
                cfg.sensors,
                cfg.interval_s,
                args.speedup,
                f"posting to {poster.url}" if poster else "CSV only",
            )
            stats = run_live(sim, args.speedup, args.duration, sink, poster, injections=args.inject)
    finally:
        sink.close()
    print_summary(sim, stats, sink, poster)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
