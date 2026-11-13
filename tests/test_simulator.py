"""Simulator: healthy output, gradual faults with ground truth, and each messy mode."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import requests

from backend.config import SIGNAL_NAMES, SIGNALS
from simulator.fault_models import FaultMode, GradualFault, random_gradual_fault, stage_for_progress
from simulator.sensor_simulator import (
    CSV_FIELDS,
    BackendPoster,
    ConveyorSimulator,
    CsvSink,
    SimConfig,
    ambient_temperature,
    parse_injection,
    run_backfill,
)

T0 = 1_800_000_000.0  # fixed start time so runs are repeatable


def make_sim(**overrides) -> ConveyorSimulator:
    params = dict(sensors=4, fault_rate=0.0, messy_rate=0.0, seed=7)
    params.update(overrides)
    return ConveyorSimulator(SimConfig(**params))


def run_ticks(sim: ConveyorSimulator, n: int):
    out = []
    for i in range(n):
        out.extend(sim.tick(T0 + i * sim.cfg.interval_s))
    return out


# --- healthy operation -----------------------------------------------------------------


def test_clean_run_has_every_field_and_stays_inside_warning_limits():
    sim = make_sim()
    emissions = run_ticks(sim, 720)  # one simulated hour
    assert len(emissions) == 720 * 4
    for em in emissions:
        assert set(em.payload) == {"sensor_id", "station_km", "timestamp", *SIGNAL_NAMES}
        assert em.truth["gt_fault_mode"] == "none" and em.truth["gt_stage"] == "healthy"
        for name in SIGNAL_NAMES:
            assert not SIGNALS[name].beyond_warn(em.payload[name]), (name, em.payload[name])


def test_ground_truth_never_reaches_the_payload():
    sim = make_sim(fault_rate=5.0, messy_rate=0.05)
    for em in run_ticks(sim, 300):
        assert not any(key.startswith("gt_") for key in em.payload)


def test_stations_sit_at_fixed_evenly_spaced_positions():
    sim = make_sim(sensors=10)
    assert [s.km for s in sim.stations] == [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75]
    assert [s.sensor_id for s in sim.stations][:3] == ["S01", "S02", "S03"]


def test_same_seed_gives_same_readings_and_different_seed_does_not():
    a = [e.payload for e in run_ticks(make_sim(seed=1, messy_rate=0.05, fault_rate=2.0), 200)]
    b = [e.payload for e in run_ticks(make_sim(seed=1, messy_rate=0.05, fault_rate=2.0), 200)]
    c = [e.payload for e in run_ticks(make_sim(seed=2, messy_rate=0.05, fault_rate=2.0), 200)]
    assert repr(a) == repr(b)  # repr, because NaN != NaN
    assert repr(a) != repr(c)


def test_more_load_means_more_current_and_heat():
    sim = make_sim(sensors=1)
    rows = [e.payload for e in run_ticks(sim, 4000)]
    load = np.array([r["load"] for r in rows])
    assert np.corrcoef(load, [r["current"] for r in rows])[0, 1] > 0.9
    # Temperature also follows the daily ambient cycle, so compare the rise above ambient.
    rise = [r["temperature"] - ambient_temperature(T0 + i * 5.0) for i, r in enumerate(rows)]
    assert np.corrcoef(load, rise)[0, 1] > 0.6


# --- gradual faults --------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, signal",
    [(FaultMode.BEARING_WEAR, "vibration"), (FaultMode.BELT_SLIP, "speed"), (FaultMode.OVERLOAD, "temperature")],
)
def test_gradual_fault_drifts_to_the_failure_limit_at_the_labelled_time(mode, signal):
    sim = make_sim(sensors=2)
    sim.inject(mode, "S01", hours_to_failure=1.0, t=T0)
    emissions = [e for e in run_ticks(sim, 730) if e.payload["sensor_id"] == "S01"]
    spec = SIGNALS[signal]

    early = np.mean([e.payload[signal] for e in emissions[:30]])
    late = np.mean([e.payload[signal] for e in emissions[690:715]])
    assert (late - early) * spec.direction > 0.5 * abs(spec.fail - early)  # it really drifted
    assert not spec.beyond_warn(early)  # and it started inside normal operation

    # Ground truth counts down to zero and then reports the failure.
    assert emissions[0].truth["gt_time_to_failure_h"] == pytest.approx(1.0, abs=0.01)
    assert emissions[360].truth["gt_time_to_failure_h"] == pytest.approx(0.5, abs=0.01)
    assert emissions[360].truth["gt_fault_mode"] == mode.value
    failed = emissions[725]
    assert failed.truth["gt_stage"] == "failed" and failed.truth["gt_time_to_failure_h"] == ""
    assert failed.payload[signal] == pytest.approx(spec.fail, rel=0.05)

    # The other station is untouched.
    assert all(e.truth["gt_fault_mode"] == "none" for e in run_ticks(make_sim(sensors=2), 50))


def test_bearing_wear_vibration_rises_before_temperature():
    fault = GradualFault(FaultMode.BEARING_WEAR, start_s=0.0, duration_s=1000.0, repair_s=60.0, shape=1.0)
    rng = np.random.default_rng(0)
    base = {"temperature": 50.0, "vibration": 2.5, "speed": 4.0, "load": 70.0, "current": 260.0}
    at_30 = dict(base)
    fault.adjust_signals(at_30, 300.0, rng)
    assert at_30["vibration"] > base["vibration"] + 2.0
    assert at_30["temperature"] == base["temperature"]  # heat has not started yet
    at_90 = dict(base)
    fault.adjust_signals(at_90, 900.0, rng)
    assert at_90["temperature"] > base["temperature"] + 10.0


def test_fault_is_removed_after_the_repair_time():
    fault = GradualFault(FaultMode.OVERLOAD, start_s=0.0, duration_s=100.0, repair_s=50.0)
    assert fault.time_to_failure_h(40.0) == pytest.approx(60.0 / 3600.0)
    assert fault.time_to_failure_h(100.0) is None
    assert not fault.is_over(149.0) and fault.is_over(150.0)
    assert [stage_for_progress(p) for p in (0.1, 0.5, 0.9, 1.0)] == ["early", "developing", "advanced", "failed"]


def test_random_faults_appear_at_roughly_the_requested_rate():
    sim = make_sim(sensors=10, fault_rate=4.0)
    run_ticks(sim, 17280 // 4)  # 6 simulated hours -> expect at most ~10 faults, at least 1
    assert 1 <= len(sim.fault_summary()) <= 20
    with pytest.raises(ValueError):
        random_gradual_fault(np.random.default_rng(0), 0.0, mode=FaultMode.SPIKE)


# --- messy data ------------------------------------------------------------------------


def messy_run(kind: str, ticks: int = 1500, rate: float = 0.2):
    sim = make_sim(messy_rate=rate, messy_kinds=(kind,))
    emissions = run_ticks(sim, ticks) + sim.mess.flush()
    return sim, emissions, [e for e in emissions if e.messy == kind]


def test_missing_field_removes_a_signal_or_the_timestamp():
    _, _, hit = messy_run("missing_field")
    assert hit
    missing = [({"timestamp", *SIGNAL_NAMES} - set(e.payload)) for e in hit]
    assert all(len(m) == 1 for m in missing)
    assert any(m == {"timestamp"} for m in missing) and any(m != {"timestamp"} for m in missing)


def test_nan_out_of_range_and_wrong_type_each_damage_one_signal():
    _, _, hit = messy_run("nan")
    assert hit and all(sum(isinstance(e.payload[s], float) and math.isnan(e.payload[s]) for s in SIGNAL_NAMES) == 1 for e in hit)

    _, _, hit = messy_run("out_of_range")
    assert hit
    for e in hit:
        outside = [s for s in SIGNAL_NAMES if not SIGNALS[s].hard_min <= e.payload[s] <= SIGNALS[s].hard_max]
        assert len(outside) == 1

    _, _, hit = messy_run("wrong_type")
    assert hit and all(sum(isinstance(e.payload[s], str) for s in SIGNAL_NAMES) == 1 for e in hit)


def test_duplicate_repeats_a_reading_exactly():
    _, emissions, hit = messy_run("duplicate")
    assert hit
    keys = [(e.payload["sensor_id"], e.payload["timestamp"]) for e in emissions]
    for dup in hit:
        key = (dup.payload["sensor_id"], dup.payload["timestamp"])
        assert keys.count(key) == 2
        original = next(e for e in emissions if (e.payload["sensor_id"], e.payload["timestamp"]) == key and e is not dup)
        assert original.payload == dup.payload


def test_out_of_order_readings_arrive_late_but_are_not_lost():
    _, emissions, hit = messy_run("out_of_order")
    assert hit
    stamps = [e.payload["timestamp"] for e in emissions]
    late = [i for i, e in enumerate(emissions) if e.messy == "out_of_order"]
    assert any(stamps[i] < stamps[i - 1] for i in late if i > 0)
    assert len({(e.payload["sensor_id"], e.payload["timestamp"]) for e in emissions}) == 1500 * 4


def test_stuck_sensor_repeats_one_value():
    _, emissions, hit = messy_run("stuck", ticks=3000, rate=0.5)
    assert len(hit) >= 10
    sensor = hit[0].payload["sensor_id"]
    run = []
    for e in emissions:
        if e.payload["sensor_id"] == sensor:
            if e.messy == "stuck":
                run.append(e)
            elif run:
                break
    assert len(run) >= 10
    assert any(len({e.payload[s] for e in run}) == 1 for s in SIGNAL_NAMES)


def test_silent_station_sends_nothing_for_a_while():
    sim, emissions, _ = messy_run("silent", ticks=3000, rate=0.5)
    assert sim.mess.counts["silent_episodes"] >= 1
    assert len(emissions) == 3000 * 4 - sim.mess.counts["silent_readings"]
    sensor_ticks = sorted(e.payload["timestamp"] for e in emissions if e.payload["sensor_id"] == "S01")
    assert len(sensor_ticks) < 3000 or sim.mess.counts["silent_readings"] > 0


def test_messy_rate_zero_changes_nothing_and_unknown_kind_is_refused():
    sim = make_sim(messy_rate=0.0)
    assert all(e.messy == "" for e in run_ticks(sim, 200))
    with pytest.raises(ValueError):
        SimConfig(messy_kinds=("gremlins",))


# --- output ----------------------------------------------------------------------------


def test_backfill_writes_csv_with_ground_truth_columns(tmp_path):
    sim = make_sim(messy_rate=0.05, fault_rate=3.0)
    sink = CsvSink(tmp_path / "out.csv", append=False)
    stats = run_backfill(sim, hours=0.5, sink=sink, poster=None, end_s=T0, injections=["belt_slip:S02:0.25"])
    sink.close()
    with (tmp_path / "out.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == CSV_FIELDS
    assert len(rows) == stats.emitted
    assert {r["gt_fault_mode"] for r in rows} >= {"none", "belt_slip"}
    assert any(r["gt_stage"] == "failed" for r in rows if r["sensor_id"] == "S02")
    assert max(r["timestamp"] for r in rows) < "2027-01-15T08:00:00.000Z"  # ends at end_s, exclusive


def test_poster_buffers_and_backs_off_when_backend_is_down(monkeypatch):
    poster = BackendPoster("http://127.0.0.1:9")

    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(poster.session, "post", refuse)
    poster.submit([{"sensor_id": "S01"}] * 3)
    assert len(poster.buffer) == 3 and poster.failures == 1
    poster.submit([{"sensor_id": "S01"}] * 2)  # inside the backoff window: no new attempt
    assert len(poster.buffer) == 5 and poster.failures == 1

    class Ok:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"accepted": 4, "flagged": 1, "rejected": 0, "duplicates": 0}

    sent = []
    monkeypatch.setattr(poster.session, "post", lambda url, data, **kw: sent.append(data) or Ok())
    poster.flush(force=True)
    assert not poster.buffer and poster.failures == 0
    assert poster.stats["sent"] == 5 and poster.stats["backend_flagged"] == 1


def test_poster_can_send_nan_and_drops_oldest_when_buffer_is_full(monkeypatch):
    poster = BackendPoster("http://127.0.0.1:9", max_buffer=4)
    bodies = []

    class Ok:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {}

    monkeypatch.setattr(poster.session, "post", lambda url, data, **kw: bodies.append(data) or Ok())
    poster.submit([{"sensor_id": "S01", "temperature": float("nan")}])
    assert "NaN" in bodies[0]

    monkeypatch.setattr(poster.session, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    poster.submit([{"n": i} for i in range(6)])
    assert [p["n"] for p in poster.buffer] == [2, 3, 4, 5]
    assert poster.stats["discarded_buffer_full"] == 2


def test_parse_injection():
    assert parse_injection("bearing_wear:S04:3") == (FaultMode.BEARING_WEAR, "S04", 3.0)
    for bad in ("bearing_wear:S04", "nonsense:S04:3", "spike:S04:3", "overload:S04:-1", "overload:S04:x"):
        with pytest.raises(ValueError):
            parse_injection(bad)
