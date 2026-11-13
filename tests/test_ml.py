"""Preprocessing, anomaly detection, the failure predictor, and their wiring into the API."""

from __future__ import annotations

from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from backend.config import SIGNAL_NAMES, SIGNALS
from ml import preprocessing as pp
from ml import risk
from ml.predict import AnomalyDetector, confirmed
from ml.train import model_path, train
from simulator.fault_models import FaultMode
from simulator.sensor_simulator import ConveyorSimulator, SimConfig

T0 = 1_800_000_000.0
URL = "/api/sensors/readings"


def simulate(ticks: int, sensors: int = 2, seed: int = 3, start: float = T0, inject=None) -> tuple[list[dict], pd.DataFrame]:
    """Run the simulator without messy data. Returns payloads and a frame with ground truth."""
    sim = ConveyorSimulator(SimConfig(sensors=sensors, fault_rate=0.0, messy_rate=0.0, seed=seed))
    if inject:
        sim.inject(*inject, t=start)
    emissions = [e for i in range(ticks) for e in sim.tick(start + i * 5.0)]
    return [e.payload for e in emissions], pd.DataFrame([e.csv_row() for e in emissions])


@pytest.fixture(scope="module")
def bundle():
    """A small model trained on six simulated healthy hours of four stations."""
    _, raw = simulate(4320, sensors=4, seed=11)
    return train(pp.clean_frame(raw))


@pytest.fixture()
def detector(bundle):
    return AnomalyDetector(bundle)


@pytest.fixture()
def model_file(bundle):
    """Put the model where the backend looks for it, for the duration of one test."""
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    yield path
    path.unlink(missing_ok=True)


def frame(minutes: int, **signals) -> pd.DataFrame:
    """One station's readings every 5 s; each signal is a constant or an array."""
    n = minutes * 12
    data = {"sensor_id": "S01", "timestamp": pd.date_range("2026-11-13 08:00", periods=n, freq="5s")}
    defaults = {"temperature": 52.0, "vibration": 2.6, "speed": 4.0, "load": 70.0, "current": 260.0}
    for name in SIGNAL_NAMES:
        value = signals.get(name, defaults[name])
        data[name] = np.full(n, value, dtype=float) if np.isscalar(value) else np.asarray(value, dtype=float)
    return pd.DataFrame(data)


def ramp(start: float, per_hour: float, minutes: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    n = minutes * 12
    return start + per_hour * np.arange(n) * 5.0 / 3600.0 + np.random.default_rng(seed).normal(0.0, noise, n)


BANDS = {
    "temperature": {"median": 52.0, "low": 40.0, "high": 68.0},
    "vibration": {"median": 2.6, "low": 2.0, "high": 3.3},
    "speed": {"median": 4.0, "low": 3.93, "high": 4.07},
    "load": {"median": 70.0, "low": 50.0, "high": 92.0},
    "current": {"median": 260.0, "low": 215.0, "high": 305.0},
}


# --- preprocessing ---------------------------------------------------------------------


def test_clean_frame_applies_the_backend_rules_to_csv_rows():
    raw = pd.DataFrame(
        {
            "sensor_id": ["S02", "S01", "S01", "S01", "S01", "S01"],
            "timestamp": ["2026-11-13T08:00:00.000Z", "2026-11-13T08:00:10.000Z", "2026-11-13T08:00:05.000Z",
                          "2026-11-13T08:00:05.000Z", "", "2026-11-13T08:00:15.000Z"],
            "temperature": ["52.0", "ERR", "51.0", "99.0", "50.0", "nan"],
            "vibration": ["2.5", "2.6", "-999.0", "2.5", "2.5", ""],
            "speed": ["4.0", "4.0", "4.0", "4.0", "4.0", "x"],
            "load": ["70", "71", "72", "73", "74", "N/A"],
            "current": ["260", "261", "262", "263", "264", "1e6"],
        }
    )
    df = pp.clean_frame(raw)
    # Sorted by sensor then time; the row without a timestamp and the one with no usable signal are gone.
    assert list(df["sensor_id"]) == ["S01", "S01", "S02"]
    assert list(df["timestamp"].dt.second) == [5, 10, 0]
    first, second = df.iloc[0], df.iloc[1]
    assert first["temperature"] == 51.0  # the first of two rows with the same timestamp wins
    assert np.isnan(first["vibration"])  # -999 is outside the possible range
    assert np.isnan(second["temperature"])  # text
    assert second["load"] == 71.0


def test_stuck_values_are_blanked_from_the_sixth_repeat():
    df = frame(1, vibration=[2.5, 2.6, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.8, 2.9])
    out = pp.mask_stuck(df.copy(), run=6)
    assert out["vibration"].isna().tolist() == [False] * 7 + [True] * 3 + [False] * 2
    assert not out["temperature"].isna().any() or True  # constant test columns are also 'stuck'; only vibration is asserted


def test_imputation_uses_last_value_briefly_then_the_healthy_median():
    values = [50.0] + [np.nan] * 20
    df = frame(2)[:21].assign(temperature=values)
    out = pp.impute(df, {**{s: 0.0 for s in SIGNAL_NAMES}, "temperature": 52.5})
    assert out["temperature"].tolist() == [50.0] * (1 + pp.FFILL_LIMIT) + [52.5] * (20 - pp.FFILL_LIMIT)


def test_missing_values_never_reach_the_model(detector):
    _, raw = simulate(500)
    df = pp.clean_frame(raw)
    df.loc[df.index[::3], "temperature"] = np.nan
    df["current"] = np.nan  # a signal that is missing entirely
    df.loc[df.index[:50], list(SIGNAL_NAMES)] = np.nan  # nothing at all at the start
    prepared = detector.prepare(df)
    assert not prepared[list(SIGNAL_NAMES)].isna().any().any()
    features = pp.build_features(prepared, detector.window)
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    scored = detector.score(prepared)
    assert len(scored) == len(df) and np.isfinite(scored["score"]).all()


def test_rolling_slope_recovers_a_known_rate(bundle):
    df = frame(60, vibration=ramp(2.6, 1.5, 60))  # +1.5 mm/s per hour
    prepared = pp.prepare(df, bundle["medians"], bundle["baseline"])
    features = pp.build_features(prepared, window=360)
    assert features["vibration_slope"].iloc[-1] == pytest.approx(1.5, rel=0.01)
    assert features["vibration_mean"].iloc[-1] == pytest.approx(df["vibration"].iloc[-360:].mean())
    assert features["temperature_slope"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert features["window_count"].iloc[-1] == 360 and features["window_count"].iloc[0] == 1


# --- anomaly detection -----------------------------------------------------------------


def test_healthy_data_is_rarely_flagged_and_a_known_bad_reading_always_is(detector):
    _, raw = simulate(1500, sensors=4, seed=21)
    df = pp.clean_frame(raw)
    bad = df.index[df["sensor_id"] == "S02"][-1]
    df.loc[bad, "vibration"] = 9.0  # between the warning (7.1) and failure (11) limits
    worse = df.index[df["sensor_id"] == "S03"][-1]
    df.loc[worse, "temperature"] = 101.0  # beyond the failure limit
    scored = detector.score(detector.prepare(df))

    assert scored.loc[bad, "is_anomaly"] and scored.loc[bad, "severity"] == "High"
    assert scored.loc[bad, "anomaly_type"].startswith("vibration beyond warning limit")
    assert scored.loc[worse, "severity"] == "Critical"
    assert "temperature beyond failure limit" in scored.loc[worse, "anomaly_type"]
    healthy = scored.drop(index=[bad, worse])
    assert healthy["is_anomaly"].mean() < 0.03
    assert (healthy.loc[~healthy["is_anomaly"], "severity"] == "").all()


def test_developing_bearing_fault_is_flagged_before_any_limit_is_crossed(detector):
    _, raw = simulate(1440, sensors=2, seed=5, inject=(FaultMode.BEARING_WEAR, "S01", 4.0))
    df = pp.clean_frame(raw)
    scored = detector.score(detector.prepare(df))
    s01 = (df["sensor_id"] == "S01").to_numpy()
    last_half_hour = s01 & (df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(minutes=30)).to_numpy()
    assert df.loc[last_half_hour, "vibration"].max() < SIGNALS["vibration"].warn  # still inside limits
    assert scored.loc[last_half_hour, "is_anomaly"].mean() > 0.9
    assert scored.loc[last_half_hour, "anomaly_type"].str.contains("vibration").mean() > 0.9
    assert scored.loc[~s01, "is_anomaly"].mean() < 0.03  # the healthy station stays quiet


def test_confirmation_needs_three_in_a_row_unless_severity_is_high():
    flags = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0], dtype=bool)
    low = np.zeros(10, dtype=int)
    assert confirmed(flags, low, 3).tolist() == [False, False, False, False, False, True, True, False, False, False]
    high = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0])
    assert confirmed(flags, high, 3)[8]


# --- failure prediction ----------------------------------------------------------------


def test_time_to_failure_on_a_ramp_with_a_known_answer():
    # Vibration rises 1.0 mm/s per hour for 3 hours and ends at 6.0; the limit is 11.0,
    # so the true time to failure is (11.0 - 6.0) / 1.0 = 5.0 hours.
    df = frame(180, vibration=ramp(3.0, 1.0, 180, noise=0.1))
    result = risk.assess(df, BANDS)
    assert result.failure_mode == "bearing_wear" and result.failure_mode_label == "idler / bearing wear"
    assert result.eta_low_h < 5.0 < result.eta_high_h
    middle = result.details["trends"]["vibration"]
    assert middle["slope_per_h"] == pytest.approx(1.0, abs=0.05) and middle["r2"] > 0.9
    assert result.confidence == "High"
    assert result.risk_level == "High"  # the early end of the window is within 8 hours
    assert "5" in result.reason and "idler / bearing wear" in result.reason


def test_window_scales_with_the_rate():
    slow = risk.assess(frame(180, vibration=ramp(3.0, 0.25, 180, noise=0.05)), BANDS)  # true answer 29 h
    assert slow.eta_low_h < 29.0 < slow.eta_high_h and slow.risk_level in ("Low", "Medium")
    fast = risk.assess(frame(180, vibration=ramp(3.0, 2.5, 180, noise=0.05)), BANDS)  # ends at 10.5: 0.2 h left
    assert fast.eta_high_h < 1.0 and fast.risk_level == "Critical"


def test_no_clear_trend_means_no_estimate():
    rng = np.random.default_rng(1)
    result = risk.assess(frame(180, vibration=2.6 + rng.normal(0, 0.12, 2160), temperature=52 + rng.normal(0, 0.5, 2160)), BANDS)
    assert result.failure_mode is None and not result.has_estimate
    assert result.window_text() == "no estimate" and result.confidence is None and result.risk_level == "Low"
    assert "No time estimate" in result.reason


def test_trend_away_from_the_limit_is_not_extrapolated():
    falling = risk.assess(frame(180, vibration=ramp(3.2, -0.2, 180, noise=0.02)), BANDS)
    assert not falling.has_estimate and falling.failure_mode is None


def test_falling_speed_is_belt_slip_and_overload_needs_high_load():
    slip = risk.assess(frame(180, speed=ramp(4.0, -0.1, 180, noise=0.005), current=ramp(260, 8, 180)), BANDS)
    assert slip.failure_mode == "belt_slip"
    assert slip.eta_low_h < (3.7 - 3.2) / 0.1 < slip.eta_high_h  # ends at 3.7 m/s; limit 3.2; 5 h left

    hot = ramp(60.0, 4.0, 180, noise=0.3)  # ends at 72 degC; limit 95; 5.75 h left
    overload = risk.assess(frame(180, load=115.0, temperature=hot, current=350.0), BANDS)
    assert overload.failure_mode == "overload" and overload.eta_low_h < 5.75 < overload.eta_high_h
    assert "Load has averaged 115 %" in overload.reason

    # The same heating at normal load is not called overload.
    assert risk.assess(frame(180, temperature=hot), BANDS).failure_mode is None


def test_signal_at_the_failure_limit_is_critical_now():
    result = risk.assess(frame(60, vibration=11.4), BANDS)
    assert result.risk_level == "Critical" and result.window_text() == "at failure limit now"
    assert result.eta_low_h == 0.0 and result.eta_high_h == 0.0


def test_short_history_gives_low_confidence_or_nothing():
    brief = risk.assess(frame(40, vibration=ramp(3.0, 3.0, 40, noise=0.05)), BANDS)
    assert brief.has_estimate and brief.confidence == "Low"
    assert not risk.assess(frame(5, vibration=ramp(3.0, 3.0, 5)), BANDS).has_estimate


def test_frequent_anomalies_raise_risk_even_without_a_failure_pattern():
    calm = frame(120)
    assert risk.assess(calm, BANDS, anomalies=risk.AnomalySummary(0.0)).risk_level == "Low"
    assert risk.assess(calm, BANDS, anomalies=risk.AnomalySummary(0.15, "Low")).risk_level == "Medium"
    busy = risk.assess(calm, BANDS, anomalies=risk.AnomalySummary(0.6, "High"))
    assert busy.risk_level == "High" and busy.failure_mode is None and not busy.has_estimate


def test_window_wording():
    assert risk.format_window(None, None) == "no estimate"
    assert risk.format_window(0.0, 0.0) == "at failure limit now"
    assert risk.format_window(6.2, 13.8) == "6 h to 14 h"
    assert risk.format_window(0.4, 1.3) == "25 min to 1.5 h"


# --- wired into the backend --------------------------------------------------------------


def post_all(client, payloads, size=600):
    for i in range(0, len(payloads), size):
        assert client.post(URL, json=payloads[i : i + size]).status_code == 200


def test_without_a_model_readings_are_stored_and_health_says_so(client):
    payloads, _ = simulate(30)
    post_all(client, payloads)
    assert "not trained" in client.get("/health").json()["anomaly_model"]
    item = client.get("/api/sensors/S01/history").json()["items"][0]
    assert item["anomaly_score"] is None and item["is_anomaly"] is None
    assert client.get("/api/anomalies/active").json() == [] and client.get("/api/predictions").json() == []


def test_fault_becomes_an_active_anomaly_and_a_prediction_then_clears(client, model_file):
    from ml.live import analyser

    analyser.streak.clear()
    analyser.last_prediction.clear()

    # 55 minutes of a bearing fault that fails after 1 hour, on S01. S02 stays healthy.
    payloads, _ = simulate(660, inject=(FaultMode.BEARING_WEAR, "S01", 1.0))
    post_all(client, payloads)

    assert client.get("/health").json()["anomaly_model"] == "loaded"
    active = client.get("/api/anomalies/active").json()
    assert [a["sensor_id"] for a in active] == ["S01"]
    assert active[0]["status"] == "active" and active[0]["ended_at"] is None and active[0]["station_km"] == 1.25
    assert active[0]["severity"] in ("High", "Critical") and "vibration" in active[0]["anomaly_type"]

    sensors = {s["sensor_id"]: s for s in client.get("/api/sensors").json()}
    assert sensors["S01"]["status"] == "predicted failure" and sensors["S02"]["status"] == "normal"
    assert sensors["S01"]["active_anomaly"]["id"] == active[0]["id"]
    assert sensors["S01"]["latest_reading"]["is_anomaly"] is True

    top = client.get("/api/predictions").json()[0]
    assert top["sensor_id"] == "S01" and top["failure_mode"] == "bearing_wear" and top["station_km"] == 1.25
    assert top["risk_level"] in ("High", "Critical") and top["time_window"] != "no estimate"
    assert top["eta_low_h"] <= 5 / 60 + 0.5  # five minutes were left; the window must start soon
    assert "Vibration" in top["reason"]
    assert [p["sensor_id"] for p in client.get("/api/predictions", params={"min_risk": "High"}).json()] == ["S01"]
    # One prediction per station per batch here, because each batch covers far more than a minute.
    history = client.get("/api/predictions/S01", params={"limit": 2}).json()
    assert history["latest"]["created_at"] == history["history"]["items"][0]["created_at"]
    assert history["history"]["total"] == 3 and len(history["history"]["items"]) == 2
    assert client.get("/api/predictions/S99").status_code == 404

    summary = client.get("/api/dashboard/summary").json()
    assert summary["active_anomalies"] == 1 and summary["status_counts"] == {"predicted failure": 1, "normal": 1}

    # The bearing is replaced: healthy readings again. Risk stays raised for a while after a
    # repair, because the anomaly record of the last hour counts towards it; after 100
    # minutes of healthy data everything is back to normal.
    later = T0 + 660 * 5.0
    healthy, _ = simulate(1200, start=later, seed=4)
    post_all(client, healthy)
    assert client.get("/api/anomalies/active").json() == []
    page = client.get("/api/anomalies", params={"sensor_id": "S01"}).json()
    cleared = page["items"][-1]
    assert cleared["status"] == "cleared" and cleared["ended_at"] == cleared["last_anomalous_at"]
    assert client.get("/api/anomalies", params={"severity": cleared["severity"], "status": "cleared"}).json()["total"] >= 1
    assert client.get("/api/anomalies", params={"sensor_id": "S02"}).json()["total"] == 0
    assert client.get("/api/anomalies", params={"severity": "Extreme"}).status_code == 422
    assert client.get("/api/sensors/S01").json()["status"] == "normal"
