"""Cleaning, imputation and feature building. Used by training, evaluation and the live backend.

One code path builds the features in all three places, so the model always sees data
prepared the same way it was trained on.

Steps:

1. **Clean** (CSV only; the backend has already done this for stored readings): apply
   the backend's rules to raw rows. Unusable values become NaN, duplicates are dropped
   (first wins), rows are sorted by sensor and time, stuck values are blanked.
2. **Impute**, in one documented way: a missing value takes the sensor's last valid
   value if that is at most `FFILL_LIMIT` readings old (1 minute at 5 s sampling);
   otherwise it takes the median of healthy training data for that signal. A NaN can
   therefore never reach the model.
3. **Remove what operating conditions explain.** Temperature, vibration, speed and
   current all move with belt load, and temperature also with time of day. A linear
   baseline fitted on healthy data predicts each of them from load and hour of day; the
   *residual* (measured minus expected) is what a fault changes.
4. **Rolling features** over the last `window` readings of each sensor: mean, standard
   deviation and slope of every signal, plus the mean residual.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.config import SIGNAL_NAMES, SIGNALS, settings

FFILL_LIMIT = 12
LOAD_SMOOTH_READINGS = 120  # 10 minutes: temperature lags load by about this much
BASELINE_SIGNALS: tuple[str, ...] = ("temperature", "vibration", "speed", "current")

FEATURE_NAMES: list[str] = (
    [s for s in SIGNAL_NAMES]
    + [f"{s}_mean" for s in SIGNAL_NAMES]
    + [f"{s}_std" for s in SIGNAL_NAMES]
    + [f"{s}_slope" for s in SIGNAL_NAMES]
    + [f"{s}_resid" for s in BASELINE_SIGNALS]
)


def signal_of(feature: str) -> str:
    return feature.split("_")[0]


# --------------------------------------------------------------------------------------
# 1. Cleaning raw CSV rows
# --------------------------------------------------------------------------------------


def load_csv(path) -> pd.DataFrame:
    """Read a simulator CSV as text and clean it with the same rules the backend applies."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    return clean_frame(raw)


def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="ISO8601")
    df["timestamp"] = stamps.dt.tz_localize(None)
    df = df[df["timestamp"].notna() & (df["sensor_id"].astype(str).str.strip() != "")]

    for name in SIGNAL_NAMES:
        spec = SIGNALS[name]
        values = pd.to_numeric(df[name], errors="coerce").astype(float)
        values = values.where(np.isfinite(values))
        df[name] = values.where((values >= spec.hard_min) & (values <= spec.hard_max))
    df = df[df[list(SIGNAL_NAMES)].notna().any(axis=1)]

    df = df.drop_duplicates(["sensor_id", "timestamp"], keep="first")
    df = df.sort_values(["sensor_id", "timestamp"], kind="stable").reset_index(drop=True)
    return mask_stuck(df)


def mask_stuck(df: pd.DataFrame, run: int | None = None) -> pd.DataFrame:
    """Blank a signal from its `run`-th identical value in a row onwards (a stuck sensor)."""
    run = run or settings.stuck_repeat_count
    if run <= 1 or df.empty:
        return df
    for name in SIGNAL_NAMES:
        changed = df[name] != df.groupby("sensor_id", sort=False)[name].shift()
        position = df.groupby(changed.cumsum()).cumcount()
        df.loc[(position >= run - 1) & df[name].notna(), name] = np.nan
    return df


# --------------------------------------------------------------------------------------
# 2-3. Imputation and the load / time-of-day baseline
# --------------------------------------------------------------------------------------


def healthy_medians(df: pd.DataFrame) -> dict[str, float]:
    return {s: float(df[s].median()) for s in SIGNAL_NAMES}


def impute(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Last valid value for up to FFILL_LIMIT readings, then the healthy median."""
    out = df.copy()
    grouped = out.groupby("sensor_id", sort=False)
    for name in SIGNAL_NAMES:
        out[name] = grouped[name].ffill(limit=FFILL_LIMIT).fillna(medians[name])
    return out


def _baseline_inputs(df: pd.DataFrame) -> np.ndarray:
    """Regressors for the baseline: smoothed load, load, and hour of day as sine and cosine."""
    smooth = (
        df.groupby("sensor_id", sort=False)["load"].rolling(LOAD_SMOOTH_READINGS, min_periods=1).mean().droplevel(0)
    ).reindex(df.index)
    hour = (
        df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0 + df["timestamp"].dt.second / 3600.0
    ).to_numpy(dtype=float)
    angle = 2.0 * np.pi * hour / 24.0
    return np.column_stack([smooth.to_numpy(dtype=float), df["load"].to_numpy(dtype=float), np.sin(angle), np.cos(angle)])


def fit_baseline(df: pd.DataFrame, healthy: np.ndarray) -> dict[str, Any]:
    """Least-squares fit of each signal on load and hour of day, using healthy rows only.

    Each sensor also gets its own offset (its mean residual), because no two stations
    read exactly alike.
    """
    x = np.column_stack([_baseline_inputs(df), np.ones(len(df))])
    baseline: dict[str, Any] = {"coef": {}, "offsets": {}}
    for name in BASELINE_SIGNALS:
        y = df[name].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(x[healthy], y[healthy], rcond=None)
        baseline["coef"][name] = coef.tolist()
        resid = pd.Series(y - x @ coef, index=df.index)[healthy]
        baseline["offsets"][name] = resid.groupby(df.loc[healthy, "sensor_id"]).mean().to_dict()
    return baseline


def add_residuals(df: pd.DataFrame, baseline: dict[str, Any]) -> pd.DataFrame:
    """Add `<signal>_resid` = measured minus what load and hour of day explain."""
    out = df.copy()
    x = np.column_stack([_baseline_inputs(out), np.ones(len(out))])
    for name in BASELINE_SIGNALS:
        offsets = out["sensor_id"].map(baseline["offsets"][name]).fillna(0.0).to_numpy(dtype=float)
        expected = x @ np.asarray(baseline["coef"][name]) + offsets
        out[f"{name}_expected"] = expected
        out[f"{name}_resid"] = out[name].to_numpy(dtype=float) - expected
    return out


def prepare(df: pd.DataFrame, medians: dict[str, float], baseline: dict[str, Any]) -> pd.DataFrame:
    """Impute and add residuals. Input must be sorted by sensor and time."""
    out = impute(df.reset_index(drop=True), medians)
    return add_residuals(out, baseline)


# --------------------------------------------------------------------------------------
# 4. Rolling features
# --------------------------------------------------------------------------------------


def build_features(prepared: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Rolling mean, standard deviation and slope per signal over each sensor's last `window` readings.

    Returns a frame with the columns in FEATURE_NAMES plus `window_count` (how many
    readings the window actually held), on the same index as `prepared`. Slopes are in
    signal units per hour and use real timestamps, so gaps do not distort them.
    """
    window = window or settings.rolling_window
    df = prepared
    work = pd.DataFrame(index=df.index)
    work["sensor_id"] = df["sensor_id"]
    hours = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    work["x"] = hours
    work["xx"] = hours * hours
    for name in SIGNAL_NAMES:
        work[name] = df[name].to_numpy(dtype=float)
        work[f"{name}_xy"] = work[name] * hours
    for name in BASELINE_SIGNALS:
        work[f"{name}_resid"] = df[f"{name}_resid"].to_numpy(dtype=float)

    rolling = work.groupby("sensor_id", sort=False).rolling(window, min_periods=2)
    mean_cols = [c for c in work.columns if c != "sensor_id"]
    means = rolling[mean_cols].mean().droplevel(0).reindex(df.index)
    stds = rolling[list(SIGNAL_NAMES)].std().droplevel(0).reindex(df.index)
    counts = rolling["x"].count().droplevel(0).reindex(df.index)

    features = pd.DataFrame(index=df.index)
    x_var = means["xx"] - means["x"] ** 2
    for name in SIGNAL_NAMES:
        features[name] = work[name]
        features[f"{name}_mean"] = means[name]
        features[f"{name}_std"] = stds[name]
        slope = (means[f"{name}_xy"] - means["x"] * means[name]) / x_var.where(x_var > 1e-12)
        features[f"{name}_slope"] = slope
    for name in BASELINE_SIGNALS:
        features[f"{name}_resid"] = means[f"{name}_resid"]

    # The first reading of a sensor has no window yet: use the value itself and no spread or slope.
    for name in SIGNAL_NAMES:
        features[f"{name}_mean"] = features[f"{name}_mean"].fillna(features[name])
    for name in BASELINE_SIGNALS:
        features[f"{name}_resid"] = features[f"{name}_resid"].fillna(work[f"{name}_resid"])
    features = features[FEATURE_NAMES].fillna(0.0)
    features["window_count"] = counts.fillna(1).astype(int)
    return features
