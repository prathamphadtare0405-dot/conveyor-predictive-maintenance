"""Anomaly detection at inference time.

`AnomalyDetector` scores prepared readings and gives every anomalous one a type and a
severity. It has no database access; `ml/live.py` connects it to the backend.

A reading is anomalous when any of three things is true:

* **Forest:** the Isolation Forest scores it below the threshold learned from healthy data.
* **Limit:** a signal is beyond its warning limit.
* **Drift:** the level of a signal, with the effect of load and time of day removed and
  averaged over the rolling window, is outside the band seen in healthy operation.

Why three. A limit is a fact about the equipment, not a statistical judgement: a bearing
at 9 mm/s is a problem however the forest scores it. The drift rule exists because of a
property of Isolation Forests found during testing: a tree can only split inside the
range of its training data, so a value far outside that range gets the same score as the
most extreme healthy value. The forest therefore under-reacts to exactly the slow,
one-signal drift this project is about, and reacts best to unusual combinations of
signals that are each within range. docs/evaluation_results.md reports what each rule
contributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.config import SIGNAL_NAMES, SIGNALS, settings
from ml import preprocessing as pp
from ml.train import model_path

SEVERITIES = ["Low", "Medium", "High", "Critical"]
DRIVER_Z = 3.0  # a feature this many standard deviations from healthy counts as a driver
DRIFT_MARGIN_SD = 4.0  # how far outside the healthy residual band counts as drift, in healthy standard deviations


class ModelNotTrained(RuntimeError):
    """The model file is missing. The message says how to create it."""


def confirmed(is_anomaly: np.ndarray, severity_rank: np.ndarray, confirm: int) -> np.ndarray:
    """Which anomalous readings raise an alarm (for one sensor, in time order).

    A reading raises an alarm when it is the `confirm`-th anomalous reading in a row or
    later, or at once when its severity is High or Critical. This keeps a single odd
    reading from opening an anomaly while letting a limit violation through immediately.
    """
    streak = np.zeros(len(is_anomaly), dtype=int)
    run = 0
    for i, flag in enumerate(is_anomaly):
        run = run + 1 if flag else 0
        streak[i] = run
    return is_anomaly & ((streak >= confirm) | (severity_rank >= SEVERITIES.index("High")))


class AnomalyDetector:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.window: int = bundle["window"]
        self.threshold: float = bundle["threshold"]
        self.score_sd: float = max(bundle["score_sd"], 1e-9)
        self.feature_names: list[str] = bundle["feature_names"]
        # Live scoring handles a few rows at a time; starting worker threads for that costs
        # more than it saves.
        bundle["forest"].n_jobs = 1

    @classmethod
    def load(cls, path: Path | None = None) -> "AnomalyDetector":
        path = path or model_path()
        if not path.exists():
            raise ModelNotTrained(
                f"No trained model at {path}. Run:  python -m simulator.sensor_simulator --backfill 48  then  python -m ml.train"
            )
        return cls(joblib.load(path))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute and add residuals. `df` must be sorted by sensor and time."""
        return pp.prepare(df, self.bundle["medians"], self.bundle["baseline"])

    def score(self, prepared: pd.DataFrame, rows: np.ndarray | None = None) -> pd.DataFrame:
        """Score readings. `rows` is an optional boolean mask of the rows to return.

        Features are always built from the whole frame (the rolling window needs the
        history), but only the requested rows are scored.

        Returned columns: score, forest_flag, limit_flag, drift_flag, is_anomaly, severity,
        anomaly_type, warming_up.

        Severity: Low = forest only, just below the threshold. Medium = forest well below
        the threshold, or drift. High = forest far below the threshold, or a signal beyond
        its warning limit. Critical = a signal at or beyond its failure limit.
        """
        features = pp.build_features(prepared, self.window)
        if rows is None:
            rows = np.ones(len(prepared), dtype=bool)
        feats = features.loc[rows]
        out = pd.DataFrame(index=feats.index)
        if feats.empty:
            return out.assign(
                score=[], forest_flag=[], limit_flag=[], drift_flag=[], is_anomaly=[], severity=[], anomaly_type=[], warming_up=[]
            )

        z = self.bundle["scaler"].transform(feats[self.feature_names].to_numpy())
        score = self.bundle["forest"].score_samples(z)
        # Until the window is half full, spread and slope are not meaningful: limits only.
        warming = feats["window_count"].to_numpy() < self.window // 2
        forest_flag = (score < self.threshold) & ~warming
        depth = np.where(forest_flag, (self.threshold - score) / self.score_sd, 0.0)

        raw = prepared.loc[feats.index, list(SIGNAL_NAMES)]
        beyond_warn = np.column_stack([_beyond(raw[s].to_numpy(), SIGNALS[s].warn, SIGNALS[s].direction) for s in SIGNAL_NAMES])
        beyond_fail = np.column_stack([_beyond(raw[s].to_numpy(), SIGNALS[s].fail, SIGNALS[s].direction, True) for s in SIGNAL_NAMES])

        # Drift: the load-compensated level (rolling mean residual) has left the band seen
        # in healthy operation by a clear margin. +1 = above the band, -1 = below, 0 = inside.
        drift = np.zeros((len(feats), len(pp.BASELINE_SIGNALS)), dtype=int)
        for j, name in enumerate(pp.BASELINE_SIGNALS):
            band = self.bundle["resid_bands"][name]
            level = feats[f"{name}_resid"].to_numpy()
            margin = DRIFT_MARGIN_SD * band["sd"]
            drift[:, j] = np.where(level > band["high"] + margin, 1, np.where(level < band["low"] - margin, -1, 0))
        drift[warming] = 0
        drift_flag = (drift != 0).any(axis=1)

        limit_flag = beyond_warn.any(axis=1)
        is_anomaly = forest_flag | limit_flag | drift_flag

        severity = np.full(len(feats), "Low", dtype=object)
        severity[(depth >= 1.0) | drift_flag] = "Medium"
        severity[(depth >= 3.0) | limit_flag] = "High"
        severity[beyond_fail.any(axis=1)] = "Critical"

        types = np.full(len(feats), "", dtype=object)
        for i in np.flatnonzero(is_anomaly):
            types[i] = self._describe(z[i], beyond_warn[i], beyond_fail[i], drift[i])

        out["score"] = score
        out["forest_flag"] = forest_flag
        out["limit_flag"] = limit_flag
        out["drift_flag"] = drift_flag
        out["is_anomaly"] = is_anomaly
        out["severity"] = np.where(is_anomaly, severity, "")
        out["anomaly_type"] = types
        out["warming_up"] = warming
        return out

    def _describe(self, z_row: np.ndarray, warn_row: np.ndarray, fail_row: np.ndarray, drift_row: np.ndarray) -> str:
        """Name the signals that drove the anomaly, strongest first, at most three."""
        parts: list[tuple[float, str]] = []
        for j, name in enumerate(SIGNAL_NAMES):
            if fail_row[j]:
                parts.append((1e9, f"{name} beyond failure limit"))
            elif warn_row[j]:
                parts.append((1e6, f"{name} beyond warning limit"))
        limited = {p[1].split(" ")[0] for p in parts}
        for j, name in enumerate(pp.BASELINE_SIGNALS):
            if drift_row[j] and name not in limited:
                parts.append((1e3, f"{name} drifting {'up' if drift_row[j] > 0 else 'down'}"))
                limited.add(name)
        best: dict[str, tuple[float, str]] = {}
        for value, feature in zip(z_row, self.feature_names):
            name = pp.signal_of(feature)
            if name in limited or abs(value) < DRIVER_Z:
                continue
            if name not in best or abs(value) > best[name][0]:
                if feature.endswith("_slope"):
                    word = "rising" if value > 0 else "falling"
                elif feature.endswith("_std"):
                    word = "unstable" if value > 0 else "unusually steady"
                else:
                    word = "high" if value > 0 else "low"
                best[name] = (abs(value), f"{name} {word}")
        parts.extend(best.values())
        parts.sort(key=lambda p: -p[0])
        return ", ".join(text for _, text in parts[:3]) or "unusual combination of signals"


def _beyond(values: np.ndarray, limit: float, direction: int, inclusive: bool = False) -> np.ndarray:
    delta = (values - limit) * direction
    return delta >= 0 if inclusive else delta > 0


_cached: tuple[float, AnomalyDetector] | None = None


def get_detector() -> AnomalyDetector | None:
    """The detector for the current model file, or None if no model has been trained.

    The file's modification time is checked on each call, so a model trained while the
    backend is running is picked up without a restart.
    """
    global _cached
    path = model_path()
    if not path.exists():
        _cached = None
        return None
    mtime = path.stat().st_mtime
    if _cached is None or _cached[0] != mtime:
        _cached = (mtime, AnomalyDetector.load(path))
    return _cached[1]
