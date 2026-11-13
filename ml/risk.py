"""Failure prediction: what might fail, where, and when.

The Isolation Forest says "this is unusual". It cannot say what is failing or how long
there is left. This module answers those questions with rules and straight-line trend
fits that can be read and checked by hand.

For one station it looks at the recent history and decides:

* **What.** Each gradual failure mode has a primary signal: vibration for idler/bearing
  wear, belt speed for slip, temperature under sustained high load for overload. The
  mode whose primary signal has moved furthest towards its failure limit is reported.
* **When.** A straight line is fitted to the primary signal's recent history (with the
  effect of load and time of day removed) and extended to the failure limit. The answer
  is a window, not a moment: the early end uses the fastest plausible rate, the late end
  the slowest.
* **How sure.** From how well the line fits and how much history it rests on. If there
  is no clear trend, the answer is "no estimate".
* **Risk.** The highest of three readings of the situation: how soon the estimate says,
  how far the signal has already moved, and how often anomalies were raised recently.

"Where" is the station itself: its id and kilometre position travel with the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.config import SIGNALS, settings

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

MODE_LABELS = {
    "bearing_wear": "idler / bearing wear",
    "belt_slip": "belt slip / drive problem",
    "overload": "sustained overload",
}
PRIMARY_SIGNAL = {"bearing_wear": "vibration", "belt_slip": "speed", "overload": "temperature"}

# Rules of the trend fit. Kept together so they can be read (and argued with) in one place.
MIN_POINTS = 20  # one-minute points needed for a fit
MIN_SPAN_H = 1.0 / 3.0  # shortest history a fit may rest on
MIN_R2 = 0.5  # below this the line does not describe the data
MIN_DRIFT_SHARE = 0.08  # drift must cover this share of the way from normal to failure
HORIZON_H = 72.0  # a trend that needs longer than this is not reported
LEVEL_MINUTES = 10.0  # "current level" is the median of the last ten minutes

# Risk from the early end of the time window.
ETA_CRITICAL_H, ETA_HIGH_H, ETA_MEDIUM_H = 2.0, 8.0, 24.0
# Risk from how far the primary signal has moved from normal (0) to the failure limit (1).
PROGRESS_CRITICAL, PROGRESS_HIGH, PROGRESS_MEDIUM = 1.0, 0.75, 0.4


@dataclass
class Trend:
    signal: str
    slope_per_h: float  # in the signal's units per hour
    slope_se: float
    recent_slope_per_h: float  # slope of the last third of the window
    r2: float
    points: int
    span_h: float
    resid_sd: float
    lookback_min: float
    clear: bool = False


@dataclass
class AnomalySummary:
    """What the anomaly detector reported for this station in the last hour."""

    share_anomalous: float = 0.0  # 0..1
    worst_severity: str | None = None
    active: bool = False


@dataclass
class RiskAssessment:
    failure_mode: str | None  # bearing_wear | belt_slip | overload | None
    failure_mode_label: str
    risk_level: str
    eta_low_h: float | None
    eta_high_h: float | None
    confidence: str | None  # High | Medium | Low, or None when there is no estimate
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_estimate(self) -> bool:
        return self.eta_low_h is not None

    def window_text(self) -> str:
        return format_window(self.eta_low_h, self.eta_high_h)


def format_hours(h: float) -> str:
    if h < 1.0:
        return f"{max(5, int(round(h * 60 / 5.0)) * 5)} min"
    if h < 10.0:
        return f"{round(h * 2) / 2:g} h"
    return f"{round(h):g} h"


def format_window(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "no estimate"
    if high <= 0.0:
        return "at failure limit now"
    return f"{format_hours(low)} to {format_hours(high)}"


# --------------------------------------------------------------------------------------
# Trend fitting
# --------------------------------------------------------------------------------------


def fit_line(t_h: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float] | None:
    """Ordinary least squares. Returns (slope, slope standard error, R^2, residual sd)."""
    n = len(t_h)
    if n < 3:
        return None
    t_c = t_h - t_h.mean()
    sxx = float(np.sum(t_c**2))
    if sxx <= 0.0:
        return None
    slope = float(np.sum(t_c * (y - y.mean())) / sxx)
    resid = y - (y.mean() + slope * t_c)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_sd = float(np.sqrt(ss_res / max(1, n - 2)))
    return slope, resid_sd / np.sqrt(sxx), r2, resid_sd


def fit_trend(signal: str, t_h: np.ndarray, y: np.ndarray, lookback_min: float) -> Trend | None:
    """Fit a line to one-minute medians of the last `lookback_min` minutes.

    One-minute medians make a single spike or a few bad readings irrelevant to the fit.
    """
    if len(t_h) == 0:
        return None
    keep = t_h >= t_h[-1] - lookback_min / 60.0
    minute = np.floor((t_h[keep] - t_h[keep][0]) * 60.0).astype(int)
    binned = pd.DataFrame({"m": minute, "t": t_h[keep], "y": y[keep]}).groupby("m").median()
    tb, yb = binned["t"].to_numpy(), binned["y"].to_numpy()
    span = float(tb[-1] - tb[0]) if len(tb) else 0.0
    if len(tb) < MIN_POINTS or span < MIN_SPAN_H:
        return None
    fit = fit_line(tb, yb)
    if fit is None:
        return None
    slope, se, r2, resid_sd = fit
    third = tb >= tb[-1] - span / 3.0
    recent = fit_line(tb[third], yb[third]) if third.sum() >= 5 else None
    return Trend(signal, slope, se, recent[0] if recent else slope, r2, len(tb), span, resid_sd, lookback_min)


def trend_is_clear(trend: Trend, normal_level: float) -> bool:
    """A trend counts only if it heads for the failure limit, fits, and has moved a real distance."""
    spec = SIGNALS[trend.signal]
    rate = trend.slope_per_h * spec.direction
    drift = rate * trend.span_h
    needed = max(3.0 * trend.resid_sd, MIN_DRIFT_SHARE * abs(spec.fail - normal_level))
    return rate > 0 and trend.r2 >= MIN_R2 and drift >= needed


def best_trend(signal: str, t_h: np.ndarray, y: np.ndarray, normal_level: float, lookback_min: float) -> Trend | None:
    """Try the full lookback, then half, then a quarter; return the longest window with a clear trend.

    A fault that began inside the lookback gives a bent line over the full window and a
    straight one over a shorter window, which is why shorter windows are tried.
    """
    fallback = None
    for minutes in (lookback_min, lookback_min / 2.0, lookback_min / 4.0):
        trend = fit_trend(signal, t_h, y, minutes)
        if trend is None:
            continue
        trend.clear = trend_is_clear(trend, normal_level)
        if trend.clear:
            return trend
        fallback = fallback or trend
    return fallback


def time_window(trend: Trend, level_now: float) -> tuple[float, float, float] | None:
    """Hours until the failure limit as (early, middle, late), or None if out of reach.

    The middle value extends the fitted line. The early end uses the faster of "slope plus
    two standard errors" and the slope of the most recent third of the window, because
    wear usually accelerates. The late end uses the slope minus two standard errors. The
    window is never narrower than 40 % below to 50 % above the middle value.
    """
    spec = SIGNALS[trend.signal]
    distance = (spec.fail - level_now) * spec.direction
    if distance <= 0.0:
        return 0.0, 0.0, 0.0
    rate = trend.slope_per_h * spec.direction
    if rate <= 0.0:
        return None
    middle = distance / rate
    if middle > HORIZON_H:
        return None
    fast = max(rate + 2.0 * trend.slope_se, trend.recent_slope_per_h * spec.direction, rate)
    slow = rate - 2.0 * trend.slope_se
    early = min(distance / fast, 0.6 * middle)
    late = max(distance / slow if slow > 0 else 3.0 * middle, 1.5 * middle)
    return early, middle, min(late, 3.0 * middle)


def confidence_of(trend: Trend) -> str:
    if trend.r2 >= 0.85 and trend.span_h >= 2.0:
        return "High"
    if trend.r2 >= 0.7 and trend.span_h >= 1.0:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------------------
# Assessment of one station
# --------------------------------------------------------------------------------------


def _max_level(*levels: str) -> str:
    return max(levels, key=RISK_LEVELS.index)


def _step_down(level: str) -> str:
    return RISK_LEVELS[max(0, RISK_LEVELS.index(level) - 1)]


def assess(
    frame: pd.DataFrame,
    bands: dict[str, dict[str, float]],
    resid_bands: dict[str, dict[str, float]] | None = None,
    anomalies: AnomalySummary | None = None,
    lookback_min: float | None = None,
) -> RiskAssessment:
    """Assess one station.

    `frame` holds that station's recent readings in time order with a `timestamp` column
    and the signal columns, already imputed. If `<signal>_resid` columns are present
    (measured minus expected from load and time of day) trends are fitted on those;
    otherwise on the raw signal. `bands` gives the normal median/low/high per signal.
    """
    anomalies = anomalies or AnomalySummary()
    lookback_min = lookback_min or settings.trend_lookback_min
    resid_bands = resid_bands or {}
    if len(frame) < 2:
        return _no_fault(anomalies, "Not enough readings yet.")

    t_h = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    recent = t_h >= t_h[-1] - LEVEL_MINUTES / 60.0
    level = {s: float(np.median(frame[s].to_numpy(dtype=float)[recent])) for s in ("temperature", "vibration", "speed", "load")}

    def progress(signal: str) -> float:
        spec = SIGNALS[signal]
        normal = bands[signal]["median"]
        return float(np.clip((level[signal] - normal) / (spec.fail - normal), 0.0, 1.5))

    def trend_for(signal: str) -> Trend | None:
        column = f"{signal}_resid" if f"{signal}_resid" in frame.columns else signal
        return best_trend(signal, t_h, frame[column].to_numpy(dtype=float), bands[signal]["median"], lookback_min)

    def out_of_band(signal: str) -> bool:
        """Is the load-compensated level outside what healthy operation showed?"""
        column = f"{signal}_resid"
        spec = SIGNALS[signal]
        if column in frame.columns and signal in resid_bands:
            now = float(np.median(frame[column].to_numpy(dtype=float)[recent]))
            edge = resid_bands[signal]["high"] if spec.direction > 0 else resid_bands[signal]["low"]
            margin = 3.0 * resid_bands[signal].get("sd", 0.0)
            return (now - edge) * spec.direction > margin
        edge = bands[signal]["high"] if spec.direction > 0 else bands[signal]["low"]
        return (level[signal] - edge) * spec.direction > 0

    trends = {s: trend_for(s) for s in ("vibration", "speed", "temperature")}
    overloaded = level["load"] >= SIGNALS["load"].warn

    candidates: dict[str, float] = {}
    if (trends["vibration"] and trends["vibration"].clear) or out_of_band("vibration"):
        candidates["bearing_wear"] = progress("vibration")
    if (trends["speed"] and trends["speed"].clear) or out_of_band("speed"):
        candidates["belt_slip"] = progress("speed")
    if overloaded:
        candidates["overload"] = max(progress("temperature"), 0.05)

    details: dict[str, Any] = {
        "levels": {k: round(v, 3) for k, v in level.items()},
        "trends": {s: _trend_dict(t) for s, t in trends.items() if t is not None},
        "candidates": {k: round(v, 3) for k, v in candidates.items()},
        "anomalies_last_hour": asdict(anomalies),
    }
    if not candidates:
        result = _no_fault(anomalies, "No signal is drifting towards a failure limit.")
        result.details = details
        return result

    mode = max(candidates, key=candidates.get)
    signal = PRIMARY_SIGNAL[mode]
    spec = SIGNALS[signal]
    trend = trends[signal]
    moved = candidates[mode]

    window = time_window(trend, level[signal]) if trend is not None and trend.clear else None
    if spec.beyond_fail(level[signal]):
        window = (0.0, 0.0, 0.0)
    confidence = None
    eta_risk = "Low"
    if window is not None:
        early, _middle, late = window
        confidence = "High" if late == 0.0 else confidence_of(trend)
        eta_risk = (
            "Critical" if early <= ETA_CRITICAL_H else "High" if early <= ETA_HIGH_H else "Medium" if early <= ETA_MEDIUM_H else "Low"
        )
        if confidence == "Low" and late > 0.0:
            eta_risk = _step_down(eta_risk)

    level_risk = (
        "Critical" if moved >= PROGRESS_CRITICAL else "High" if moved >= PROGRESS_HIGH else "Medium" if moved >= PROGRESS_MEDIUM else "Low"
    )
    risk = _max_level(eta_risk, level_risk, _anomaly_risk(anomalies))

    reason = _reason(mode, signal, level, bands, trend, window, confidence, anomalies, overloaded)
    return RiskAssessment(
        failure_mode=mode,
        failure_mode_label=MODE_LABELS[mode],
        risk_level=risk,
        eta_low_h=None if window is None else round(window[0], 2),
        eta_high_h=None if window is None else round(window[2], 2),
        confidence=confidence,
        reason=reason,
        details={**details, "risk_parts": {"time": eta_risk, "level": level_risk, "anomalies": _anomaly_risk(anomalies)}},
    )


def _anomaly_risk(a: AnomalySummary) -> str:
    """Risk carried by the anomaly record alone: frequency and worst severity in the last hour."""
    severe = a.worst_severity in ("High", "Critical")
    if a.share_anomalous >= 0.5 or (a.worst_severity == "Critical" and a.share_anomalous >= 0.2):
        return "High"
    if a.share_anomalous >= 0.1 or (severe and a.share_anomalous >= 0.02):
        return "Medium"
    return "Low"


def _no_fault(anomalies: AnomalySummary, text: str) -> RiskAssessment:
    risk = _anomaly_risk(anomalies)
    if risk != "Low":
        text += (
            f" Anomalies were raised on {anomalies.share_anomalous:.0%} of readings in the last hour"
            f" (worst severity {anomalies.worst_severity}), but they do not match a gradual failure pattern."
        )
    return RiskAssessment(None, "none identified", risk, None, None, None, text + " No time estimate.")


def _trend_dict(t: Trend) -> dict[str, Any]:
    return {
        "slope_per_h": round(t.slope_per_h, 4),
        "recent_slope_per_h": round(t.recent_slope_per_h, 4),
        "r2": round(t.r2, 3),
        "points": t.points,
        "span_h": round(t.span_h, 2),
        "lookback_min": t.lookback_min,
        "clear": t.clear,
    }


def _reason(mode, signal, level, bands, trend, window, confidence, anomalies, overloaded) -> str:
    spec = SIGNALS[signal]
    name = {"vibration": "Vibration", "speed": "Belt speed", "temperature": "Temperature"}[signal]
    parts = []
    if mode == "overload":
        parts.append(f"Load has averaged {level['load']:.0f} % over the last {LEVEL_MINUTES:.0f} minutes (warning limit {SIGNALS['load'].warn:g} %).")
    parts.append(
        f"{name} is at {level[signal]:.2f} {spec.unit} (normal about {bands[signal]['median']:.2f}, failure limit {spec.fail:g})."
    )
    if window is not None and window[2] == 0.0:
        parts.append("It is at or beyond the failure limit now.")
    elif window is not None:
        verb = "rising" if spec.direction > 0 else "falling"
        parts.append(
            f"With load and time of day accounted for, it has been {verb} by {abs(trend.slope_per_h):.2f} {spec.unit} per hour "
            f"over the last {format_hours(trend.span_h)} (line fit R² {trend.r2:.2f}). "
            f"At that rate it reaches the limit in {format_window(window[0], window[2])} (confidence {confidence})."
        )
    elif trend is not None and trend.slope_per_h * spec.direction > 0:
        parts.append(f"The recent trend is not clear enough to extrapolate (line fit R² {trend.r2:.2f}), so there is no time estimate.")
    else:
        parts.append("It is not currently moving towards the limit, so there is no time estimate.")
    parts.append(f"Pattern matches {MODE_LABELS[mode]}.")
    if anomalies.share_anomalous > 0:
        parts.append(
            f"Anomalies on {anomalies.share_anomalous:.0%} of readings in the last hour"
            + (f", worst severity {anomalies.worst_severity}." if anomalies.worst_severity else ".")
        )
    return " ".join(parts)
