"""Conveyor health dashboard.

    streamlit run dashboard/app.py

Everything shown comes from the backend API; nothing is hardcoded. All times are UTC
and are the readings' own timestamps (with a time-accelerated simulator these run ahead
of the wall clock).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import SIGNALS, settings  # noqa: E402

API = os.getenv("BACKEND_URL", settings.backend_url).rstrip("/")

# Status is never shown by colour alone: each state also has its own marker shape and label.
STATUS_STYLE = {
    "normal": {"color": "#0ca30c", "symbol": "circle", "icon": "●"},
    "abnormal": {"color": "#fab219", "symbol": "diamond", "icon": "◆"},
    "predicted failure": {"color": "#d03b3b", "symbol": "triangle-up", "icon": "▲"},
    "offline": {"color": "#898781", "symbol": "x", "icon": "✕"},
}
STATUS_ORDER = ["predicted failure", "abnormal", "offline", "normal"]
LEVELS = ["Low", "Medium", "High", "Critical"]
LEVEL_ICON = {"Low": "○", "Medium": "◐", "High": "●", "Critical": "▲"}
INK, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = "#2a78d6"
SIGNAL_TITLES = {
    "temperature": "Temperature (°C)",
    "vibration": "Vibration (mm/s RMS)",
    "speed": "Belt speed (m/s)",
    "load": "Belt load (% of rated)",
    "current": "Motor current (A)",
}

st.set_page_config(page_title="Conveyor health", layout="wide")


# --------------------------------------------------------------------------------------
# API access
# --------------------------------------------------------------------------------------


class BackendDown(Exception):
    pass


def api_get(path: str, **params: Any) -> Any:
    try:
        response = requests.get(f"{API}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=10)
    except requests.RequestException as exc:
        raise BackendDown(str(exc)) from exc
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise BackendDown(f"{path} returned HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


def fmt_time(value: str | None) -> str:
    if not value:
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def level_label(level: str) -> str:
    return f"{LEVEL_ICON.get(level, '')} {level}"


# --------------------------------------------------------------------------------------
# Sidebar: refresh and filters
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh now", width="stretch"):
        st.rerun()
    auto = st.toggle("Auto refresh", value=os.getenv("DASHBOARD_AUTO_REFRESH", "1") != "0")
    interval = st.number_input("Refresh every (seconds)", 2, 120, settings.dashboard_refresh_s)
    st.divider()
    st.subheader("Filters")
    st.caption("Filters apply to the anomaly, prediction and data-quality tables and to the trend charts.")
    hours_choice = st.selectbox(
        "Period (ending at the latest reading)",
        ["Last hour", "Last 3 hours", "Last 12 hours", "Last 24 hours", "Everything"],
        index=1,
    )
    severity_filter = st.multiselect("Severity / risk level", LEVELS, default=LEVELS)
    st.caption(f"Backend: {API}")

PERIOD_HOURS = {"Last hour": 1, "Last 3 hours": 3, "Last 12 hours": 12, "Last 24 hours": 24, "Everything": None}[hours_choice]

# --------------------------------------------------------------------------------------
# Load everything, or explain what is wrong
# --------------------------------------------------------------------------------------

st.title("Conveyor belt health")

try:
    summary = api_get("/api/dashboard/summary")
    sensors = api_get("/api/sensors") or []
    active = api_get("/api/anomalies/active") or []
    predictions = api_get("/api/predictions") or []
    quality = api_get("/api/data-quality", recent=100)
except BackendDown as exc:
    st.error(
        "The backend is not reachable, so there is nothing to show.\n\n"
        f"Tried: `{API}`\n\nStart it from the project root with:\n\n"
        "```\nuvicorn backend.main:app\n```\n\n"
        f"Details: {exc}"
    )
    if auto:
        time.sleep(interval)
        st.rerun()
    st.stop()

if not sensors:
    st.info(
        "The backend is running but has not received any readings yet.\n\n"
        "Start the simulator from the project root:\n\n"
        "```\npython -m simulator.sensor_simulator --speedup 60\n```"
    )
    if auto:
        time.sleep(interval)
        st.rerun()
    st.stop()

if summary["anomaly_model"] != "loaded":
    st.warning(
        "No trained anomaly model was found. Readings are being stored, but anomalies and predictions are not "
        "being produced. Run `python -m simulator.sensor_simulator --backfill 48` and then `python -m ml.train`."
    )

latest_time = pd.Timestamp(summary["latest_reading_time"]) if summary["latest_reading_time"] else None
period_start = latest_time - timedelta(hours=PERIOD_HOURS) if latest_time is not None and PERIOD_HOURS else None

sensor_ids = [s["sensor_id"] for s in sensors]
with st.sidebar:
    sensor_filter = st.multiselect("Sensor stations", sensor_ids, default=[], placeholder="All stations")
selected = set(sensor_filter or sensor_ids)

# --- headline numbers --------------------------------------------------------------------

counts = summary["status_counts"]
tiles = st.columns(6)
tiles[0].metric("Stations", summary["sensors_total"])
for column, status in zip(tiles[1:5], STATUS_ORDER):
    column.metric(f"{STATUS_STYLE[status]['icon']} {status.capitalize()}", counts.get(status, 0))
tiles[5].metric("Active anomalies", summary["active_anomalies"])
st.caption(
    f"Latest reading: {fmt_time(summary['latest_reading_time'])} UTC · "
    f"{summary['readings_stored']:,} readings stored · conveyor length {summary['conveyor_length_km']:g} km"
)

tab_belt, tab_anomalies, tab_predictions, tab_trends, tab_quality = st.tabs(
    ["Belt status", "Anomalies", "Predictions", "Sensor trends", "Data quality"]
)

# --------------------------------------------------------------------------------------
# 1. Belt status
# --------------------------------------------------------------------------------------

with tab_belt:
    st.subheader("Sensor stations along the conveyor")
    fig = go.Figure()
    length = summary["conveyor_length_km"]
    fig.add_trace(go.Scatter(x=[0, length], y=[0, 0], mode="lines", line=dict(color=AXIS, width=6), hoverinfo="skip", showlegend=False))
    for status in STATUS_ORDER:
        group = [s for s in sensors if s["status"] == status]
        style = STATUS_STYLE[status]
        hover = []
        for s in group:
            r = s["latest_reading"] or {}
            lines = [f"<b>{s['sensor_id']}</b> at {s['station_km']:g} km: {status}"]
            for name, spec in SIGNALS.items():
                value = r.get(name)
                lines.append(f"{name}: {'no value' if value is None else f'{value:g} {spec.unit}'}")
            lines.append(f"reading time: {fmt_time(r.get('timestamp'))}")
            if s["prediction"] and s["prediction"]["failure_mode"]:
                p = s["prediction"]
                lines.append(f"prediction: {p['failure_mode_label']}, {p['time_window']} (risk {p['risk_level']})")
            hover.append("<br>".join(lines))
        # Every status is always drawn in the legend, so the key is complete even when a state is absent.
        fig.add_trace(
            go.Scatter(
                x=[s["station_km"] for s in group] or [None],
                y=[0] * len(group) or [None],
                mode="markers+text",
                name=f"{status} ({len(group)})",
                marker=dict(color=style["color"], symbol=style["symbol"], size=22, line=dict(color="#fcfcfb", width=2)),
                text=[s["sensor_id"] for s in group],
                textposition="top center",
                textfont=dict(color=INK, size=13),
                hovertext=hover,
                hoverinfo="text",
            )
        )
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        legend=dict(orientation="h", y=-0.35, x=0, font=dict(color=INK)),
        xaxis=dict(title="Position along the conveyor (km)", range=[-0.15, length + 0.15], gridcolor=GRID, zeroline=False, color=MUTED),
        yaxis=dict(visible=False, range=[-1, 1.4]),
    )
    st.plotly_chart(fig, width="stretch")

    rows = []
    for s in sensors:
        r = s["latest_reading"] or {}
        p = s["prediction"] or {}
        rows.append(
            {
                "Station": s["sensor_id"],
                "km": s["station_km"],
                "Status": f"{STATUS_STYLE[s['status']]['icon']} {s['status']}",
                "Temp °C": r.get("temperature"),
                "Vibration mm/s": r.get("vibration"),
                "Speed m/s": r.get("speed"),
                "Load %": r.get("load"),
                "Current A": r.get("current"),
                "Reading time (UTC)": fmt_time(r.get("timestamp")),
                "Last heard (s ago)": s["seconds_since_last_message"],
                "Risk": level_label(p["risk_level"]) if p else "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Offline means no message has arrived for "
        f"{settings.sensor_offline_timeout_s:g} s. A blank value means that signal was missing or unusable in the latest reading."
    )

# --------------------------------------------------------------------------------------
# 2. Anomalies
# --------------------------------------------------------------------------------------


def anomaly_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Station": a["sensor_id"],
                "km": a["station_km"],
                "Severity": level_label(a["severity"]),
                "Type (signals that drove it)": a["anomaly_type"],
                "Started (UTC)": fmt_time(a["started_at"]),
                "Last anomalous reading": fmt_time(a["last_anomalous_at"]),
                "Ended": fmt_time(a["ended_at"]) if a["ended_at"] else "still active",
                "Anomalous readings": a["reading_count"],
            }
            for a in items
        ]
    )


with tab_anomalies:
    st.subheader("Active anomalies")
    shown = [a for a in active if a["sensor_id"] in selected and a["severity"] in severity_filter]
    shown.sort(key=lambda a: (-LEVELS.index(a["severity"]), a["station_km"]))
    if shown:
        st.dataframe(anomaly_table(shown), hide_index=True, width="stretch")
    elif active:
        st.info("There are active anomalies, but none match the current filters.")
    else:
        st.success("No active anomalies.")

    st.subheader("Anomaly history")
    try:
        page = api_get("/api/anomalies", start=period_start.isoformat() if period_start is not None else None, limit=500)
    except BackendDown as exc:
        page = None
        st.error(f"Could not load the anomaly history: {exc}")
    if page:
        items = [a for a in page["items"] if a["sensor_id"] in selected and a["severity"] in severity_filter]
        if items:
            st.dataframe(anomaly_table(items), hide_index=True, width="stretch")
            st.caption(f"{len(items)} of {page['total']} anomalies in the period match the filters.")
        else:
            st.info("No anomalies in this period match the filters.")

# --------------------------------------------------------------------------------------
# 3. Predictions
# --------------------------------------------------------------------------------------

with tab_predictions:
    st.subheader("What might fail, where, and when")
    shown = [p for p in predictions if p["sensor_id"] in selected and p["risk_level"] in severity_filter]
    at_risk = [p for p in shown if p["failure_mode"] or p["risk_level"] != "Low"]
    if not predictions:
        st.info("No predictions yet. The first one appears about a minute of data after readings start.")
    elif not at_risk:
        st.success("No station shows a developing failure under the current filters.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Where": f"{p['sensor_id']} at {p['station_km']:g} km",
                        "What might fail": p["failure_mode_label"],
                        "When (estimated window)": p["time_window"],
                        "Risk": level_label(p["risk_level"]),
                        "Confidence": p["confidence"] or "-",
                        "As of (UTC)": fmt_time(p["created_at"]),
                    }
                    for p in at_risk
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Why**")
        for p in at_risk:
            st.markdown(f"- **{p['sensor_id']} ({p['station_km']:g} km), {level_label(p['risk_level'])}:** {p['reason']}")
    quiet = [p["sensor_id"] for p in shown if p not in at_risk]
    if quiet:
        st.caption("No developing failure and no time estimate for: " + ", ".join(sorted(quiet)) + ".")
    st.caption(
        "The time window comes from extending the recent trend of the failing signal to its failure limit. "
        "\"No estimate\" means there is no clear trend to extend."
    )

# --------------------------------------------------------------------------------------
# 4. Sensor trends
# --------------------------------------------------------------------------------------

with tab_trends:
    default = next((s["sensor_id"] for s in sensors if s["status"] in ("predicted failure", "abnormal") and s["sensor_id"] in selected), None)
    options = [s for s in sensor_ids if s in selected]
    sensor_id = st.selectbox("Station", options, index=options.index(default) if default in options else 0)
    try:
        history = api_get(
            f"/api/sensors/{sensor_id}/history",
            start=period_start.isoformat() if period_start is not None else None,
            limit=5000,
            order="desc",
        )
    except BackendDown as exc:
        history = None
        st.error(f"Could not load the history: {exc}")
    if history and history["items"]:
        df = pd.DataFrame(history["items"]).iloc[::-1]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        flagged = df[df["is_anomaly"] == True]  # noqa: E712
        if history["total"] > len(df):
            st.caption(f"Showing the newest {len(df):,} of {history['total']:,} readings in the period.")
        st.caption("One chart per signal, each on its own scale. Dashed lines are the warning and failure limits. "
                   "Ringed points are readings the detector flagged.")
        for name, spec in SIGNALS.items():
            chart = go.Figure()
            chart.add_trace(
                go.Scatter(x=df["timestamp"], y=df[name], mode="lines", line=dict(color=SERIES, width=2), name=name,
                           hovertemplate="%{x|%H:%M:%S}<br>%{y:.2f} " + spec.unit + "<extra></extra>")
            )
            if len(flagged):
                chart.add_trace(
                    go.Scatter(x=flagged["timestamp"], y=flagged[name], mode="markers", name="flagged as anomalous",
                               marker=dict(color="rgba(0,0,0,0)", size=8, line=dict(color="#d03b3b", width=1.5)),
                               hovertemplate="%{x|%H:%M:%S}<br>%{y:.2f} " + spec.unit + " (anomalous)<extra></extra>")
                )
            values = df[name].dropna()
            # Limits are drawn only when they are near the data; otherwise they would flatten the line.
            if len(values):
                low, high = values.min(), values.max()
                pad = max((high - low) * 0.6, abs(spec.fail - spec.warn) * 0.5)
                for limit, label in ((spec.warn, "warning limit"), (spec.fail, "failure limit")):
                    if low - pad <= limit <= high + pad:
                        chart.add_hline(y=limit, line=dict(color=MUTED, width=1, dash="dash"),
                                        annotation_text=f"{label} {limit:g}", annotation_font_color=INK, annotation_position="top left")
            chart.update_layout(
                title=dict(text=SIGNAL_TITLES[name], font=dict(size=14, color=INK)),
                height=210, margin=dict(l=10, r=10, t=36, b=10), showlegend=False,
                plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb", hovermode="x unified",
                xaxis=dict(gridcolor=GRID, color=MUTED), yaxis=dict(gridcolor=GRID, color=MUTED),
            )
            st.plotly_chart(chart, width="stretch")
        with st.expander("Readings as a table"):
            st.dataframe(df.iloc[::-1][["timestamp", *SIGNALS, "flags", "arrived_late", "is_anomaly", "severity"]], hide_index=True, width="stretch")
    elif history is not None:
        st.info("No readings for this station in the selected period.")

# --------------------------------------------------------------------------------------
# 5. Data quality
# --------------------------------------------------------------------------------------

with tab_quality:
    st.subheader("Readings that were rejected, flagged, duplicated or late")
    totals = quality["totals"]
    q = st.columns(5)
    q[0].metric("Stored", f"{totals['readings_stored']:,}")
    q[1].metric("Stored with a flag", f"{totals['flagged_readings']:,}")
    q[2].metric("Rejected (not stored)", f"{totals['rejected']:,}")
    q[3].metric("Duplicates (not stored)", f"{totals['duplicates']:,}")
    q[4].metric("Arrived late", f"{totals['late']:,}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Per station**")
        per = [s for s in quality["per_sensor"] if s["sensor_id"] in selected or s["sensor_id"] == "unknown"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Station": s["sensor_id"], "Stored": s["readings_stored"], "Flagged": s["flagged_readings"],
                        "Rejected": s["rejected"], "Duplicates": s["duplicates"], "Late": s["late"],
                    }
                    for s in per
                ]
            ),
            hide_index=True, width="stretch",
        )
    with right:
        st.markdown("**By reason**")
        if quality["by_reason"]:
            reasons = pd.Series(quality["by_reason"]).sort_values()
            bar = go.Figure(go.Bar(x=reasons.values, y=reasons.index, orientation="h", marker=dict(color=SERIES),
                                   text=reasons.values, textposition="outside", textfont=dict(color=INK),
                                   hovertemplate="%{y}: %{x}<extra></extra>"))
            bar.update_layout(height=max(180, 34 * len(reasons)), margin=dict(l=10, r=40, t=10, b=10),
                              plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
                              xaxis=dict(gridcolor=GRID, color=MUTED, title="log entries"), yaxis=dict(color=INK))
            st.plotly_chart(bar, width="stretch")
        else:
            st.success("Nothing has been rejected or flagged.")
    st.markdown("**Most recent log entries**")
    events = [e for e in quality["recent_events"] if e["sensor_id"] in selected or e["sensor_id"] == "unknown"]
    if events:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Station": e["sensor_id"], "Outcome": e["outcome"], "Reason": e["reason"], "Field": e["field"],
                        "Detail": e["detail"], "Reading time (UTC)": fmt_time(e["timestamp"]), "Payload": e["raw"],
                    }
                    for e in events
                ]
            ),
            hide_index=True, width="stretch",
        )
    else:
        st.info("No log entries for the selected stations.")

if auto:
    time.sleep(interval)
    st.rerun()
