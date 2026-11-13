"""Runs the analysis when readings are ingested.

Called by the backend after a batch has been stored. For every station that sent new
readings it:

1. scores the new readings with the anomaly detector and stores score and severity;
2. opens, extends or clears the station's anomaly;
3. every `PREDICTION_INTERVAL_S` of data time, runs the failure predictor and stores
   the result.

This runs inside the ingest request. That is fine at this scale (see docs/DESIGN.md).

Two limits, both deliberate:

* A reading that arrives more than ten minutes behind the station's newest reading is
  stored but not scored. Scoring it would mean rebuilding old rolling windows, and an
  alarm about ten minutes ago is of little use.
* The count of consecutive anomalous readings is held in memory. After a restart an
  anomaly needs the configured number of anomalous readings again before it opens.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend import crud
from backend.config import SIGNAL_NAMES, settings
from ml import preprocessing as pp
from ml import risk
from ml.predict import SEVERITIES, AnomalyDetector, get_detector

log = logging.getLogger("analysis")

LATE_SCORING_LIMIT = timedelta(minutes=10)
ROW_COLUMNS = ["id", "sensor_id", "timestamp", *SIGNAL_NAMES, "flags", "is_anomaly", "severity"]
MIN_ROWS_FOR_PREDICTION = 30


def frame_from_rows(rows: list[tuple]) -> pd.DataFrame:
    """Stored readings as a frame, with stuck signals blanked so they get imputed."""
    columns = list(zip(*rows)) if rows else [()] * len(ROW_COLUMNS)
    data: dict[str, Any] = {}
    for name, values in zip(ROW_COLUMNS, columns):
        # Built column by column: NULL signals become NaN directly, which is much faster
        # than letting pandas infer types from row objects.
        data[name] = np.array(values, dtype=float) if name in SIGNAL_NAMES else list(values)
    df = pd.DataFrame(data, columns=ROW_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    stuck = np.fromiter(("stuck:" in f for f in data["flags"]), dtype=bool, count=len(df))
    for i in np.flatnonzero(stuck):
        for flag in df.at[i, "flags"].split(","):
            if flag.startswith("stuck:"):
                df.at[i, flag.split(":", 1)[1]] = np.nan
    return df


class LiveAnalyser:
    def __init__(self) -> None:
        self.streak: dict[str, int] = defaultdict(int)
        self.last_prediction: dict[str, datetime] = {}
        self._warned_no_model = False

    def run(self, db: Session, stored: dict[str, list[datetime]]) -> dict[str, Any]:
        """Analyse the readings just stored. `stored` maps sensor id to their data timestamps."""
        summary = {"model": False, "scored": 0, "anomalous": 0, "opened": 0, "cleared": 0, "predictions": 0}
        if not stored:
            return summary
        detector = get_detector()
        if detector is None:
            if not self._warned_no_model:
                log.warning("No trained model: readings are stored but not analysed. Run `python -m ml.train`.")
                self._warned_no_model = True
            return summary
        self._warned_no_model = False
        summary["model"] = True

        newest = {s: max(ts) for s, ts in stored.items()}
        wanted = {s: [t for t in ts if t >= newest[s] - LATE_SCORING_LIMIT] for s, ts in stored.items()}
        # History needed: one rolling window before the oldest reading that will be scored.
        window_span = timedelta(seconds=detector.window * settings.sim_interval_s * 1.1)
        since = min(min(ts) for ts in wanted.values()) - window_span
        rows = crud.reading_rows(db, list(stored), since)
        if not rows:
            return summary
        df = frame_from_rows(rows)
        prepared = detector.prepare(df)

        mask = np.zeros(len(df), dtype=bool)
        for sensor_id, stamps in wanted.items():
            mask |= ((df["sensor_id"] == sensor_id) & df["timestamp"].isin(stamps)).to_numpy()
        scored = detector.score(prepared, mask)
        if scored.empty:
            return summary

        crud.save_scores(
            db,
            [
                {"id": int(df.at[i, "id"]), "anomaly_score": float(r.score), "is_anomaly": bool(r.is_anomaly), "severity": r.severity}
                for i, r in zip(scored.index, scored.itertuples())
            ],
        )
        summary["scored"] = len(scored)
        summary["anomalous"] = int(scored["is_anomaly"].sum())

        for sensor_id in stored:
            sensor = crud.get_sensor(db, sensor_id)
            idx = [i for i in scored.index if df.at[i, "sensor_id"] == sensor_id]
            if sensor is None or not idx:
                continue
            self._track_anomaly(db, sensor, df.loc[idx, "timestamp"], scored.loc[idx], summary)
            if self._prediction_due(sensor_id, newest[sensor_id]):
                if self._predict(db, sensor, detector, newest[sensor_id]):
                    summary["predictions"] += 1
        db.commit()
        return summary

    # --- anomalies ---------------------------------------------------------------------

    def _track_anomaly(self, db: Session, sensor, stamps: pd.Series, scored: pd.DataFrame, summary: dict[str, Any]) -> None:
        active = crud.active_anomaly(db, sensor.id)
        confirm = settings.anomaly_confirm_readings
        for ts, row in zip(stamps, scored.itertuples()):
            ts = ts.to_pydatetime()
            if row.is_anomaly:
                self.streak[sensor.id] += 1
                if active is not None:
                    crud.extend_anomaly(active, ts, row.anomaly_type, row.severity, float(row.score))
                elif self.streak[sensor.id] >= confirm or SEVERITIES.index(row.severity) >= SEVERITIES.index("High"):
                    active = crud.open_anomaly(db, sensor, ts, row.anomaly_type, row.severity, float(row.score))
                    summary["opened"] += 1
            else:
                self.streak[sensor.id] = 0
                if active is not None and (ts - active.last_anomalous_at).total_seconds() >= settings.anomaly_clear_after_s:
                    crud.clear_anomaly(active)
                    active = None
                    summary["cleared"] += 1

    # --- predictions -------------------------------------------------------------------

    def _prediction_due(self, sensor_id: str, newest: datetime) -> bool:
        last = self.last_prediction.get(sensor_id)
        return last is None or newest < last or (newest - last).total_seconds() >= settings.prediction_interval_s

    def _predict(self, db: Session, sensor, detector: AnomalyDetector, newest: datetime) -> bool:
        rows = crud.reading_rows(db, [sensor.id], newest - timedelta(minutes=settings.trend_lookback_min))
        rows = [r for r in rows if r[2] <= newest]
        if len(rows) < MIN_ROWS_FOR_PREDICTION:
            return False
        df = frame_from_rows(rows)
        prepared = detector.prepare(df)

        last_hour = df[df["timestamp"] >= newest - timedelta(hours=1)]
        flagged = last_hour[last_hour["is_anomaly"] == True]  # noqa: E712  (column holds True/False/None)
        scored_count = int(last_hour["is_anomaly"].notna().sum())
        severities = [s for s in flagged["severity"] if s in SEVERITIES]
        anomalies = risk.AnomalySummary(
            share_anomalous=len(flagged) / scored_count if scored_count else 0.0,
            worst_severity=max(severities, key=SEVERITIES.index) if severities else None,
            active=crud.active_anomaly(db, sensor.id) is not None,
        )
        columns = ["timestamp", *SIGNAL_NAMES, *[f"{s}_resid" for s in pp.BASELINE_SIGNALS]]
        assessment = risk.assess(prepared[columns], detector.bundle["bands"], detector.bundle["resid_bands"], anomalies)
        crud.add_prediction(db, sensor, newest, assessment)
        self.last_prediction[sensor.id] = newest
        return True


analyser = LiveAnalyser()
