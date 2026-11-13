"""Evaluate the anomaly detector and the failure predictor against the simulator's ground truth.

    python -m ml.evaluation

A fresh simulator run is generated with a different seed from the training data and a
fixed end time, so the numbers are repeatable and the model has not seen the data. The
run goes through the same cleaning, imputation and feature code as production.

Everything printed here is computed in this run. The report is also written to
docs/evaluation_results.md. These numbers describe performance on simulated data only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.config import PROJECT_ROOT, settings
from ml import preprocessing as pp
from ml import risk
from ml.predict import SEVERITIES, AnomalyDetector, ModelNotTrained, confirmed
from simulator.sensor_simulator import ConveyorSimulator, CsvSink, SimConfig, run_backfill

EVAL_SEED = 1042
EVAL_HOURS = 72.0
EVAL_END = datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc).timestamp()
PREDICT_EVERY_MIN = 10
# Random faults may leave a mode out of a run. One fault of each mode is therefore started at
# the beginning, on fixed stations, so that every mode is always part of the evaluation.
EVAL_INJECTIONS = ["bearing_wear:S02:18", "belt_slip:S05:12", "overload:S08:9"]
GRADUAL = ("bearing_wear", "belt_slip", "overload")
STAGES = ("early", "developing", "advanced", "failed")


def make_eval_data(path: Path, hours: float, seed: int) -> pd.DataFrame:
    sim = ConveyorSimulator(SimConfig(seed=seed))
    sink = CsvSink(path, append=False)
    try:
        run_backfill(sim, hours, sink, poster=None, end_s=EVAL_END, injections=EVAL_INJECTIONS)
    finally:
        sink.close()
    return pp.load_csv(path)


def pct(part: float, whole: float) -> str:
    return "n/a" if whole == 0 else f"{100.0 * part / whole:.1f} %"


def fault_episodes(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Gradual fault episodes read back from the ground-truth columns."""
    episodes = []
    for sensor, g in df.groupby("sensor_id", sort=False):
        mode = g["gt_fault_mode"].to_numpy()
        run_id = np.cumsum(np.r_[True, mode[1:] != mode[:-1]])
        for _, run in g.groupby(run_id):
            m = run["gt_fault_mode"].iloc[0]
            if m not in GRADUAL:
                continue
            ttf = pd.to_numeric(run["gt_time_to_failure_h"], errors="coerce")
            with_ttf = run[ttf.notna()]
            if with_ttf.empty:
                continue  # the data starts after this fault's failure
            failure = (with_ttf["timestamp"] + pd.to_timedelta(ttf[ttf.notna()], unit="h")).median()
            start = run["timestamp"].iloc[0]
            episodes.append(
                {
                    "sensor_id": sensor,
                    "mode": m,
                    "start": start,
                    "failure": failure,
                    "duration_h": (failure - start).total_seconds() / 3600.0,
                    "failure_in_data": bool(failure <= df["timestamp"].max()),
                    # A fault already under way when the data begins has an unknown start.
                    "start_in_data": bool(start > df["timestamp"].min() + pd.Timedelta(minutes=1)),
                }
            )
    return sorted(episodes, key=lambda e: e["start"])


# --------------------------------------------------------------------------------------
# Anomaly detector
# --------------------------------------------------------------------------------------


def evaluate_detector(df: pd.DataFrame, scored: pd.DataFrame, episodes: list[dict[str, Any]], window: int) -> dict[str, Any]:
    truth = (df["gt_fault_mode"] != "none").to_numpy()
    rank = scored["severity"].map({s: i for i, s in enumerate(SEVERITIES)}).fillna(-1).to_numpy(dtype=int)
    alarm = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("sensor_id", sort=False).indices.items():
        alarm[idx] = confirmed(scored["is_anomaly"].to_numpy()[idx], rank[idx], settings.anomaly_confirm_readings)

    tp, fp = int((alarm & truth).sum()), int((alarm & ~truth).sum())
    fn, tn = int((~alarm & truth).sum()), int((~alarm & ~truth).sum())

    # Healthy rows whose rolling window still contains fault readings (just after a repair).
    unhealthy = pd.Series(truth.astype(float), index=df.index)
    tainted = (unhealthy.groupby(df["sensor_id"], sort=False).rolling(window, min_periods=1).sum().droplevel(0).reindex(df.index) > 0).to_numpy()
    settled = ~truth & ~tainted
    fp_settled = int((alarm & settled).sum())

    by_stage = {}
    for stage in (*STAGES, "sudden"):
        rows = (df["gt_stage"] == stage).to_numpy()
        by_stage[stage] = (int((alarm & rows).sum()), int(rows.sum()))
    by_mode = {}
    for mode in (*GRADUAL, "blockage", "spike"):
        rows = (df["gt_fault_mode"] == mode).to_numpy()
        by_mode[mode] = (int((alarm & rows).sum()), int(rows.sum()))

    # Event level: how long after a fault starts is the first alarm raised?
    delays = []
    for ep in episodes:
        rows = ((df["sensor_id"] == ep["sensor_id"]) & (df["timestamp"] >= ep["start"]) & (df["timestamp"] < ep["failure"])).to_numpy()
        hit = df.loc[rows & alarm, "timestamp"]
        ep["first_alarm"] = hit.iloc[0] if len(hit) else None
        if len(hit) and ep["start_in_data"]:
            delay_h = (hit.iloc[0] - ep["start"]).total_seconds() / 3600.0
            delays.append((delay_h, delay_h / ep["duration_h"]))

    # False-alarm events: alarms on settled healthy data, merged when closer than the clear period.
    events = 0
    for _, idx in df.groupby("sensor_id", sort=False).indices.items():
        times = df["timestamp"].to_numpy()[idx][(alarm & settled)[idx]]
        if len(times):
            gaps = np.diff(times).astype("timedelta64[s]").astype(float)
            events += 1 + int((gaps > settings.anomaly_clear_after_s).sum())
    sensor_days = settled.sum() * settings.sim_interval_s / 86400.0

    rules = {}
    for label, column in (("Isolation Forest score below threshold", "forest_flag"),
                          ("A signal beyond its warning limit", "limit_flag"),
                          ("Load-compensated level outside the healthy band (drift)", "drift_flag")):
        flag = scored[column].to_numpy().astype(bool)
        rules[label] = (float((flag & truth).sum() / truth.sum()) if truth.sum() else float("nan"),
                        float((flag & settled).sum() / settled.sum()) if settled.sum() else float("nan"))
    only_forest = scored["forest_flag"].to_numpy().astype(bool) & ~scored["limit_flag"].to_numpy().astype(bool) & ~scored["drift_flag"].to_numpy().astype(bool)
    rules["Isolation Forest alone (neither other rule fired)"] = (
        float((only_forest & truth).sum() / truth.sum()) if truth.sum() else float("nan"),
        float((only_forest & settled).sum() / settled.sum()) if settled.sum() else float("nan"),
    )

    return {
        "rows": len(df),
        "fault_rows": int(truth.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "false_alarm_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "false_alarm_rate_settled": fp_settled / int(settled.sum()) if settled.sum() else float("nan"),
        "settled_rows": int(settled.sum()),
        "by_stage": by_stage,
        "by_mode": by_mode,
        "delays": delays,
        "episodes_alarmed": sum(1 for e in episodes if e["first_alarm"] is not None),
        "false_alarm_events": events,
        "false_alarm_events_per_sensor_day": events / sensor_days if sensor_days else float("nan"),
        "rules": rules,
    }


# --------------------------------------------------------------------------------------
# Failure predictor
# --------------------------------------------------------------------------------------


def run_predictor(df: pd.DataFrame, prepared: pd.DataFrame, scored: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Replay the predictor every PREDICT_EVERY_MIN minutes for every station."""
    lookback = pd.Timedelta(minutes=settings.trend_lookback_min)
    hour = pd.Timedelta(hours=1)
    cols = ["timestamp", *pp.SIGNAL_NAMES, *[f"{s}_resid" for s in pp.BASELINE_SIGNALS]]
    records = []
    for sensor, idx in df.groupby("sensor_id", sort=False).indices.items():
        sub = prepared.iloc[idx][cols].reset_index(drop=True)
        truth = df.iloc[idx].reset_index(drop=True)
        flags = scored.iloc[idx].reset_index(drop=True)
        times = sub["timestamp"].to_numpy()
        t = sub["timestamp"].iloc[0] + pd.Timedelta(minutes=30)
        end = sub["timestamp"].iloc[-1]
        while t <= end:
            b = int(np.searchsorted(times, np.datetime64(t), side="right"))
            a = int(np.searchsorted(times, np.datetime64(t - lookback), side="left"))
            h = int(np.searchsorted(times, np.datetime64(t - hour), side="left"))
            if b - a >= 2:
                recent = flags.iloc[h:b]
                anomalous = recent[recent["is_anomaly"]]
                worst = max(anomalous["severity"], key=SEVERITIES.index) if len(anomalous) else None
                summary = risk.AnomalySummary(float(recent["is_anomaly"].mean()) if len(recent) else 0.0, worst)
                result = risk.assess(sub.iloc[a:b], bundle["bands"], bundle["resid_bands"], summary)
                last = truth.iloc[b - 1]
                healthy_window = bool((truth["gt_fault_mode"].iloc[a:b] == "none").all())
                records.append(
                    {
                        "sensor_id": sensor,
                        "time": t,
                        "gt_mode": last["gt_fault_mode"],
                        "gt_stage": last["gt_stage"],
                        "gt_ttf_h": pd.to_numeric(last["gt_time_to_failure_h"], errors="coerce"),
                        "healthy_window": healthy_window,
                        "mode": result.failure_mode,
                        "risk": result.risk_level,
                        "eta_low": result.eta_low_h,
                        "eta_high": result.eta_high_h,
                        "confidence": result.confidence,
                    }
                )
            t += pd.Timedelta(minutes=PREDICT_EVERY_MIN)
    return pd.DataFrame.from_records(records)


def evaluate_predictor(preds: pd.DataFrame, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    warn = preds["risk"].isin(["High", "Critical"])

    for ep in episodes:
        during = preds[(preds["sensor_id"] == ep["sensor_id"]) & (preds["time"] >= ep["start"]) & (preds["time"] < ep["failure"])]
        warned = during[during["risk"].isin(["High", "Critical"])]
        estimated = during[during["eta_low"].notna()]
        ep["warning_lead_h"] = (ep["failure"] - warned["time"].iloc[0]).total_seconds() / 3600.0 if len(warned) else None
        first_mode = warned["mode"].iloc[0] if len(warned) else None
        ep["warning_mode"] = first_mode if isinstance(first_mode, str) else ("none named" if len(warned) else None)
        ep["first_estimate_lead_h"] = (ep["failure"] - estimated["time"].iloc[0]).total_seconds() / 3600.0 if len(estimated) else None

    # Accuracy of the time window, on predictions made while a gradual fault was developing.
    est = preds[preds["gt_mode"].isin(GRADUAL) & preds["gt_ttf_h"].notna() & preds["eta_low"].notna()].copy()
    est["mid"] = (est["eta_low"] + est["eta_high"]) / 2.0
    est["inside"] = (est["gt_ttf_h"] >= est["eta_low"]) & (est["gt_ttf_h"] <= est["eta_high"])
    est["sooner"] = est["gt_ttf_h"] < est["eta_low"]  # failure came before the window opened
    est["later"] = est["gt_ttf_h"] > est["eta_high"]
    est["abs_err"] = (est["mid"] - est["gt_ttf_h"]).abs()
    est["bucket"] = pd.cut(est["gt_ttf_h"], [0, 1, 4, 12, 1000], labels=["under 1 h", "1 to 4 h", "4 to 12 h", "over 12 h"], right=False)

    developing = preds[preds["gt_mode"].isin(GRADUAL) & preds["gt_ttf_h"].notna()]
    named = developing[developing["mode"].notna()]
    healthy = preds[preds["healthy_window"]]
    failed = preds[preds["gt_stage"] == "failed"]
    sensor_days = len(healthy) * PREDICT_EVERY_MIN / 1440.0
    false_warn = healthy[healthy["risk"].isin(["High", "Critical"])]

    return {
        "predictions": len(preds),
        "est": est,
        "developing": len(developing),
        "developing_with_estimate": int(developing["eta_low"].notna().sum()),
        "developing_by_stage": {
            s: (int(developing[developing["gt_stage"] == s]["eta_low"].notna().sum()),
                int(developing[developing["gt_stage"] == s]["risk"].isin(["High", "Critical"]).sum()),
                int((developing["gt_stage"] == s).sum()))
            for s in STAGES[:3]
        },
        "mode_named": len(named),
        "mode_correct": int((named["mode"] == named["gt_mode"]).sum()),
        "confusion": pd.crosstab(named["gt_mode"], named["mode"]) if len(named) else pd.DataFrame(),
        "healthy": len(healthy),
        "healthy_risk": healthy["risk"].value_counts().to_dict(),
        "healthy_estimates": int(healthy["eta_low"].notna().sum()),
        "false_warnings": len(false_warn),
        "false_warnings_per_sensor_day": len(false_warn) / sensor_days if sensor_days else float("nan"),
        "failed": len(failed),
        "failed_critical": int((failed["risk"] == "Critical").sum()),
        "warn_total": int(warn.sum()),
    }


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def render(df, bundle, det, pred, episodes, hours, seed) -> str:
    L: list[str] = []
    add = L.append
    add("# Evaluation results")
    add("")
    add(f"Generated by `python -m ml.evaluation` on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC.")
    add("")
    add("**These numbers describe performance on simulated data only.** The simulator and the predictor share "
        "the same failure limits, and simulated faults are cleaner than real ones. They show that the pipeline "
        "works end to end and where it is weak; they do not show how it would perform on a real conveyor.")
    add("")
    add("## Setup")
    add("")
    add(f"- Evaluation data: {hours:g} simulated hours, {df['sensor_id'].nunique()} stations, seed {seed} "
        f"(training data used seed {settings.sim_seed}), fault rate {settings.sim_fault_rate:g} per station per day, "
        f"messy rate {settings.sim_messy_rate:.0%}. {len(df):,} usable readings after cleaning.")
    add("- Same simulated conveyor as the training data (same station offsets), different days: different load, noise, faults and messy data.")
    add(f"- Besides the random faults, one fault of each gradual mode is started at the beginning of the run ({', '.join(EVAL_INJECTIONS)}), "
        "so that no mode is missing from the evaluation.")
    add(f"- Model: trained {bundle['trained_at']} on {bundle['rows_trained_on']:,} healthy readings; "
        f"rolling window {bundle['window']} readings; threshold at the lowest {bundle['contamination']:.2%} of healthy training scores.")
    add(f"- An alarm is an anomalous reading that is the {settings.anomaly_confirm_readings}rd in a row or has severity High or Critical.")
    add(f"- The predictor is replayed every {PREDICT_EVERY_MIN} minutes per station with a {settings.trend_lookback_min:g}-minute lookback.")
    add(f"- Gradual fault episodes in the data: {len(episodes)} "
        f"({sum(e['failure_in_data'] for e in episodes)} reach failure inside the data).")
    add("")

    add("## Anomaly detector")
    add("")
    add("Reading level. A reading counts as a true fault reading whenever the simulator had a fault active at "
        "that station, including the early stage, when the fault has barely changed any signal.")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Readings evaluated | {det['rows']:,} ({det['fault_rows']:,} during a fault) |")
    add(f"| Precision | {det['precision']:.3f} ({det['tp']:,} true alarms of {det['tp'] + det['fp']:,} alarms) |")
    add(f"| Recall | {det['recall']:.3f} ({det['tp']:,} of {det['fault_rows']:,} fault readings) |")
    add(f"| False-alarm rate, all healthy readings | {det['false_alarm_rate']:.4f} ({det['fp']:,} of {det['fp'] + det['tn']:,}) |")
    add(f"| False-alarm rate, healthy for a full window | {det['false_alarm_rate_settled']:.4f} (of {det['settled_rows']:,}) |")
    add(f"| False-alarm events on healthy stations | {det['false_alarm_events']} ({det['false_alarm_events_per_sensor_day']:.2f} per station per day) |")
    add("")
    add("The difference between the two false-alarm rates is the period just after a repair, when the rolling "
        "window still contains fault readings.")
    add("")
    add("Recall by degradation stage:")
    add("")
    add("| Stage | Alarmed readings | Readings | Recall |")
    add("|---|---|---|---|")
    for stage, (hit, total) in det["by_stage"].items():
        add(f"| {stage} | {hit:,} | {total:,} | {pct(hit, total)} |")
    add("")
    add("Recall by fault mode:")
    add("")
    add("| Mode | Alarmed readings | Readings | Recall |")
    add("|---|---|---|---|")
    for mode, (hit, total) in det["by_mode"].items():
        add(f"| {mode} | {hit:,} | {total:,} | {pct(hit, total)} |")
    add("")
    add("Contribution of the two rules (before the confirmation step):")
    add("")
    add("| Rule | Share of fault readings flagged | Share of settled healthy readings flagged |")
    add("|---|---|---|")
    for label, (recall, far) in det["rules"].items():
        add(f"| {label} | {recall:.3f} | {far:.4f} |")
    add("")
    add(f"Gradual faults with at least one alarm before failure: {det['episodes_alarmed']} of {len(episodes)}.")
    if det["delays"]:
        d = np.array(det["delays"])
        add(f"Time from fault start to first alarm (faults whose start is in the data, n={len(d)}): "
            f"median {np.median(d[:, 0]):.1f} h, range {d[:, 0].min():.1f} to {d[:, 0].max():.1f} h; "
            f"as a share of the fault's life: median {np.median(d[:, 1]):.0%}.")
    add("")

    add("## Failure predictor")
    add("")
    add("### How early the warning came")
    add("")
    add("A warning is the first prediction with risk High or Critical before the simulated failure.")
    add("")
    add("| Station | True mode | Fault length | Warning lead time | Mode named at warning | First time estimate |")
    add("|---|---|---|---|---|---|")
    for ep in episodes:
        lead = "no warning" if ep["warning_lead_h"] is None else f"{ep['warning_lead_h']:.1f} h before failure"
        first = "none" if ep["first_estimate_lead_h"] is None else f"{ep['first_estimate_lead_h']:.1f} h before failure"
        note = "" if ep["failure_in_data"] else " (failure after end of data)"
        length = f"{ep['duration_h']:.1f} h" if ep["start_in_data"] else f"over {ep['duration_h']:.1f} h (began before the data)"
        add(f"| {ep['sensor_id']} | {ep['mode']} | {length}{note} | {lead} | {ep['warning_mode'] or '-'} | {first} |")
    add("")
    complete = [e for e in episodes if e["failure_in_data"]]
    warned = [e for e in complete if e["warning_lead_h"] is not None]
    add(f"Of the {len(complete)} faults that reached failure inside the data, {len(warned)} had a warning beforehand.")
    if warned:
        leads = np.array([e["warning_lead_h"] for e in warned])
        share = np.array([e["warning_lead_h"] / e["duration_h"] for e in warned if e["start_in_data"]])
        add(f"Warning lead time: median {np.median(leads):.1f} h, shortest {leads.min():.1f} h, longest {leads.max():.1f} h."
            + (f" As a share of the fault's life: median {np.median(share):.0%}." if len(share) else ""))
        add(f"Failure mode named correctly at the warning: {sum(e['warning_mode'] == e['mode'] for e in warned)} of {len(warned)}.")
    add("")

    add("### How far the estimated window was from the true time")
    add("")
    est = pred["est"]
    add(f"Predictions made while a gradual fault was developing: {pred['developing']:,}. "
        f"A time window was given for {pred['developing_with_estimate']:,} of them ({pct(pred['developing_with_estimate'], pred['developing'])}); "
        "the rest said \"no estimate\".")
    add("")
    add("| Stage | Predictions | With a time window | Risk High or Critical |")
    add("|---|---|---|---|")
    for stage, (n_est, n_warn, total) in pred["developing_by_stage"].items():
        add(f"| {stage} | {total:,} | {pct(n_est, total)} | {pct(n_warn, total)} |")
    add("")
    if len(est):
        add("| True time to failure | Windows given | True time inside window | Failure came sooner than window | Later than window | Median error of window midpoint |")
        add("|---|---|---|---|---|---|")
        for bucket, g in est.groupby("bucket", observed=True):
            add(f"| {bucket} | {len(g):,} | {pct(g['inside'].sum(), len(g))} | {pct(g['sooner'].sum(), len(g))} | {pct(g['later'].sum(), len(g))} | {g['abs_err'].median():.1f} h |")
        add(f"| **all** | {len(est):,} | {pct(est['inside'].sum(), len(est))} | {pct(est['sooner'].sum(), len(est))} | {pct(est['later'].sum(), len(est))} | {est['abs_err'].median():.1f} h |")
        add("")
        add("By stated confidence:")
        add("")
        add("| Confidence | Windows given | True time inside window | Median error of window midpoint |")
        add("|---|---|---|---|")
        for conf in ("High", "Medium", "Low"):
            g = est[est["confidence"] == conf]
            if len(g):
                add(f"| {conf} | {len(g):,} | {pct(g['inside'].sum(), len(g))} | {g['abs_err'].median():.1f} h |")
    add("")
    add("### Failure mode")
    add("")
    add(f"A mode was named in {pred['mode_named']:,} of {pred['developing']:,} predictions made during a developing fault; "
        f"it was the right one in {pct(pred['mode_correct'], pred['mode_named'])} of those.")
    if len(pred["confusion"]):
        add("")
        modes = list(pred["confusion"].columns)
        add("| True mode \\ named mode | " + " | ".join(modes) + " |")
        add("|---|" + "---|" * len(modes))
        for true_mode, row in pred["confusion"].iterrows():
            add(f"| {true_mode} | " + " | ".join(str(int(row[m])) for m in modes) + " |")
    add("")
    add("### False warnings and behaviour after failure")
    add("")
    counts = pred["healthy_risk"]
    add(f"Predictions on stations with no fault anywhere in the lookback window: {pred['healthy']:,}. Risk levels: "
        + ", ".join(f"{lvl} {counts.get(lvl, 0):,}" for lvl in risk.RISK_LEVELS) + ".")
    add(f"False warnings (High or Critical on a healthy station): {pred['false_warnings']} "
        f"({pred['false_warnings_per_sensor_day']:.2f} per station per day). "
        f"Time windows given on healthy stations: {pred['healthy_estimates']}.")
    add(f"After a simulated failure and before the repair, {pred['failed_critical']} of {pred['failed']} predictions were Critical.")
    add("")
    add("## Limits of this evaluation")
    add("")
    add("- One simulated run of one seed. A different seed gives different faults and different numbers; "
        "with this few fault episodes the percentages move a lot.")
    add("- The simulator drives each failing signal to exactly the limit the predictor extrapolates to. "
        "On a real belt the failure point is not known that precisely.")
    add("- Simulated faults change one or two signals smoothly. Real faults are noisier, can stall, and can overlap.")
    add("- The baseline uses hour of day as a stand-in for ambient temperature, which the simulator makes perfectly periodic.")
    add("- Sudden events (blockage, spike) cannot be predicted by design; they are only detected.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.evaluation", description=__doc__.split("\n\n")[0])
    parser.add_argument("--hours", type=float, default=EVAL_HOURS)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "evaluation_results.md")
    args = parser.parse_args(argv)

    try:
        detector = AnomalyDetector.load()
    except ModelNotTrained as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Generating {args.hours:g} h of evaluation data (seed {args.seed}) ...")
    df = make_eval_data(settings.data_dir / "eval.csv", args.hours, args.seed)
    episodes = fault_episodes(df)
    print(f"  {len(df):,} readings, {len(episodes)} gradual fault episodes")

    print("Scoring readings ...")
    prepared = detector.prepare(df)
    scored = detector.score(prepared)
    det = evaluate_detector(df, scored, episodes, detector.window)

    print("Replaying the predictor ...")
    preds = run_predictor(df, prepared, scored, detector.bundle)
    pred = evaluate_predictor(preds, episodes)

    report = render(df, detector.bundle, det, pred, episodes, args.hours, args.seed)
    args.out.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
