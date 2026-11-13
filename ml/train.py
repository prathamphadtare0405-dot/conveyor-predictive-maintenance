"""Train the anomaly detector on simulator backfill data.

    python -m simulator.sensor_simulator --backfill 48
    python -m ml.train

What is trained, and on what:

* The model learns **normal operation only**. Rows are used when the simulator's hidden
  label says the station was healthy and had been healthy for a full rolling window.
  On a real belt this corresponds to training on a commissioning period that the
  maintenance team signs off as fault-free. The labels are used for nothing else.
* Messy readings stay in: they go through the same cleaning and imputation as in
  production, so the model sees imputed values during training too.
* The alarm threshold is a quantile of the training scores: `IFOREST_CONTAMINATION` is
  the share of healthy readings that are allowed to score as anomalous.

The result is one file, `ml/models/anomaly_model.joblib`, holding everything needed at
inference time (medians for imputation, baseline, scaler, forest, threshold).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from backend.config import SIGNAL_NAMES, settings
from ml import preprocessing as pp

MODEL_FILE = "anomaly_model.joblib"
MAX_TRAIN_ROWS = 150_000


def model_path() -> Path:
    return settings.ml_model_dir / MODEL_FILE


def settled_healthy_mask(df: pd.DataFrame, window: int) -> np.ndarray:
    """True where the station is healthy now and was for the whole rolling window before."""
    unhealthy = (df["gt_fault_mode"] != "none").astype(float)
    recent_unhealthy = (
        unhealthy.groupby(df["sensor_id"], sort=False).rolling(window, min_periods=1).sum().droplevel(0).reindex(df.index)
    )
    return (recent_unhealthy == 0).to_numpy()


def train(df: pd.DataFrame, window: int | None = None, contamination: float | None = None, seed: int = 0) -> dict[str, Any]:
    """Fit baseline, scaler and Isolation Forest. `df` is a cleaned frame with gt columns."""
    window = window or settings.rolling_window
    contamination = contamination if contamination is not None else settings.iforest_contamination
    if "gt_fault_mode" not in df.columns:
        raise ValueError("Training data needs the simulator's gt_fault_mode column to select healthy periods.")

    healthy = (df["gt_fault_mode"] == "none").to_numpy()
    if healthy.sum() < 1000:
        raise ValueError(f"Only {int(healthy.sum())} healthy rows; generate more backfill data first.")

    medians = pp.healthy_medians(df[healthy])
    imputed = pp.impute(df, medians)
    baseline = pp.fit_baseline(imputed, healthy)
    prepared = pp.add_residuals(imputed, baseline)
    features = pp.build_features(prepared, window)

    usable = settled_healthy_mask(df, window) & (features["window_count"].to_numpy() >= window // 2)
    x_all = features.loc[usable, pp.FEATURE_NAMES]
    if len(x_all) > MAX_TRAIN_ROWS:
        x_all = x_all.sample(MAX_TRAIN_ROWS, random_state=seed)

    scaler = StandardScaler().fit(x_all.to_numpy())
    forest = IsolationForest(n_estimators=200, max_samples=512, random_state=seed, n_jobs=-1)
    forest.fit(scaler.transform(x_all.to_numpy()))
    scores = forest.score_samples(scaler.transform(x_all.to_numpy()))  # higher = more normal

    bands = {
        s: {
            "median": float(prepared.loc[usable, s].median()),
            "low": float(prepared.loc[usable, s].quantile(0.001)),
            "high": float(prepared.loc[usable, s].quantile(0.999)),
        }
        for s in SIGNAL_NAMES
    }
    resid_bands = {
        s: {
            "low": float(features.loc[usable, f"{s}_resid"].quantile(0.001)),
            "high": float(features.loc[usable, f"{s}_resid"].quantile(0.999)),
            "sd": float(features.loc[usable, f"{s}_resid"].std()),
        }
        for s in pp.BASELINE_SIGNALS
    }
    return {
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "window": window,
        "contamination": contamination,
        "feature_names": list(pp.FEATURE_NAMES),
        "medians": medians,
        "baseline": baseline,
        "scaler": scaler,
        "forest": forest,
        "threshold": float(np.quantile(scores, contamination)),
        "score_median": float(np.median(scores)),
        "score_sd": float(np.std(scores)),
        "bands": bands,
        "resid_bands": resid_bands,
        "rows_total": int(len(df)),
        "rows_healthy": int(healthy.sum()),
        "rows_trained_on": int(len(x_all)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.train", description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=settings.data_dir / "backfill.csv", help="simulator CSV (default %(default)s)")
    parser.add_argument("--out", type=Path, default=model_path(), help="where to write the model (default %(default)s)")
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(f"No training data at {args.data}.\nRun first:  python -m simulator.sensor_simulator --backfill 48", file=sys.stderr)
        return 1

    print(f"Reading {args.data} ...")
    df = pp.load_csv(args.data)
    print(f"  {len(df):,} usable rows from {df['sensor_id'].nunique()} sensors, "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")
    try:
        bundle = train(df)
    except ValueError as exc:
        print(f"Cannot train: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    print(f"  healthy rows: {bundle['rows_healthy']:,}; trained on {bundle['rows_trained_on']:,} "
          f"(healthy for a full {bundle['window']}-reading window)")
    print(f"  features: {len(bundle['feature_names'])}; trees: 200")
    print(f"  score threshold: {bundle['threshold']:.4f} (the lowest {bundle['contamination']:.2%} of healthy training scores)")
    print("  baseline (expected value from load and hour of day), residual spread on healthy data:")
    for name, band in bundle["resid_bands"].items():
        print(f"    {name:<12} sd {band['sd']:.3f}   99.8% of healthy residual means within [{band['low']:.3f}, {band['high']:.3f}]")
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
