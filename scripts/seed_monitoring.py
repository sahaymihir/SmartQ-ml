"""Populate the monitoring store with an honest shadow backtest.

Replays held-out rows through the LIVE triage + LOS models and records each
prediction together with its true label, so the dashboard shows real accuracy,
error, and feature-drift numbers from day one. Rows are tagged source='backtest'
so they are never mistaken for production traffic.

Run:  python seed_monitoring.py [n_rows]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import main
import monitoring

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"


def _clean(row: dict, model) -> dict:
    """Keep only fields the request model accepts, dropping NaNs and negative
    sentinels (e.g. pain_score = -1 meaning 'not assessed')."""
    out = {}
    for k in model.model_fields:
        if k not in row or pd.isna(row[k]):
            continue
        v = row[k]
        if isinstance(v, (int, float)) and v < 0:
            continue
        out[k] = v
    return out


def run(n_rows: int = 1500) -> None:
    monitoring.DB_PATH.unlink(missing_ok=True)
    monitoring.init_db()
    triage = main.load_artifacts()
    los = main.load_los_artifacts()

    df = pd.read_csv(DATA_PATH, low_memory=False).dropna(subset=["ed_los_hours"]).reset_index(drop=True)
    _, test_idx = train_test_split(range(len(df)), test_size=0.20, random_state=42)
    sample = df.iloc[list(test_idx)]
    if len(sample) > n_rows:
        sample = sample.sample(n_rows, random_state=7)
    print(f"Replaying {len(sample)} held-out rows through live models...")

    for _, raw in sample.iterrows():
        row = raw.to_dict()

        # --- Triage classifier ---
        tri_payload = main.PredictionRequest(**_clean(row, main.PredictionRequest))
        t0 = time.perf_counter()
        tri_res = main.run_inference(main.build_feature_frame(tri_payload, triage), triage)
        pid = monitoring.log_prediction(
            "/predict", tri_payload.model_dump(exclude_none=True),
            tri_res.priority_class, tri_res.confidence,
            round((time.perf_counter() - t0) * 1000, 2), source="backtest",
        )
        monitoring.record_outcome(pid, int(row["triage_acuity"]))

        # --- LOS regressor ---
        los_payload = main.LosRequest(**_clean(row, main.LosRequest))
        t0 = time.perf_counter()
        los_res = main.run_los_inference(
            main.frame_from_provided(los_payload.model_dump(exclude_none=True), los), los
        )
        pid = monitoring.log_prediction(
            "/predict-los", los_payload.model_dump(exclude_none=True),
            los_res.predicted_hours, None,
            round((time.perf_counter() - t0) * 1000, 2), source="backtest",
        )
        monitoring.record_outcome(pid, float(row["ed_los_hours"]))

    s = monitoring.summary()
    print(f"\nSeeded {s['total']} predictions.")
    print(f"  Live triage accuracy : {s['triage_accuracy']}  (n={s['triage_labelled_n']})")
    print(f"  Live LOS MAE (hours) : {s['los_mae_hours']}  (n={s['los_labelled_n']})")
    print(f"  Avg latency (ms)     : {s['avg_latency_ms']}")
    print(f"  Drift features       : {list(s['drift'])}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
