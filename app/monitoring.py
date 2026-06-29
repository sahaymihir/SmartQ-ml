"""Lightweight model-monitoring store: persist every prediction, attach outcomes
when known, and summarise live accuracy / error / feature drift.

Stdlib sqlite3 only — no external DB. Drift uses PSI (Population Stability Index)
against the decile reference snapshot written by train_los_v1.py.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "monitoring.db"
REFERENCE_STATS_PATH = BASE_DIR / "models" / "reference_stats.json"

# PSI bands (industry-standard): <0.1 stable, 0.1-0.25 minor shift, >0.25 drift.
PSI_WATCH = 0.1
PSI_DRIFT = 0.25


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                endpoint    TEXT    NOT NULL,
                inputs      TEXT    NOT NULL,
                predicted   REAL,
                confidence  REAL,
                latency_ms  REAL,
                actual      REAL,
                source      TEXT    NOT NULL DEFAULT 'live'
            )
            """
        )


def log_prediction(
    endpoint: str,
    inputs: dict,
    predicted: float | None,
    confidence: float | None,
    latency_ms: float | None,
    source: str = "live",
) -> int:
    """Record one prediction. Returns its row id (used later to attach an outcome)."""
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (ts, endpoint, inputs, predicted, confidence, latency_ms, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, endpoint, json.dumps(inputs), predicted, confidence, latency_ms, source),
        )
        return int(cur.lastrowid)


def record_outcome(prediction_id: int, actual: float) -> bool:
    """Attach the observed ground truth to a logged prediction."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE predictions SET actual = ? WHERE id = ?", (actual, prediction_id)
        )
        return cur.rowcount > 0


def _psi(reference_props: list[float], recent_counts: np.ndarray) -> float:
    ref = np.asarray(reference_props, dtype=float)
    recent = recent_counts / recent_counts.sum() if recent_counts.sum() else recent_counts
    # Floor to avoid div-by-zero / log(0); standard PSI epsilon.
    eps = 1e-4
    ref = np.clip(ref, eps, None)
    recent = np.clip(recent, eps, None)
    return float(np.sum((recent - ref) * np.log(recent / ref)))


def _drift(rows: list[sqlite3.Row]) -> dict[str, dict]:
    if not REFERENCE_STATS_PATH.exists():
        return {}
    reference = json.loads(REFERENCE_STATS_PATH.read_text())
    inputs = [json.loads(r["inputs"]) for r in rows]
    out: dict[str, dict] = {}
    for feature, stats in reference.items():
        values = [row[feature] for row in inputs if row.get(feature) is not None]
        if len(values) < 20:  # too few live samples to judge drift
            continue
        counts, _ = np.histogram(values, bins=np.asarray(stats["edges"]))
        psi = _psi(stats["proportions"], counts)
        status = "drift" if psi > PSI_DRIFT else "watch" if psi > PSI_WATCH else "stable"
        out[feature] = {"psi": round(psi, 4), "status": status, "n": len(values)}
    return out


def summary() -> dict:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id").fetchall()

    if not rows:
        return {"total": 0, "message": "No predictions logged yet. Run seed_monitoring.py."}

    triage = [r for r in rows if r["endpoint"] == "/predict"]
    los = [r for r in rows if r["endpoint"] == "/predict-los"]

    # Live triage accuracy (only rows with a recorded outcome).
    triage_labelled = [r for r in triage if r["actual"] is not None]
    triage_accuracy = (
        sum(1 for r in triage_labelled if int(r["predicted"]) == int(r["actual"]))
        / len(triage_labelled)
        if triage_labelled
        else None
    )

    # Live LOS MAE.
    los_labelled = [r for r in los if r["actual"] is not None]
    los_mae = (
        float(np.mean([abs(r["predicted"] - r["actual"]) for r in los_labelled]))
        if los_labelled
        else None
    )

    # Triage confidence histogram (10 bins 0..1).
    confidences = [r["confidence"] for r in triage if r["confidence"] is not None]
    conf_counts, _ = np.histogram(confidences, bins=np.linspace(0, 1, 11)) if confidences else (np.zeros(10), None)

    # Predicted triage class distribution.
    class_dist: dict[str, int] = {}
    for r in triage:
        if r["predicted"] is not None:
            key = str(int(r["predicted"]))
            class_dist[key] = class_dist.get(key, 0) + 1

    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]

    return {
        "total": len(rows),
        "by_endpoint": {"/predict": len(triage), "/predict-los": len(los)},
        "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else None,
        "triage_accuracy": round(triage_accuracy, 4) if triage_accuracy is not None else None,
        "triage_labelled_n": len(triage_labelled),
        "los_mae_hours": round(los_mae, 3) if los_mae is not None else None,
        "los_labelled_n": len(los_labelled),
        "confidence_histogram": {
            "bins": [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)],
            "counts": [int(c) for c in conf_counts],
        },
        "triage_class_distribution": dict(sorted(class_dist.items())),
        "drift": _drift(rows),
    }


if __name__ == "__main__":
    # ponytail: self-check — insert, label, summarise, assert the numbers add up.
    DB_PATH.unlink(missing_ok=True)
    init_db()
    pid = log_prediction("/predict", {"age": 40, "heart_rate": 80}, predicted=3, confidence=0.8, latency_ms=5)
    record_outcome(pid, 3)
    log_prediction("/predict-los", {"age": 40}, predicted=2.0, confidence=None, latency_ms=4)
    record_outcome(log_prediction("/predict-los", {"age": 50}, 4.0, None, 4), 5.0)
    s = summary()
    assert s["total"] == 3, s
    assert s["triage_accuracy"] == 1.0, s
    assert s["los_mae_hours"] == 1.0, s  # one labelled LOS row: |4-5| = 1.0
    DB_PATH.unlink(missing_ok=True)
    print("monitoring self-check passed:", json.dumps(s["by_endpoint"]))
