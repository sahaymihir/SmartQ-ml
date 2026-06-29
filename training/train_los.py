"""Train the ED Length-of-Stay (LOS) regression model.

Predicts ``ed_los_hours`` from intake vitals + demographics + triage acuity — the
column the triage classifier discards. Baseline = predict the mean LOS; the model
has to beat that MAE. Reuses the triage pipeline's engineered-vitals features so
inference at runtime matches training exactly.

Run:  python train_los_v1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

from auto_ml_pipeline_v3 import add_application_features, safe_transform_with_unknown

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
LOS_MODEL_DIR = BASE_DIR / "models" / "los_v1" / "model"
MODEL_PATH = LOS_MODEL_DIR / "los_model_v1.pkl"
REFERENCE_STATS_PATH = BASE_DIR / "models" / "reference_stats.json"

TARGET_COL = "ed_los_hours"
# patient_id / triage_nurse_id are identifiers; disposition is decided at the END of
# the stay, so it would leak the target. triage_acuity is assigned at triage (before
# the stay) and is a legitimate, realistic predictor — keep it.
DROP_COLUMNS = ["patient_id", "triage_nurse_id", "disposition"]
# Numeric features tracked for drift monitoring (PSI reference snapshot).
DRIFT_FEATURES = ["age", "heart_rate", "spo2", "systolic_bp", "respiratory_rate", "temperature_c"]


def build_reference_stats(df: pd.DataFrame) -> dict:
    """Decile bin edges + training proportions per drift feature, for PSI at runtime."""
    stats: dict[str, dict] = {}
    for col in DRIFT_FEATURES:
        if col not in df.columns:
            continue
        edges = np.unique(np.quantile(df[col].dropna(), np.linspace(0, 1, 11)))
        if len(edges) < 3:  # near-constant feature, skip
            continue
        counts, _ = np.histogram(df[col].dropna(), bins=edges)
        proportions = counts / counts.sum()
        stats[col] = {"edges": edges.tolist(), "proportions": proportions.tolist()}
    return stats


def main() -> None:
    LOS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.dropna(subset=[TARGET_COL]).copy()
    print(f"Rows: {len(df):,}  Target: {TARGET_COL} (mean={df[TARGET_COL].mean():.2f}h)")

    # Same application-time feature engineering the triage model uses.
    df, _ = add_application_features(df)

    y = df[TARGET_COL].astype(float)
    X = df.drop(columns=[TARGET_COL])

    # Simple median/mode imputation (matches the classifier pipeline's cleaning).
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    for c in numeric_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    for c in categorical_cols:
        if X[c].isna().any():
            mode = X[c].mode(dropna=True)
            X[c] = X[c].fillna(mode.iloc[0] if not mode.empty else "missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Reference distribution snapshot from the TRAIN split only (no leakage).
    REFERENCE_STATS_PATH.write_text(json.dumps(build_reference_stats(X_train), indent=2))
    print(f"Wrote drift reference stats: {REFERENCE_STATS_PATH}")

    # Per-feature defaults (raw medians/modes) so the service can fill sparse payloads.
    numeric_defaults = {c: float(X_train[c].median()) for c in numeric_cols}
    categorical_defaults = {c: str(X_train[c].mode().iloc[0]) for c in categorical_cols}

    # Encode categoricals (fit on train, apply to test with unknown-safe transform).
    feature_encoders: dict[str, LabelEncoder] = {}
    for c in categorical_cols:
        enc = LabelEncoder().fit(X_train[c].astype(str))
        X_train[c] = safe_transform_with_unknown(enc, X_train[c])
        X_test[c] = safe_transform_with_unknown(enc, X_test[c])
        feature_encoders[c] = enc

    scaler = StandardScaler().fit(X_train[numeric_cols])
    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Baseline: predict the train-set mean LOS for everyone.
    baseline_pred = np.full(len(y_test), float(y_train.mean()))
    model_pred = model.predict(X_test)

    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    model_mae = float(mean_absolute_error(y_test, model_pred))
    improvement = (baseline_mae - model_mae) / baseline_mae * 100

    print("\n=== LOS REGRESSION RESULTS (held-out 20%) ===")
    print(f"{'':12}{'MAE (h)':>10}{'RMSE (h)':>10}{'R^2':>10}")
    print(f"{'baseline':12}{baseline_mae:>10.3f}{rmse(y_test, baseline_pred):>10.3f}{0.0:>10.3f}")
    print(f"{'xgboost':12}{model_mae:>10.3f}{rmse(y_test, model_pred):>10.3f}{r2_score(y_test, model_pred):>10.3f}")
    print(f"\nMAE improvement over baseline: {improvement:.1f}%")

    bundle = {
        "model": model,
        "selected_features": list(X.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_label_encoders": feature_encoders,
        "scaler": scaler,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "target_column": TARGET_COL,
        "baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "model_rmse": rmse(y_test, model_pred),
        "model_r2": float(r2_score(y_test, model_pred)),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved bundle: {MODEL_PATH}")

    assert model_mae < baseline_mae, "Model failed to beat the mean baseline — investigate."


if __name__ == "__main__":
    main()
