from __future__ import annotations

import warnings
from pathlib import Path
import shutil
import subprocess

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODELS_DIR = BASE_DIR / "models"
TRIAGE_MODEL_DIR = MODELS_DIR / "triage_v3" / "model"
MODEL_PATH = TRIAGE_MODEL_DIR / "triage_model_v3.pkl"
FEATURES_PATH = TRIAGE_MODEL_DIR / "feature_cols_v3.pkl"
SCALER_PATH = TRIAGE_MODEL_DIR / "scaler_v3.pkl"

TARGET_COL = "triage_acuity"
DROP_COLUMNS = ["triage_nurse_id", "patient_id", "disposition", "ed_los_hours"]
OLD_BASELINE_ACCURACY = 0.8499
DEFAULT_TUNING_SAMPLE_SIZE = 10000
CPU_XGB_TUNING_SAMPLE_SIZE = 3000


def print_header(title: str) -> None:
    line = "=" * 96
    print(f"\n{line}\n{title}\n{line}")


def print_subheader(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_assumption(message: str) -> None:
    print(f"ASSUMPTION: {message}")


def detect_gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def format_counts(values: np.ndarray, encoder: LabelEncoder) -> str:
    counts = pd.Series(values).value_counts().sort_index()
    labels = encoder.inverse_transform(counts.index.to_numpy(dtype=int))
    return ", ".join(f"{label}={int(count)}" for label, count in zip(labels, counts.tolist()))


def safe_transform_with_unknown(
    encoder: LabelEncoder, series: pd.Series, unknown_token: str = "__unknown__"
) -> np.ndarray:
    values = series.astype(str).copy()
    unseen_mask = ~values.isin(encoder.classes_)
    if unseen_mask.any():
        if unknown_token not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, unknown_token)
        values.loc[unseen_mask] = unknown_token
    return encoder.transform(values)


def compute_weighted_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))


def add_application_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    engineered = df.copy()
    created_features = [
        "shock_index",
        "spo2_resp_interaction",
        "high_fever_flag",
        "low_bp_flag",
        "tachycardia_flag",
        "hypoxia_flag",
        "elderly_flag",
        "multi_risk_flag",
    ]

    if "shock_index" in engineered.columns:
        print_assumption(
            "The existing `shock_index` column is recomputed from arrival vitals so it matches "
            "the application-time feature definition exactly."
        )

    systolic_safe = engineered["systolic_bp"].replace(0, np.nan)
    engineered["shock_index"] = (engineered["heart_rate"] / systolic_safe).replace(
        [np.inf, -np.inf], np.nan
    )
    engineered["spo2_resp_interaction"] = engineered["spo2"] * engineered["respiratory_rate"]
    engineered["high_fever_flag"] = (engineered["temperature_c"] > 38.5).astype(int)
    engineered["low_bp_flag"] = (engineered["systolic_bp"] < 90).astype(int)
    engineered["tachycardia_flag"] = (engineered["heart_rate"] > 100).astype(int)
    engineered["hypoxia_flag"] = (engineered["spo2"] < 94).astype(int)
    engineered["elderly_flag"] = (engineered["age"] > 65).astype(int)
    engineered["multi_risk_flag"] = engineered[
        ["high_fever_flag", "low_bp_flag", "tachycardia_flag", "hypoxia_flag", "elderly_flag"]
    ].sum(axis=1)

    if engineered["shock_index"].isna().any():
        engineered["shock_index"] = engineered["shock_index"].fillna(engineered["shock_index"].median())

    return engineered, created_features


def fit_random_forest_selector(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    X_encoded = X.copy()

    for col in categorical_cols:
        codes, _ = pd.factorize(X_encoded[col].astype(str), sort=True)
        X_encoded[col] = codes.astype(float)

    selector = VarianceThreshold(threshold=0.01)
    X_reduced = selector.fit_transform(X_encoded)
    reduced_cols = X.columns[selector.get_support()].tolist()

    print(f"Features before VarianceThreshold: {X.shape[1]}")
    print(f"Features after VarianceThreshold : {len(reduced_cols)}")

    print_assumption(
        "RandomForest feature ranking uses 100 trees, random_state=42, n_jobs=-1 because the "
        "step specifies importances but not the forest hyperparameters."
    )
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    selector_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    selector_model.fit(X_reduced, y_encoded)

    importances = pd.DataFrame(
        {
            "feature": reduced_cols,
            "importance": selector_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importances.reset_index(drop=True)


def preprocess_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    dict[str, LabelEncoder],
    StandardScaler,
    LabelEncoder,
]:
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    categorical_cols = [col for col in X_train_raw.columns if not pd.api.types.is_numeric_dtype(X_train_raw[col])]
    numeric_cols = [col for col in X_train_raw.columns if col not in categorical_cols]

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    feature_encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        encoder.fit(X_train[col].astype(str))
        X_train[col] = safe_transform_with_unknown(encoder, X_train[col])
        X_test[col] = safe_transform_with_unknown(encoder, X_test[col])
        feature_encoders[col] = encoder

    scaler = StandardScaler()
    if numeric_cols:
        scaler.fit(X_train[numeric_cols])
        X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    else:
        print_assumption("No numeric features remained after selection, so StandardScaler was fit but unused.")

    target_encoder = LabelEncoder()
    target_encoder.fit(y_train_raw)
    y_train = target_encoder.transform(y_train_raw)
    y_test = target_encoder.transform(y_test_raw)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_cols,
        numeric_cols,
        feature_encoders,
        scaler,
        target_encoder,
    )


def maybe_apply_smote(X_train: pd.DataFrame, y_train: np.ndarray, target_encoder: LabelEncoder):
    class_counts = pd.Series(y_train).value_counts().sort_index()
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Training class counts before SMOTE: {format_counts(y_train, target_encoder)}")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.4f}")

    if imbalance_ratio <= 2.0:
        print("SMOTE not applied because no class is underrepresented by more than 2x.")
        return X_train, y_train, False

    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        print_assumption(
            "SMOTE was skipped because at least one class has fewer than 2 samples in the "
            "training split."
        )
        return X_train, y_train, False

    k_neighbors = min(5, min_class_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied with k_neighbors={k_neighbors}")
    print(f"Training class counts after SMOTE : {format_counts(y_resampled, target_encoder)}")
    return pd.DataFrame(X_resampled, columns=X_train.columns), y_resampled, True


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    roc_auc = compute_weighted_roc_auc(y_test, probabilities)
    return accuracy, roc_auc, predictions, probabilities


def build_lightgbm(n_classes: int, use_gpu: bool) -> LGBMClassifier:
    del n_classes
    params = dict(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
    )
    if use_gpu:
        params["device"] = "gpu"
    return LGBMClassifier(**params)


def build_xgboost(n_classes: int, use_gpu: bool) -> XGBClassifier:
    params = dict(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        tree_method="hist",
        max_bin=256,
    )
    if use_gpu:
        params["device"] = "cuda"
    return XGBClassifier(**params)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRIAGE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    gpu_available = detect_gpu_available()

    print_header("STEP 1 - LOAD & CLEAN")
    print(f"Loading dataset: {DATA_PATH}")
    print(f"GPU requested: yes")
    print(f"GPU available: {'yes' if gpu_available else 'no'}")
    if not gpu_available:
        print_assumption(
            "A working NVIDIA driver/GPU is not available in this environment, so the pipeline "
            "will continue on CPU with faster hist-based tree settings."
        )
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Original shape: {df.shape}")
    print(f"Dropping leakage/post-triage columns: {DROP_COLUMNS}")
    df = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    if df[TARGET_COL].isna().any():
        dropped = int(df[TARGET_COL].isna().sum())
        df = df.dropna(subset=[TARGET_COL]).copy()
        print_assumption(
            f"Dropped {dropped} rows with missing target values because they cannot be used for "
            "supervised training."
        )

    feature_cols = [col for col in df.columns if col != TARGET_COL]
    numeric_cols = [
        col
        for col in feature_cols
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
    ]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "missing"
            df[col] = df[col].fillna(fill_value)

    duplicates_removed = int(df.duplicated().sum())
    if duplicates_removed:
        df = df.drop_duplicates().copy()

    print(f"Cleaned shape: {df.shape}")
    print(f"Duplicate rows removed: {duplicates_removed}")

    print_header("STEP 2 - FEATURE ENGINEERING (APPLICATION-AWARE)")
    df, created_features = add_application_features(df)
    print(f"New feature count: {len(created_features)}")
    print(f"Created features: {created_features}")
    print(f"Shape after feature engineering: {df.shape}")

    print_header("STEP 3 - FEATURE SELECTION")
    X_full = df.drop(columns=[TARGET_COL]).copy()
    y_full = df[TARGET_COL].copy()
    importance_df = fit_random_forest_selector(X_full, y_full)
    selected_count = min(40, len(importance_df))
    selected_features = importance_df.head(selected_count)["feature"].tolist()
    print(f"Selected top features: {selected_count}")
    for row in importance_df.head(selected_count).itertuples(index=False):
        print(f"{row.feature}: {row.importance:.6f}")

    print_header("STEP 4 - PREPROCESS")
    print_assumption(
        "Label encoders and StandardScaler are fit on the training split only, then applied to "
        "the held-out test split to avoid preprocessing leakage."
    )
    X_selected = df[selected_features].copy()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        selected_categorical_cols,
        selected_numeric_cols,
        feature_encoders,
        scaler,
        target_encoder,
    ) = preprocess_split(X_selected, y_full)
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Categorical features encoded: {len(selected_categorical_cols)}")
    print(f"Numeric features scaled     : {len(selected_numeric_cols)}")
    print(f"Train class distribution: {format_counts(y_train, target_encoder)}")
    print(f"Test class distribution : {format_counts(y_test, target_encoder)}")

    print_header("STEP 5 - SMOTE")
    X_train_model, y_train_model, smote_applied = maybe_apply_smote(X_train, y_train, target_encoder)

    print_header("STEP 6 - TRAIN 3 MODELS")
    n_classes = len(target_encoder.classes_)
    lgbm_model = build_lightgbm(n_classes, gpu_available)
    xgb_model = build_xgboost(n_classes, gpu_available)
    ensemble_model = VotingClassifier(
        estimators=[
            ("lightgbm", build_lightgbm(n_classes, gpu_available)),
            ("xgboost", build_xgboost(n_classes, gpu_available)),
        ],
        voting="soft",
    )
    print(f"Training device mode: {'GPU' if gpu_available else 'CPU'}")

    model_specs = [
        ("LightGBM", lgbm_model),
        ("XGBoost", xgb_model),
        ("SoftVotingEnsemble", ensemble_model),
    ]
    model_results: dict[str, dict[str, object]] = {}

    for name, estimator in model_specs:
        model = clone(estimator)
        model.fit(X_train_model, y_train_model)
        accuracy, roc_auc, predictions, probabilities = evaluate_model(model, X_test, y_test)
        model_results[name] = {
            "model": model,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "predictions": predictions,
            "probabilities": probabilities,
        }
        print(f"{name} accuracy : {accuracy:.6f}")
        print(f"{name} ROC-AUC  : {roc_auc:.6f}")

    print_header("STEP 7 - HYPERPARAMETER TUNE")
    print_assumption(
        "The best individual model is chosen by Step 6 held-out accuracy, using ROC-AUC as the "
        "tie-breaker. The ensemble is excluded from tuning because you requested tuning of the "
        "best individual model."
    )
    individual_names = ["LightGBM", "XGBoost"]
    best_individual_name = sorted(
        individual_names,
        key=lambda name: (
            model_results[name]["accuracy"],
            model_results[name]["roc_auc"],
        ),
        reverse=True,
    )[0]
    print(f"Best individual model from Step 6: {best_individual_name}")

    tuning_sample_limit = DEFAULT_TUNING_SAMPLE_SIZE
    if best_individual_name == "LightGBM":
        search_estimator = build_lightgbm(n_classes, gpu_available)
        param_distributions = {
            "n_estimators": randint(300, 801),
            "learning_rate": uniform(0.01, 0.05),
            "num_leaves": randint(31, 191),
            "min_child_samples": randint(10, 61),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        }
    else:
        search_estimator = build_xgboost(n_classes, gpu_available)
        if not gpu_available:
            tuning_sample_limit = CPU_XGB_TUNING_SAMPLE_SIZE
            print_assumption(
                f"Because GPU is unavailable, XGBoost tuning is limited to a {tuning_sample_limit:,}-row "
                "stratified sample with a tighter search range so the pipeline finishes in-turn."
            )
        param_distributions = {
            "n_estimators": randint(200, 501),
            "learning_rate": uniform(0.02, 0.04),
            "max_depth": randint(4, 9),
            "min_child_weight": randint(1, 6),
            "subsample": uniform(0.75, 0.25),
            "colsample_bytree": uniform(0.75, 0.25),
        }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tuning_sample_size = min(tuning_sample_limit, len(X_train_model))
    if tuning_sample_size < len(X_train_model):
        print_assumption(
            f"RandomizedSearchCV is run on a stratified sample of {tuning_sample_size:,} training rows to keep "
            "the 20-iteration, 3-fold search practical in-turn. The best params are then "
            "refit on the full training data."
        )
        X_tune, _, y_tune, _ = train_test_split(
            X_train_model,
            y_train_model,
            train_size=tuning_sample_size,
            stratify=y_train_model,
            random_state=42,
        )
    else:
        X_tune, y_tune = X_train_model, y_train_model

    search = RandomizedSearchCV(
        estimator=search_estimator,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="accuracy",
        cv=cv,
        refit=True,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_tune, y_tune)
    tuned_model = clone(search_estimator).set_params(**search.best_params_)
    tuned_model.fit(X_train_model, y_train_model)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.6f}")

    print_header("STEP 8 - FINAL EVALUATION")
    final_accuracy, final_roc_auc, final_predictions, final_probabilities = evaluate_model(
        tuned_model, X_test, y_test
    )
    report = classification_report(
        y_test,
        final_predictions,
        labels=list(range(n_classes)),
        target_names=[str(label) for label in target_encoder.classes_],
        digits=4,
        zero_division=0,
    )
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, final_predictions, labels=list(range(n_classes))),
        index=[str(label) for label in target_encoder.classes_],
        columns=[str(label) for label in target_encoder.classes_],
    )
    delta_vs_old = final_accuracy - OLD_BASELINE_ACCURACY

    print(f"Accuracy: {final_accuracy:.6f}")
    print_subheader("Classification Report")
    print(report)
    print_subheader("Confusion Matrix")
    print(conf_matrix.to_string())
    print_subheader("Weighted ROC-AUC")
    print(f"{final_roc_auc:.6f}")
    print_subheader("Delta vs old baseline")
    print(f"Old baseline accuracy: {OLD_BASELINE_ACCURACY:.4f}")
    print(f"Accuracy delta      : {delta_vs_old:+.6f}")

    print_header("STEP 9 - CONFIDENCE THRESHOLDING FOR THE APPLICATION")
    max_probabilities = final_probabilities.max(axis=1)
    low_confidence_mask = max_probabilities < 0.60
    low_confidence_count = int(low_confidence_mask.sum())
    low_confidence_pct = low_confidence_count / len(y_test) * 100
    print("Low-confidence rule: max class probability < 0.60")
    print(f"Low-confidence cases: {low_confidence_count}")
    print(f"Low-confidence share: {low_confidence_pct:.2f}%")

    print_header("STEP 10 - ADJACENCY ERROR ANALYSIS")
    actual_labels = target_encoder.inverse_transform(y_test)
    predicted_labels = target_encoder.inverse_transform(final_predictions)
    error_distances = np.abs(actual_labels.astype(int) - predicted_labels.astype(int))
    adjacent_errors = int(((error_distances == 1) & (actual_labels != predicted_labels)).sum())
    dangerous_errors = int((error_distances >= 2).sum())
    total_errors = int((actual_labels != predicted_labels).sum())
    print(f"Total errors     : {total_errors}")
    print(f"Adjacent errors  : {adjacent_errors}")
    print(f"Dangerous errors : {dangerous_errors}")

    artifact_bundle = {
        "model": tuned_model,
        "drop_columns": DROP_COLUMNS,
        "created_features": created_features,
        "selected_features": selected_features,
        "categorical_columns": selected_categorical_cols,
        "numeric_columns": selected_numeric_cols,
        "feature_label_encoders": feature_encoders,
        "target_encoder": target_encoder,
        "scaler": scaler,
        "source_file": str(DATA_PATH.name),
        "target_column": TARGET_COL,
        "smote_applied": smote_applied,
        "best_individual_model_name": best_individual_name,
        "old_baseline_accuracy": OLD_BASELINE_ACCURACY,
        "final_accuracy": final_accuracy,
        "final_weighted_roc_auc": final_roc_auc,
    }
    joblib.dump(artifact_bundle, MODEL_PATH)
    joblib.dump(selected_features, FEATURES_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print_subheader("Saved Artifacts")
    print(f"Best model bundle: {MODEL_PATH}")
    print(f"Feature list     : {FEATURES_PATH}")
    print(f"StandardScaler   : {SCALER_PATH}")


if __name__ == "__main__":
    main()
