from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TRIAGE_MODEL_DIR = MODELS_DIR / "triage_v3" / "model"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = TRIAGE_MODEL_DIR / "triage_model_v3.pkl"
FEATURES_PATH = TRIAGE_MODEL_DIR / "feature_cols_v3.pkl"
SCALER_PATH = TRIAGE_MODEL_DIR / "scaler_v3.pkl"
REPORT_PATH = REPORTS_DIR / "latest_model_evaluation.md"
METRICS_PATH = REPORTS_DIR / "latest_model_metrics.json"

CONFIDENCE_THRESHOLD = 0.60
TARGET_COLUMN = "triage_acuity"


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def load_bundle() -> tuple[dict, list[str], object]:
    bundle = joblib.load(MODEL_PATH)
    features = list(joblib.load(FEATURES_PATH))
    scaler = joblib.load(SCALER_PATH)

    if bundle["selected_features"] != features:
        raise ValueError("feature_cols_v3.pkl does not match the selected features in triage_model_v3.pkl")

    return bundle, features, scaler


def clean_dataframe(df: pd.DataFrame, target_col: str, drop_columns: list[str]) -> tuple[pd.DataFrame, pd.Series, int]:
    cleaned = df.drop(columns=drop_columns, errors="ignore").copy()
    cleaned = cleaned.dropna(subset=[target_col]).copy()

    missingness = cleaned.isna().mean().sort_values(ascending=False)

    feature_columns = [column for column in cleaned.columns if column != target_col]
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(cleaned[column]) or pd.api.types.is_bool_dtype(cleaned[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    for column in numeric_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in categorical_columns:
        if cleaned[column].isna().any():
            mode = cleaned[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "missing"
            cleaned[column] = cleaned[column].fillna(fill_value)

    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_rows:
        cleaned = cleaned.drop_duplicates().copy()

    return cleaned, missingness, duplicate_rows


def add_training_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    systolic_safe = engineered["systolic_bp"].replace(0, np.nan)
    engineered["shock_index"] = (engineered["heart_rate"] / systolic_safe).replace([np.inf, -np.inf], np.nan)
    engineered["shock_index"] = engineered["shock_index"].fillna(engineered["shock_index"].median())
    engineered["spo2_resp_interaction"] = engineered["spo2"] * engineered["respiratory_rate"]
    engineered["high_fever_flag"] = (engineered["temperature_c"] > 38.5).astype(int)
    engineered["low_bp_flag"] = (engineered["systolic_bp"] < 90).astype(int)
    engineered["tachycardia_flag"] = (engineered["heart_rate"] > 100).astype(int)
    engineered["hypoxia_flag"] = (engineered["spo2"] < 94).astype(int)
    engineered["elderly_flag"] = (engineered["age"] > 65).astype(int)
    engineered["multi_risk_flag"] = engineered[
        ["high_fever_flag", "low_bp_flag", "tachycardia_flag", "hypoxia_flag", "elderly_flag"]
    ].sum(axis=1)

    return engineered


def compute_runtime_multi_risk(df: pd.DataFrame) -> pd.Series:
    high_fever_flag = (df["temperature_c"] > 38.5).astype(int)
    low_bp_flag = (df["systolic_bp"] < 90).astype(int)
    tachycardia_flag = (df["heart_rate"] > 100).astype(int)
    hypoxia_flag = (df["spo2"] < 94).astype(int)
    elderly_flag = (df["age"] > 65).astype(int)

    return high_fever_flag + low_bp_flag + tachycardia_flag + hypoxia_flag + elderly_flag


def transform_categorical(encoder, series: pd.Series) -> np.ndarray:
    values = series.astype(str).copy()
    unseen_mask = ~values.isin(encoder.classes_)
    if unseen_mask.any():
        unknown_token = "__unknown__"
        if unknown_token not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, unknown_token)
        values.loc[unseen_mask] = unknown_token
    return encoder.transform(values)


def preprocess_frame(
    raw_frame: pd.DataFrame,
    bundle: dict,
    scaler,
) -> pd.DataFrame:
    processed = raw_frame.copy()

    for column, encoder in bundle["feature_label_encoders"].items():
        processed[column] = transform_categorical(encoder, processed[column])

    numeric_columns = bundle["numeric_columns"]
    if numeric_columns:
        processed[numeric_columns] = scaler.transform(processed[numeric_columns].astype(float))

    return processed


def build_metric_tables(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_labels))),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )

    class_rows: list[dict[str, float | str]] = []
    for label in class_labels:
        row = report[label]
        class_rows.append(
            {
                "class": label,
                "precision": round_float(row["precision"]),
                "recall": round_float(row["recall"]),
                "f1_score": round_float(row["f1-score"]),
                "support": int(row["support"]),
            }
        )

    summary = {
        "macro_precision": round_float(report["macro avg"]["precision"]),
        "macro_recall": round_float(report["macro avg"]["recall"]),
        "macro_f1": round_float(report["macro avg"]["f1-score"]),
        "weighted_precision": round_float(report["weighted avg"]["precision"]),
        "weighted_recall": round_float(report["weighted avg"]["recall"]),
        "weighted_f1": round_float(report["weighted avg"]["f1-score"]),
    }
    return class_rows, summary


def compute_weighted_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))


def plot_class_distribution(distribution: pd.Series, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    colors = ["#8b1e3f", "#c04c2c", "#d98f2b", "#6e9c3a", "#2f7f5f"]
    plt.bar(distribution.index.astype(str), distribution.values, color=colors[: len(distribution)])
    plt.title("KTAS Class Distribution in train.csv")
    plt.xlabel("KTAS triage_acuity")
    plt.ylabel("Rows")
    for index, value in enumerate(distribution.values):
        plt.text(index, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_missingness(missingness: pd.Series, path: Path) -> None:
    selected = missingness[missingness > 0].head(12).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(selected.index, selected.values * 100, color="#4b7bec")
    plt.title("Top Missing Features Before Imputation")
    plt.xlabel("Missing values (%)")
    for idx, value in enumerate(selected.values * 100):
        plt.text(value + 0.05, idx, f"{value:.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix on Held-out Test Split")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            color = "white" if cm[row, column] > threshold else "black"
            plt.text(column, row, f"{cm[row, column]}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_columns: list[str], path: Path) -> list[dict[str, float | str]]:
    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top_features = feature_importance.head(15).copy()
    plot_frame = top_features.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_frame["feature"], plot_frame["importance"], color="#1f8a70")
    plt.title("Top 15 XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

    return [
        {
            "feature": row.feature,
            "importance": round_float(row.importance, 6),
        }
        for row in top_features.itertuples(index=False)
    ]


def plot_confidence_analysis(
    confidences: np.ndarray,
    correctness: np.ndarray,
    path: Path,
) -> dict[str, float | list[dict[str, float | str]]]:
    bins = np.array([0.0, 0.60, 0.70, 0.80, 0.90, 1.0])
    bucket_labels = ["<0.60", "0.60-0.69", "0.70-0.79", "0.80-0.89", "0.90-1.00"]
    bucket_rows: list[dict[str, float | str]] = []

    bucket_accuracy = []
    bucket_counts = []
    bucket_positions = np.arange(len(bucket_labels))

    for index in range(len(bins) - 1):
        lower = bins[index]
        upper = bins[index + 1]
        if index == len(bins) - 2:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(mask.sum())
        accuracy = float(correctness[mask].mean()) if count else 0.0
        bucket_counts.append(count)
        bucket_accuracy.append(accuracy)
        bucket_rows.append(
            {
                "bucket": bucket_labels[index],
                "count": count,
                "accuracy": round_float(accuracy),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].hist(confidences, bins=20, color="#287271", edgecolor="white")
    axes[0].axvline(CONFIDENCE_THRESHOLD, color="#b22222", linestyle="--", linewidth=2)
    axes[0].set_title("Max Probability Distribution")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Predictions")

    axes[1].bar(bucket_positions, np.array(bucket_accuracy) * 100, color="#f4a259")
    axes[1].set_title("Accuracy by Confidence Bucket")
    axes[1].set_xlabel("Confidence bucket")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xticks(bucket_positions, bucket_labels, rotation=20)
    axes[1].set_ylim(0, 100)
    for position, accuracy in zip(bucket_positions, bucket_accuracy, strict=False):
        axes[1].text(position, accuracy * 100 + 1, f"{accuracy * 100:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "bucket_metrics": bucket_rows,
        "low_confidence_share": round_float((confidences < CONFIDENCE_THRESHOLD).mean()),
    }


def plot_multiclass_roc(y_true: np.ndarray, y_proba: np.ndarray, class_labels: list[str], path: Path) -> None:
    y_binarized = label_binarize(y_true, classes=list(range(len(class_labels))))

    plt.figure(figsize=(8, 6))
    palette = ["#8b1e3f", "#c04c2c", "#d98f2b", "#6e9c3a", "#2f7f5f"]
    for class_index, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_binarized[:, class_index], y_proba[:, class_index])
        plt.plot(fpr, tpr, label=f"Class {label}", color=palette[class_index % len(palette)], linewidth=2)

    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    plt.title("One-vs-Rest ROC Curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def markdown_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return "\n".join([header_line, separator_line, *body_lines])


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    bundle, feature_columns, scaler = load_bundle()
    data_path = DATA_DIR / bundle["source_file"]

    raw_dataset = pd.read_csv(data_path, low_memory=False)
    cleaned_dataset, missingness, duplicate_rows = clean_dataframe(
        raw_dataset,
        target_col=bundle.get("target_column", TARGET_COLUMN),
        drop_columns=bundle.get("drop_columns", []),
    )
    engineered_dataset = add_training_features(cleaned_dataset)

    auxiliary_inventory = {
        "train_rows": int(len(raw_dataset)),
        "train_columns": int(raw_dataset.shape[1]),
        "chief_complaints_rows": count_csv_rows(DATA_DIR / "chief_complaints.csv"),
        "patient_history_rows": count_csv_rows(DATA_DIR / "patient_history.csv"),
        "test_rows": count_csv_rows(DATA_DIR / "test.csv"),
    }

    X_raw = engineered_dataset[feature_columns].copy()
    y_raw = engineered_dataset[bundle.get("target_column", TARGET_COLUMN)].astype(str).copy()

    (
        X_train_raw,
        X_test_raw,
        y_train_raw,
        y_test_raw,
    ) = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        stratify=y_raw,
        random_state=42,
    )

    target_encoder = bundle["target_encoder"]
    class_labels = [str(label) for label in target_encoder.classes_]
    y_test = target_encoder.transform(y_test_raw)

    X_test_processed = preprocess_frame(X_test_raw, bundle, scaler)
    model = bundle["model"]
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)

    accuracy = float(accuracy_score(y_test, y_pred))
    weighted_roc_auc = compute_weighted_roc_auc(y_test, y_proba)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    weighted_precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    confusion = confusion_matrix(y_test, y_pred, labels=list(range(len(class_labels))))

    confidences = y_proba.max(axis=1)
    correctness = (y_pred == y_test).astype(int)
    low_confidence_mask = confidences < CONFIDENCE_THRESHOLD

    actual_labels = target_encoder.inverse_transform(y_test)
    predicted_labels = target_encoder.inverse_transform(y_pred)
    error_distance = np.abs(actual_labels.astype(int) - predicted_labels.astype(int))
    total_errors = int((actual_labels != predicted_labels).sum())
    adjacent_errors = int(((error_distance == 1) & (actual_labels != predicted_labels)).sum())
    dangerous_errors = int((error_distance >= 2).sum())

    class_report_rows, report_summary = build_metric_tables(y_test, y_pred, class_labels)

    runtime_contract_raw = X_test_raw.copy()
    runtime_contract_raw["multi_risk_flag"] = compute_runtime_multi_risk(runtime_contract_raw)
    runtime_contract_processed = preprocess_frame(runtime_contract_raw, bundle, scaler)
    runtime_contract_predictions = model.predict(runtime_contract_processed)
    runtime_contract_accuracy = float(accuracy_score(y_test, runtime_contract_predictions))
    multi_risk_mismatch_rate = float(
        (runtime_contract_raw["multi_risk_flag"] != X_test_raw["multi_risk_flag"]).mean()
    )

    class_distribution = engineered_dataset[bundle.get("target_column", TARGET_COLUMN)].value_counts().sort_index()
    plot_class_distribution(class_distribution, FIGURES_DIR / "class_distribution_v3.png")
    plot_missingness(missingness, FIGURES_DIR / "missingness_v3.png")
    plot_confusion(confusion, class_labels, FIGURES_DIR / "confusion_matrix_v3.png")
    feature_importance_rows = plot_feature_importance(
        model,
        feature_columns,
        FIGURES_DIR / "feature_importance_v3.png",
    )
    confidence_summary = plot_confidence_analysis(
        confidences,
        correctness,
        FIGURES_DIR / "confidence_analysis_v3.png",
    )
    plot_multiclass_roc(y_test, y_proba, class_labels, FIGURES_DIR / "roc_curves_v3.png")

    clinically_important_missingness = []
    for column in [
        "systolic_bp",
        "diastolic_bp",
        "mean_arterial_pressure",
        "pulse_pressure",
        "shock_index",
        "respiratory_rate",
        "temperature_c",
    ]:
        clinically_important_missingness.append(
            {
                "feature": column,
                "missing_pct": round_float(float(missingness.get(column, 0.0) * 100), 2),
            }
        )

    metrics_payload = {
        "dataset_inventory": auxiliary_inventory,
        "overall_metrics": {
            "accuracy": round_float(accuracy),
            "weighted_roc_auc": round_float(weighted_roc_auc),
            "macro_f1": round_float(macro_f1),
            "weighted_f1": round_float(weighted_f1),
            "weighted_precision": round_float(weighted_precision),
            "weighted_recall": round_float(weighted_recall),
            "low_confidence_share": round_float(float(low_confidence_mask.mean())),
            "test_rows": int(len(y_test)),
            "total_errors": total_errors,
            "adjacent_errors": adjacent_errors,
            "dangerous_errors": dangerous_errors,
            "duplicate_rows_removed": duplicate_rows,
            "old_baseline_accuracy": round_float(bundle.get("old_baseline_accuracy", 0.0)),
            "accuracy_delta_vs_old": round_float(accuracy - float(bundle.get("old_baseline_accuracy", 0.0))),
        },
        "confidence_summary": confidence_summary,
        "runtime_contract_check": {
            "training_style_accuracy": round_float(accuracy),
            "current_service_contract_accuracy": round_float(runtime_contract_accuracy),
            "accuracy_delta": round_float(runtime_contract_accuracy - accuracy),
            "multi_risk_mismatch_rate": round_float(multi_risk_mismatch_rate),
        },
        "per_class_metrics": class_report_rows,
        "feature_importance": feature_importance_rows,
        "clinically_important_missingness": clinically_important_missingness,
        "model_metadata": {
            "model_family": bundle.get("best_individual_model_name", "XGBoost"),
            "source_file": bundle.get("source_file", ""),
            "selected_feature_count": len(feature_columns),
            "smote_applied": bool(bundle.get("smote_applied", False)),
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_lines = [
        "# SmartQ ML Service: v3 Model Evaluation",
        "",
        "## What This Model Checks",
        "",
        "- The model predicts KTAS triage acuity from structured arrival-time data, not free-form text alone.",
        "- It looks at vitals, neurological status, pain, oxygenation, hemodynamic stability, arrival context, and recent emergency-care utilization.",
        "- It also derives secondary risk signals inside the service, including `shock_index`, `mean_arterial_pressure`, `pulse_pressure`, `spo2_resp_interaction`, `hypoxia_flag`, `high_fever_flag`, `tachycardia_flag`, and `multi_risk_flag`.",
        "",
        "## What Most Influences the Score",
        "",
        "- The strongest drivers in the saved v3 model are neurological status (`gcs_total`), the NEWS2 severity score, pain severity, oxygen saturation, prior ED visits, temperature, and blood-pressure-derived risk signals.",
        "- This means the score moves most when the patient shows clear physiological instability or abnormal alertness.",
        "",
        markdown_table(feature_importance_rows, ["feature", "importance"]),
        "",
        "![Top feature importances](figures/feature_importance_v3.png)",
        "",
        "## How Accurate It Is",
        "",
        f"- Accuracy: **{accuracy * 100:.2f}%**",
        f"- Weighted ROC-AUC: **{weighted_roc_auc:.4f}**",
        f"- Macro F1: **{macro_f1:.4f}**",
        f"- Weighted F1: **{weighted_f1:.4f}**",
        f"- Weighted Precision: **{weighted_precision:.4f}**",
        f"- Weighted Recall: **{weighted_recall:.4f}**",
        f"- Low-confidence share at `< {CONFIDENCE_THRESHOLD:.2f}`: **{low_confidence_mask.mean() * 100:.2f}%**",
        f"- Total mistakes on the held-out split: **{total_errors:,} / {len(y_test):,}**",
        f"- Adjacent errors (`|pred-actual| = 1`): **{adjacent_errors:,}**",
        f"- Dangerous errors (`|pred-actual| >= 2`): **{dangerous_errors:,}**",
        f"- Improvement over old baseline accuracy ({bundle.get('old_baseline_accuracy', 0.0):.4f}): **{(accuracy - float(bundle.get('old_baseline_accuracy', 0.0))) * 100:+.02f} percentage points**",
        "",
        "### Per-Class Performance",
        "",
        f"- KTAS 1 recall: **{class_report_rows[0]['recall']:.4f}**",
        f"- KTAS 2 recall: **{class_report_rows[1]['recall']:.4f}**",
        f"- KTAS 3 recall: **{class_report_rows[2]['recall']:.4f}**",
        f"- KTAS 4 recall: **{class_report_rows[3]['recall']:.4f}**",
        f"- KTAS 5 recall: **{class_report_rows[4]['recall']:.4f}**",
        f"- Macro F1 shows balanced multiclass behavior at **{report_summary['macro_f1']:.4f}**.",
        "",
        markdown_table(class_report_rows, ["class", "precision", "recall", "f1_score", "support"]),
        "",
        "![Confusion matrix](figures/confusion_matrix_v3.png)",
        "",
        "## What the Confidence Flag Means",
        "",
        f"- `low_confidence = true` is triggered when the model's highest class probability is below **{CONFIDENCE_THRESHOLD:.2f}**.",
        f"- That flag catches **{low_confidence_mask.sum():,}** of **{len(y_test):,}** held-out predictions.",
        f"- Accuracy is only **{confidence_summary['bucket_metrics'][0]['accuracy'] * 100:.1f}%** below the threshold, but rises to **{confidence_summary['bucket_metrics'][-1]['accuracy'] * 100:.1f}%** in the `0.90-1.00` confidence bucket.",
        "- In practice, this makes the flag useful for manual review and conservative routing.",
        "",
        markdown_table(confidence_summary["bucket_metrics"], ["bucket", "count", "accuracy"]),
        "",
        "![Confidence analysis](figures/confidence_analysis_v3.png)",
        "",
        "## What It Does Not Do",
        "",
        "- It does not explain *why* a patient feels unwell in natural language; it maps structured triage inputs to an acuity class.",
        "- It does not predict queue wait time or doctor consultation duration. That requires SmartQ's own timing data and a separate ETA model.",
        "- It does not replace clinician judgment. It should be used as a decision-support signal with guardrails and low-confidence review.",
        "",
        "## Why SmartQ Uses a Trained Model Instead of Prompt-Only AI Scoring",
        "",
        "- A trained tabular model is deterministic: the same vitals produce the same score every time.",
        "- It is measurable: we can report accuracy, ROC-AUC, class-wise recall, and dangerous-error counts.",
        "- It is cheaper and faster than sending every triage request to a large language model.",
        "- It is also easier to audit, because we can see which structured variables influence the prediction.",
        "",
        "## Dataset Snapshot",
        "",
        f"- Primary modeling table: `train.csv` with {auxiliary_inventory['train_rows']:,} rows and {auxiliary_inventory['train_columns']} columns.",
        f"- Supplementary table: `chief_complaints.csv` with {auxiliary_inventory['chief_complaints_rows']:,} rows.",
        f"- Supplementary table: `patient_history.csv` with {auxiliary_inventory['patient_history_rows']:,} rows.",
        f"- Held-out unlabeled package file: `test.csv` with {auxiliary_inventory['test_rows']:,} rows.",
        f"- Runtime model family: tuned `{bundle.get('best_individual_model_name', 'XGBoost')}` multiclass classifier with {len(feature_columns)} selected features and SMOTE applied during training.",
        "",
        "## Class Balance",
        "",
        markdown_table(
            [
                {
                    "class": str(label),
                    "rows": int(count),
                    "share_pct": round_float(count / len(engineered_dataset) * 100, 2),
                }
                for label, count in class_distribution.items()
            ],
            ["class", "rows", "share_pct"],
        ),
        "",
        "![KTAS class distribution](figures/class_distribution_v3.png)",
        "",
        "## Missingness Snapshot",
        "",
        markdown_table(clinically_important_missingness, ["feature", "missing_pct"]),
        "",
        "![Feature missingness](figures/missingness_v3.png)",
        "",
        "## Error Pattern",
        "",
        "- Most mistakes are adjacent KTAS levels, which is expected in ordinal acuity tasks where the boundary between classes 2/3/4 is clinically noisy.",
        f"- The dangerous-error rate is **{dangerous_errors / len(y_test) * 100:.2f}%** of held-out predictions, which is low but important enough to justify retaining the low-confidence flag and human override path.",
        "",
        "## One-vs-Rest ROC Curves",
        "",
        "![ROC curves](figures/roc_curves_v3.png)",
        "",
        "## Consistency Check Against the Current FastAPI Service",
        "",
        f"- Replaying the held-out split with the saved training-style `multi_risk_flag` gives **{accuracy * 100:.2f}%** accuracy.",
        f"- Replaying the same split while swapping `multi_risk_flag` to the **current FastAPI service definition** gives **{runtime_contract_accuracy * 100:.2f}%** accuracy.",
        f"- The `multi_risk_flag` value changes on **{multi_risk_mismatch_rate * 100:.2f}%** of held-out rows.",
        "- After alignment, these numbers should match. Any future non-zero mismatch here should be treated as training-vs-inference drift.",
        "",
        "## Improvement Opportunities",
        "",
        "- Add probability calibration checks and drift monitoring once SmartQ starts collecting real queue data.",
        "- Collect SmartQ visit outcomes so the model can be revalidated on your own population instead of only on the source dataset split.",
        "- Add explicit audit logging for low-confidence cases, manual overrides, and final doctor-reviewed disposition.",
        "- For future ETA modeling, train a separate regression/forecasting stack instead of reusing this triage dataset. This dataset is strong for acuity, not queue wait time.",
        "",
        "## Repro Commands",
        "",
        "```bash",
        "cd ml_service",
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements-dev.txt",
        "python evaluate_saved_model.py",
        "```",
        "",
        f"Structured metrics are also saved to `{METRICS_PATH.relative_to(BASE_DIR)}`.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Report written to {REPORT_PATH}")
    print(f"Metrics written to {METRICS_PATH}")
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
