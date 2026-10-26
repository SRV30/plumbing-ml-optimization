"""Run an initial cancellation-prediction experiment.

Run from repository root:
    python src/train_cancellation_model.py

The dataset has a final-state 'cancelled' label. The 'modified' field is
intentionally excluded because the current validation found nonblank modified
timestamps on exactly the cancelled-task count (444), making it a strong leakage
risk. Task IDs are also excluded.

This is an exploratory held-out test, not evidence of production performance.
It uses fields in the static instance that might be known for planning, but their
availability at a real cancellation-prediction decision time must be justified.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TASKS_PATH = DATA_DIR / "tasks.csv"

METRICS_PATH = RESULTS_DIR / "cancellation_model_metrics.json"
COMPARISON_PATH = RESULTS_DIR / "cancellation_model_comparison.csv"
PREDICTIONS_PATH = RESULTS_DIR / "cancellation_test_predictions.csv"
ROC_FIGURE_PATH = FIGURES_DIR / "cancellation_roc_comparison.png"
PR_FIGURE_PATH = FIGURES_DIR / "cancellation_precision_recall.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized cancellation label: {value!r}")


def parse_json_list(value: Any, field: str, task_id: int) -> list:
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Task {task_id}: invalid JSON in {field}: {error}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"Task {task_id}: {field} must be a JSON list.")
    return parsed


def make_features(row: dict[str, Any]) -> dict[str, float]:
    """Build features while intentionally excluding task ID and modified time."""
    task_id = int(row["task_id"])
    created = pd.to_datetime(row["created"], errors="coerce")
    window_start = pd.to_datetime(row["time_window_start"], errors="coerce")
    window_end = pd.to_datetime(row["time_window_end"], errors="coerce")
    if pd.isna(created) or pd.isna(window_start) or pd.isna(window_end):
        raise ValueError(f"Task {task_id}: invalid creation or window timestamp.")

    skills = parse_json_list(row["skills_json"], "skills_json", task_id)
    eligible_ids = parse_json_list(
        row["relevant_technician_ids_json"],
        "relevant_technician_ids_json",
        task_id,
    )

    window_minutes = (window_end - window_start).total_seconds() / 60
    if window_minutes < 0:
        raise ValueError(f"Task {task_id}: negative time-window length.")

    
    
    features: dict[str, float] = {
        "duration_minutes": float(row["duration"]),
        "priority": float(row["priority"]),
        "penalty": float(row["penalty"]),
        "window_length_minutes": float(window_minutes),
        "window_start_hour": float(window_start.hour + window_start.minute / 60),
        "window_start_weekday": float(window_start.dayofweek),
        "created_hour": float(created.hour + created.minute / 60),
        "created_weekday": float(created.dayofweek),
        "creation_to_window_start_hours": float(
            (window_start - created).total_seconds() / 3600
        ),
        "required_skill_count": float(len(skills)),
        "eligible_technician_count": float(len(eligible_ids)),
    }
    for skill in skills:
        
        features[f"required_skill={skill}"] = 1.0
    return features


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute class-sensitive and ranking metrics; cancellation=True is positive."""
    result: dict[str, float | int | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cancellation_precision": float(precision_score(y_true, y_pred, pos_label=True, zero_division=0)),
        "cancellation_recall": float(recall_score(y_true, y_pred, pos_label=True, zero_division=0)),
        "cancellation_f1": float(f1_score(y_true, y_pred, pos_label=True, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    result.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    })
    return result


def main() -> None:
    if not TASKS_PATH.exists():
        print(f"Processed task file not found: {TASKS_PATH}")
        print("Run python src/parse_tdc_records.py first.")
        return

    tasks = pd.read_csv(TASKS_PATH, keep_default_na=False)
    if "cancelled" not in tasks.columns:
        raise ValueError("The task CSV has no 'cancelled' target field.")

    y = np.array([parse_bool(value) for value in tasks["cancelled"]], dtype=bool)
    feature_records = [
        make_features(row)
        for row in tasks.to_dict(orient="records")
    ]
    task_ids = tasks["task_id"].astype(int).to_numpy()

    classes, counts = np.unique(y, return_counts=True)
    class_counts = {str(bool(label)): int(count) for label, count in zip(classes, counts)}
    if len(classes) != 2 or min(counts) < 2:
        raise ValueError("Cancellation target needs both classes and at least two rows per class.")

    indices = np.arange(len(tasks))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_train, y_test = y[train_idx], y[test_idx]
    x_train = [feature_records[i] for i in train_idx]
    x_test = [feature_records[i] for i in test_idx]

    models = {
        "Majority baseline": Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic regression": Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", LogisticRegression(
                max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "Random forest": Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    output: dict[str, Any] = {
        "experiment": "initial_cancellation_prediction",
        "target": "cancelled (True is the positive class)",
        "split": {
            "method": "stratified random hold-out",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        },
        "class_counts_full_data": class_counts,
        "features_used": [
            "duration", "priority", "penalty", "time-window length",
            "window start hour and weekday", "creation hour and weekday",
            "creation-to-window-start lead time", "number of required skills",
            "number of relevant technicians", "required-skill indicator features",
        ],
        "features_excluded_for_leakage_or_identity": [
            "modified timestamp (strong leakage risk: observed nonblank count equals cancelled count)",
            "task_id (record identifier, not a generalizable business feature)",
            "cancelled itself (target)",
        ],
        "caveats": [
            "Exploratory hold-out result on a single static dataset.",
            "The source data is telecommunications field service, not plumbing-company data.",
            "A real deployment decision time and the availability of every feature must be established before claiming predictive performance.",
            "The hold-out is random/stratified, not a future-time deployment simulation.",
        ],
        "models": {},
    }

    prediction_rows = []
    roc_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    pr_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    comparison_rows = []

    print("CANCELLATION PREDICTION EXPERIMENT")
    print("=================================")
    print(f"Tasks: {len(tasks):,}")
    print(f"Cancellation class counts: {class_counts}")
    print(f"Training rows: {len(train_idx):,}; test rows: {len(test_idx):,}")
    print("Leakage precaution: 'modified' timestamp and task ID are excluded.")
    print("\nTest-set results (cancellation=True is positive):")

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test).astype(bool)
        y_prob = model.predict_proba(x_test)[:, list(model.named_steps["classifier"].classes_).index(True)]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        output["models"][name] = metrics
        comparison_rows.append({"model": name, **metrics})

        print(f"\n{name}")
        print(f"  Accuracy:             {metrics['accuracy']:.4f}")
        print(f"  Balanced accuracy:    {metrics['balanced_accuracy']:.4f}")
        print(f"  Cancellation precision: {metrics['cancellation_precision']:.4f}")
        print(f"  Cancellation recall:    {metrics['cancellation_recall']:.4f}")
        print(f"  Cancellation F1:        {metrics['cancellation_f1']:.4f}")
        print(f"  Average precision:      {metrics['average_precision']:.4f}")
        print(f"  ROC-AUC:                {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}")
        print(
            "  Confusion matrix TN/FP/FN/TP: "
            f"{metrics['true_negative']}/{metrics['false_positive']}/"
            f"{metrics['false_negative']}/{metrics['true_positive']}"
        )

        prediction_rows_for_model = []
        for idx, true_value, pred_value, probability in zip(test_idx, y_test, y_pred, y_prob):
            row = {
                "task_id": int(task_ids[idx]),
                "model": name,
                "actual_cancelled": bool(true_value),
                "predicted_cancelled": bool(pred_value),
                "cancellation_probability": round(float(probability), 6),
            }
            prediction_rows.append(row)
            prediction_rows_for_model.append(row)

        if metrics["roc_auc"] is not None:
            from sklearn.metrics import roc_curve, precision_recall_curve
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=True)
            roc_data.append((name, fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=True)
            pr_data.append((name, precision, recall))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(comparison_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(prediction_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    if roc_data:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, fpr, tpr in roc_data:
            ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Cancellation prediction — ROC curve")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(ROC_FIGURE_PATH, dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for name, precision, recall in pr_data:
            ax.plot(recall, precision, label=name)
        prevalence = float(y_test.mean())
        ax.axhline(prevalence, linestyle="--", label=f"Test cancellation prevalence ({prevalence:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Cancellation prediction — Precision–Recall curve")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PR_FIGURE_PATH, dpi=180)
        plt.close(fig)

    print("\nSaved outputs:")
    print(f"- {METRICS_PATH.relative_to(ROOT)}")
    print(f"- {COMPARISON_PATH.relative_to(ROOT)}")
    print(f"- {PREDICTIONS_PATH.relative_to(ROOT)}")
    if roc_data:
        print(f"- {ROC_FIGURE_PATH.relative_to(ROOT)}")
        print(f"- {PR_FIGURE_PATH.relative_to(ROOT)}")
    print("\nInterpretation: compare against the majority baseline and inspect precision/recall.")
    print("Do not treat this random hold-out as proof of future or plumbing-business performance.")


if __name__ == "__main__":
    main()
