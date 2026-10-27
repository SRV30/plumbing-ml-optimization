"""Evaluate cancellation prediction with a chronological hold-out split.

Run from the repository root:
    python src/train_cancellation_model_temporal.py

Older tasks (by created timestamp) are used for training; the newest 20% form
the test set. The 'modified' field, task ID, and recorded duration are excluded.
Duration is omitted because the dataset does not establish that it is known at
cancellation-prediction time. This evaluates temporal separation better than a
random split, but remains exploratory because the dataset is one static telecom
field-service instance and final-state labels may not describe a deployable process.
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TASKS_PATH = DATA_DIR / "tasks.csv"

METRICS_PATH = RESULTS_DIR / "cancellation_temporal_metrics.json"
COMPARISON_PATH = RESULTS_DIR / "cancellation_temporal_comparison.csv"
PREDICTIONS_PATH = RESULTS_DIR / "cancellation_temporal_predictions.csv"
ROC_PATH = FIGURES_DIR / "cancellation_temporal_roc.png"
PR_PATH = FIGURES_DIR / "cancellation_temporal_precision_recall.png"

TEST_FRACTION = 0.20
RANDOM_STATE = 42


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized cancellation label: {value!r}")


def parse_json_list(value: Any, field: str, task_id: int) -> list:
    try:
        result = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Task {task_id}: invalid {field}: {error}") from error
    if not isinstance(result, list):
        raise ValueError(f"Task {task_id}: {field} must be a JSON list.")
    return result


def make_features(row: dict[str, Any]) -> dict[str, float]:
    """Use candidate pre-decision fields; exclude task ID and modified timestamp."""
    task_id = int(row["task_id"])
    created = pd.to_datetime(row["created"], errors="coerce")
    window_start = pd.to_datetime(row["time_window_start"], errors="coerce")
    window_end = pd.to_datetime(row["time_window_end"], errors="coerce")
    if pd.isna(created) or pd.isna(window_start) or pd.isna(window_end):
        raise ValueError(f"Task {task_id}: invalid date feature.")

    skills = parse_json_list(row["skills_json"], "skills_json", task_id)
    eligible = parse_json_list(
        row["relevant_technician_ids_json"], "relevant_technician_ids_json", task_id
    )
    window_minutes = (window_end - window_start).total_seconds() / 60
    if window_minutes < 0:
        raise ValueError(f"Task {task_id}: negative time-window length.")

    features = {
        "priority": float(row["priority"]),
        "penalty": float(row["penalty"]),
        "window_length_minutes": float(window_minutes),
        "window_start_hour": float(window_start.hour + window_start.minute / 60),
        "window_start_weekday": float(window_start.dayofweek),
        "created_hour": float(created.hour + created.minute / 60),
        "created_weekday": float(created.dayofweek),
        "creation_to_window_start_hours": float((window_start - created).total_seconds() / 3600),
        "required_skill_count": float(len(skills)),
        "eligible_technician_count": float(len(eligible)),
    }
    for skill in skills:
        features[f"required_skill={skill}"] = 1.0
    return features


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int | None]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cancellation_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "cancellation_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "cancellation_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main() -> None:
    if not TASKS_PATH.exists():
        print(f"Task table not found: {TASKS_PATH}")
        print("Run python src/parse_tdc_records.py first.")
        return

    frame = pd.read_csv(TASKS_PATH, keep_default_na=False)
    frame["created_parsed"] = pd.to_datetime(frame["created"], errors="coerce")
    if frame["created_parsed"].isna().any():
        raise ValueError("Some task creation timestamps could not be parsed.")

    frame["target"] = [parse_bool(value) for value in frame["cancelled"]]
    frame["task_id_int"] = frame["task_id"].astype(int)
    frame = frame.sort_values(["created_parsed", "task_id_int"]).reset_index(drop=True)

    test_count = max(1, int(np.ceil(len(frame) * TEST_FRACTION)))
    split_index = len(frame) - test_count
    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()
    y_train = train["target"].to_numpy(dtype=bool)
    y_test = test["target"].to_numpy(dtype=bool)

    if len(np.unique(y_train)) != 2:
        raise ValueError("Chronological training portion does not contain both classes.")
    if len(np.unique(y_test)) != 2:
        raise ValueError(
            "Chronological test portion does not contain both classes. "
            "Inspect class counts by creation date before selecting a different evaluation design."
        )

    x_train = [make_features(row) for row in train.to_dict(orient="records")]
    x_test = [make_features(row) for row in test.to_dict(orient="records")]

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

    result: dict[str, Any] = {
        "experiment": "chronological_cancellation_prediction",
        "target": "cancelled (True positive class)",
        "split": "older 80% by created timestamp for training; latest 20% for testing",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_created_min": train["created_parsed"].min().isoformat(),
        "train_created_max": train["created_parsed"].max().isoformat(),
        "test_created_min": test["created_parsed"].min().isoformat(),
        "test_created_max": test["created_parsed"].max().isoformat(),
        "train_class_counts": {},
        "features_excluded": ["task_id", "modified timestamp", "cancelled target"],
        "caveats": [
            "Single static telecommunications field-service instance, not plumbing-specific data.",
            "A chronological hold-out is more deployment-oriented than a random split but does not prove causal or production performance.",
            "Feature availability at the actual prediction decision time must be confirmed.",
            "The final-state cancellation label may reflect events occurring after the recorded creation timestamp.",
        ],
        "models": {},
    }
    result["train_class_counts"] = {
        "cancelled": int(y_train.sum()),
        "not_cancelled": int((~y_train).sum()),
    }
    result["test_class_counts"] = {
        "cancelled": int(y_test.sum()),
        "not_cancelled": int((~y_test).sum()),
    }

    roc_results = []
    pr_results = []
    summary_rows = []
    prediction_rows = []

    print("CHRONOLOGICAL CANCELLATION PREDICTION")
    print("=====================================")
    print(f"Training rows: {len(train):,} ({train['created_parsed'].min().date()} to {train['created_parsed'].max().date()})")
    print(f"Test rows:     {len(test):,} ({test['created_parsed'].min().date()} to {test['created_parsed'].max().date()})")
    print(f"Training cancellations: {int(y_train.sum()):,} / {len(y_train):,}")
    print(f"Test cancellations:     {int(y_test.sum()):,} / {len(y_test):,}")
    print("Excluded leakage/identity fields: modified, task_id.")
    print("\nTest-set metrics:")

    from sklearn.metrics import roc_curve, precision_recall_curve

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test).astype(bool)
        classes = list(model.named_steps["classifier"].classes_)
        probabilities = model.predict_proba(x_test)[:, classes.index(True)]
        score = metrics(y_test, predictions, probabilities)
        result["models"][name] = score
        summary_rows.append({"model": name, **score})

        print(f"\n{name}")
        print(f"  Accuracy: {score['accuracy']:.4f}")
        print(f"  Balanced accuracy: {score['balanced_accuracy']:.4f}")
        print(f"  Cancellation precision/recall/F1: {score['cancellation_precision']:.4f} / {score['cancellation_recall']:.4f} / {score['cancellation_f1']:.4f}")
        print(f"  Average precision: {score['average_precision']:.4f}")
        print(f"  ROC-AUC: {score['roc_auc']:.4f}")
        print(f"  Confusion matrix TN/FP/FN/TP: {score['true_negative']}/{score['false_positive']}/{score['false_negative']}/{score['true_positive']}")

        fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label=True)
        precision, recall, _ = precision_recall_curve(y_test, probabilities, pos_label=True)
        roc_results.append((name, fpr, tpr))
        pr_results.append((name, precision, recall))

        for task_id, actual, predicted, probability in zip(
            test["task_id_int"].to_numpy(), y_test, predictions, probabilities
        ):
            prediction_rows.append({
                "task_id": int(task_id),
                "model": name,
                "actual_cancelled": bool(actual),
                "predicted_cancelled": bool(predicted),
                "cancellation_probability": round(float(probability), 6),
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, fpr, tpr in roc_results:
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Chronological cancellation prediction — ROC")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, precision, recall in pr_results:
        ax.plot(recall, precision, label=name)
    ax.axhline(float(y_test.mean()), linestyle="--", label="Test prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Chronological cancellation prediction — Precision–Recall")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PR_PATH, dpi=180)
    plt.close(fig)

    print("\nSaved outputs:")
    print(f"- {METRICS_PATH.relative_to(ROOT)}")
    print(f"- {COMPARISON_PATH.relative_to(ROOT)}")
    print(f"- {PREDICTIONS_PATH.relative_to(ROOT)}")
    print(f"- {ROC_PATH.relative_to(ROOT)}")
    print(f"- {PR_PATH.relative_to(ROOT)}")
    print("\nThis is a stricter exploratory evaluation, not production proof.")


if __name__ == "__main__":
    main()
