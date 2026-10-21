"""Evaluate task-duration prediction using a chronological hold-out split.

Run from repository root:
    python src/train_duration_model_temporal.py

Older tasks (ordered by creation timestamp) are used for training and newer tasks
for testing. Duration is the target. Task ID, cancellation outcome, and the final
modification timestamp are excluded from features. This remains exploratory:
confirm that the recorded duration means the quantity you want to estimate and
that each feature would be available at prediction time before making business claims.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TASKS_PATH = DATA_DIR / "tasks.csv"

METRICS_PATH = RESULTS_DIR / "duration_temporal_metrics.json"
COMPARISON_PATH = RESULTS_DIR / "duration_temporal_comparison.csv"
PREDICTIONS_PATH = RESULTS_DIR / "duration_temporal_predictions.csv"
PLOT_PATH = FIGURES_DIR / "duration_temporal_mae_comparison.png"

TEST_FRACTION = 0.20
RANDOM_STATE = 42


def parse_json_list(value: Any, field: str, task_id: int) -> list:
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Task {task_id}: invalid {field}: {error}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"Task {task_id}: {field} must be a JSON list.")
    return parsed


def make_features(row: dict[str, Any]) -> dict[str, float]:
    """Build candidate pre-service features and omit task ID, target and outcome."""
    task_id = int(row["task_id"])
    created = pd.to_datetime(row["created"], errors="coerce")
    window_start = pd.to_datetime(row["time_window_start"], errors="coerce")
    window_end = pd.to_datetime(row["time_window_end"], errors="coerce")
    if pd.isna(created) or pd.isna(window_start) or pd.isna(window_end):
        raise ValueError(f"Task {task_id}: invalid timestamp in duration features.")

    skills = parse_json_list(row["skills_json"], "skills_json", task_id)
    eligible_ids = parse_json_list(
        row["relevant_technician_ids_json"], "relevant_technician_ids_json", task_id
    )
    window_minutes = (window_end - window_start).total_seconds() / 60
    if window_minutes < 0:
        raise ValueError(f"Task {task_id}: negative window length.")

    features: dict[str, float] = {
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


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return {
        "mae_minutes": float(mean_absolute_error(y_true, predictions)),
        "rmse_minutes": rmse,
        "median_absolute_error_minutes": float(median_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def main() -> None:
    if not TASKS_PATH.exists():
        print(f"Task table not found: {TASKS_PATH}")
        print("Run python src/parse_tdc_records.py first.")
        return

    tasks = pd.read_csv(TASKS_PATH, keep_default_na=False)
    tasks["created_parsed"] = pd.to_datetime(tasks["created"], errors="coerce")
    if tasks["created_parsed"].isna().any():
        raise ValueError("Some creation timestamps could not be parsed.")

    tasks["task_id_int"] = tasks["task_id"].astype(int)
    tasks["duration_target"] = pd.to_numeric(tasks["duration"], errors="raise").astype(float)
    if (tasks["duration_target"] <= 0).any():
        raise ValueError("Duration target has non-positive values.")

    tasks = tasks.sort_values(["created_parsed", "task_id_int"]).reset_index(drop=True)
    test_count = max(1, int(np.ceil(len(tasks) * TEST_FRACTION)))
    split_index = len(tasks) - test_count
    train = tasks.iloc[:split_index].copy()
    test = tasks.iloc[split_index:].copy()

    x_train = [make_features(row) for row in train.to_dict(orient="records")]
    x_test = [make_features(row) for row in test.to_dict(orient="records")]
    y_train = train["duration_target"].to_numpy(dtype=float)
    y_test = test["duration_target"].to_numpy(dtype=float)
    test_ids = test["task_id_int"].to_numpy(dtype=int)

    models = {
        "Median baseline": Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("regressor", DummyRegressor(strategy="median")),
        ]),
        "Ridge regression": Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            ("regressor", Ridge(alpha=10.0)),
        ]),
        "Random forest": Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("regressor", RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    results: dict[str, Any] = {
        "experiment": "chronological_task_duration_prediction",
        "target": "duration minutes",
        "split": "oldest 80% by created timestamp for training; newest 20% for testing",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_created_min": train["created_parsed"].min().isoformat(),
        "train_created_max": train["created_parsed"].max().isoformat(),
        "test_created_min": test["created_parsed"].min().isoformat(),
        "test_created_max": test["created_parsed"].max().isoformat(),
        "train_duration_summary": {
            "mean": float(np.mean(y_train)),
            "median": float(np.median(y_train)),
            "min": float(np.min(y_train)),
            "max": float(np.max(y_train)),
        },
        "test_duration_summary": {
            "mean": float(np.mean(y_test)),
            "median": float(np.median(y_test)),
            "min": float(np.min(y_test)),
            "max": float(np.max(y_test)),
        },
        "features_excluded": [
            "task_id (identifier)",
            "duration (target)",
            "cancelled (final outcome)",
            "modified timestamp (possible outcome leakage)",
        ],
        "caveats": [
            "Dataset comes from telecommunications field service, not a plumbing company.",
            "This estimates the recorded duration field; confirm whether it represents planned duration or realized service time.",
            "Feature availability at the intended prediction time must be established.",
            "One chronological hold-out on one static instance is exploratory and not proof of deployment performance.",
        ],
        "models": {},
    }

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    y_model_predictions: dict[str, np.ndarray] = {}

    print("CHRONOLOGICAL TASK-DURATION PREDICTION")
    print("======================================")
    print(f"Training rows: {len(train):,} ({train['created_parsed'].min().date()} to {train['created_parsed'].max().date()})")
    print(f"Test rows:     {len(test):,} ({test['created_parsed'].min().date()} to {test['created_parsed'].max().date()})")
    print(f"Training duration median/mean: {np.median(y_train):.2f} / {np.mean(y_train):.2f} minutes")
    print(f"Test duration median/mean:     {np.median(y_test):.2f} / {np.mean(y_test):.2f} minutes")
    print("\nTest-set results:")

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_test), dtype=float)
        score = regression_metrics(y_test, predictions)
        results["models"][name] = score
        metrics_rows.append({"model": name, **score})
        y_model_predictions[name] = predictions

        print(f"\n{name}")
        print(f"  MAE:                    {score['mae_minutes']:.3f} minutes")
        print(f"  RMSE:                   {score['rmse_minutes']:.3f} minutes")
        print(f"  Median absolute error:  {score['median_absolute_error_minutes']:.3f} minutes")
        print(f"  R-squared:              {score['r2']:.4f}")

        for task_id, actual, predicted in zip(test_ids, y_test, predictions):
            prediction_rows.append({
                "task_id": int(task_id),
                "model": name,
                "actual_duration_minutes": round(float(actual), 3),
                "predicted_duration_minutes": round(float(predicted), 3),
                "absolute_error_minutes": round(float(abs(actual - predicted)), 3),
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    names = list(models.keys())
    maes = [results["models"][name]["mae_minutes"] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(names, maes)
    ax.set_title("Chronological task-duration prediction error")
    ax.set_ylabel("Mean absolute error (minutes; lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    print("\nSaved outputs:")
    print(f"- {METRICS_PATH.relative_to(ROOT)}")
    print(f"- {COMPARISON_PATH.relative_to(ROOT)}")
    print(f"- {PREDICTIONS_PATH.relative_to(ROOT)}")
    print(f"- {PLOT_PATH.relative_to(ROOT)}")
    print("\nInterpret against the median baseline; lower MAE/RMSE and credible held-out R-squared are desirable.")
    print("Treat this as exploratory until the target and prediction-time fields are justified.")


if __name__ == "__main__":
    main()
