"""Train a quantile duration model for risk-aware technician scheduling.

Run from the repository root:
    python src/train_duration_quantile_model.py

Uses an 80/20 chronological split by task creation time. The median model
estimates a typical duration; the 80th-quantile model gives a conservative
planning estimate intended to reduce schedule overruns. The outputs are
exploratory and must not be interpreted as validated plumbing-business results.

Inputs:
    data/processed/tasks.csv

Outputs:
    results/duration_quantile_metrics.json
    results/duration_quantile_predictions.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_pinball_loss, median_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "tasks.csv"
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "duration_quantile_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "duration_quantile_predictions.csv"
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
    """Build candidate pre-service features; exclude IDs and final outcomes."""
    task_id = int(row["task_id"])
    created = pd.to_datetime(row["created"], errors="coerce")
    window_start = pd.to_datetime(row["time_window_start"], errors="coerce")
    window_end = pd.to_datetime(row["time_window_end"], errors="coerce")
    if pd.isna(created) or pd.isna(window_start) or pd.isna(window_end):
        raise ValueError(f"Task {task_id}: invalid timestamp in features.")

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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error_minutes": float(median_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH.relative_to(ROOT)}. Run src/parse_tdc_records.py first."
        )

    frame = pd.read_csv(DATA_PATH, keep_default_na=False)
    frame["created_parsed"] = pd.to_datetime(frame["created"], errors="coerce")
    if frame["created_parsed"].isna().any():
        raise ValueError("Some task creation timestamps could not be parsed.")
    frame["task_id_int"] = pd.to_numeric(frame["task_id"], errors="raise").astype(int)
    frame["duration_target"] = pd.to_numeric(frame["duration"], errors="raise").astype(float)
    if (frame["duration_target"] <= 0).any():
        raise ValueError("Duration target contains non-positive values.")

    frame = frame.sort_values(["created_parsed", "task_id_int"]).reset_index(drop=True)
    test_count = max(1, int(np.ceil(len(frame) * TEST_FRACTION)))
    split_index = len(frame) - test_count
    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()
    if train.empty or test.empty:
        raise ValueError("Not enough rows for a chronological train/test split.")

    x_train_records = [make_features(row) for row in train.to_dict(orient="records")]
    x_test_records = [make_features(row) for row in test.to_dict(orient="records")]
    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(x_train_records)
    x_test = vectorizer.transform(x_test_records)
    y_train = train["duration_target"].to_numpy(dtype=float)
    y_test = test["duration_target"].to_numpy(dtype=float)

    
    median_baseline = DummyRegressor(strategy="median").fit(
        np.zeros((len(y_train), 1)), y_train
    )
    baseline_pred = median_baseline.predict(np.zeros((len(y_test), 1)))

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(x_train, y_train)
    ridge_pred = np.maximum(1.0, ridge.predict(x_test))

    
    
    q50_model = GradientBoostingRegressor(
        loss="quantile", alpha=0.50, n_estimators=200, learning_rate=0.04,
        max_depth=2, min_samples_leaf=8, random_state=RANDOM_STATE,
    )
    q80_model = GradientBoostingRegressor(
        loss="quantile", alpha=0.80, n_estimators=200, learning_rate=0.04,
        max_depth=2, min_samples_leaf=8, random_state=RANDOM_STATE,
    )
    q50_model.fit(x_train, y_train)
    q80_model.fit(x_train, y_train)
    q50 = np.maximum(1.0, q50_model.predict(x_test))
    q80 = np.maximum(q50, q80_model.predict(x_test))

    q80_coverage = float(np.mean(y_test <= q80))
    metrics = {
        "experiment": "chronological_quantile_duration_prediction",
        "target": "recorded duration in minutes",
        "split": "oldest 80% by created timestamp train; newest 20% test",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_created_min": train["created_parsed"].min().isoformat(),
        "train_created_max": train["created_parsed"].max().isoformat(),
        "test_created_min": test["created_parsed"].min().isoformat(),
        "test_created_max": test["created_parsed"].max().isoformat(),
        "features_excluded": [
            "task_id (identifier)", "duration (target)", "cancelled (final outcome)",
            "modified timestamp (possible outcome leakage)",
        ],
        "models": {
            "median_baseline": regression_metrics(y_test, baseline_pred),
            "ridge_point_estimate": regression_metrics(y_test, ridge_pred),
            "gradient_boosting_q50": {
                **regression_metrics(y_test, q50),
                "pinball_loss": float(mean_pinball_loss(y_test, q50, alpha=0.50)),
            },
            "gradient_boosting_q80_planning_estimate": {
                "pinball_loss": float(mean_pinball_loss(y_test, q80, alpha=0.80)),
                "empirical_coverage": q80_coverage,
                "target_quantile": 0.80,
                "mean_planned_duration_minutes": float(np.mean(q80)),
                "mean_underprediction_minutes": float(np.mean(np.maximum(y_test - q80, 0))),
                "underprediction_rate": float(np.mean(y_test > q80)),
                "mean_buffer_vs_q50_minutes": float(np.mean(q80 - q50)),
            },
        },
        "caveats": [
            "The source dataset is telecommunications field-service data, not plumbing data.",
            "The target is the recorded duration field; its meaning must be confirmed before operational use.",
            "The 80th quantile is a conservative candidate, not a guarantee of schedule feasibility.",
            "A single chronological hold-out does not establish production performance.",
            "Only features available at the actual planning time should be used.",
        ],
    }

    prediction_rows = []
    for idx, (_, row) in enumerate(test.iterrows()):
        prediction_rows.append({
            "task_id": int(row["task_id_int"]),
            "created": row["created_parsed"].isoformat(),
            "actual_duration_minutes": float(y_test[idx]),
            "median_baseline_minutes": float(baseline_pred[idx]),
            "ridge_predicted_minutes": float(ridge_pred[idx]),
            "q50_predicted_duration_minutes": float(q50[idx]),
            "q80_planned_duration_minutes": float(q80[idx]),
            "q80_underpredicted": bool(y_test[idx] > q80[idx]),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    print("QUANTILE DURATION MODEL — CHRONOLOGICAL HOLD-OUT")
    print("================================================")
    print(f"Training rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Median baseline MAE: {metrics['models']['median_baseline']['mae_minutes']:.2f} min")
    print(f"Ridge MAE:           {metrics['models']['ridge_point_estimate']['mae_minutes']:.2f} min")
    print(f"Gradient boosting q50 MAE: {metrics['models']['gradient_boosting_q50']['mae_minutes']:.2f} min")
    print(f"q80 empirical coverage: {q80_coverage:.2%} (target 80%)")
    print(f"Mean q80 planning buffer over q50: {np.mean(q80 - q50):.2f} min")
    print(f"Saved {METRICS_PATH.relative_to(ROOT)}")
    print(f"Saved {PREDICTIONS_PATH.relative_to(ROOT)}")
    print("Interpret results cautiously: this is a research prototype on non-plumbing data.")


if __name__ == "__main__":
    main()
