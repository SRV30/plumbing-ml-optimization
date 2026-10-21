"""Train and calibrate a conservative Ridge duration estimate chronologically.

Run from repository root:
    python src/train_calibrated_duration_model.py

Temporal split:
- Oldest 60%: train Ridge regression.
- Next 20%: calibrate a one-sided residual buffer.
- Newest 20%: final test set.

The one-sided buffer is the higher 90th percentile of (actual duration - raw
prediction) on the calibration period, floored at zero. This aims to reduce
underprediction; it is not guaranteed to achieve 90% coverage under distribution
shift. The target duration's real decision-time meaning must still be justified.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TASKS_PATH = DATA_DIR / "tasks.csv"

METRICS_PATH = RESULTS_DIR / "duration_calibrated_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "duration_calibrated_test_predictions.csv"
COMPARISON_PATH = RESULTS_DIR / "duration_calibrated_comparison.csv"
PLOT_PATH = FIGURES_DIR / "duration_calibration_comparison.png"

TRAIN_FRACTION = 0.60
CALIBRATION_FRACTION = 0.20
BUFFER_QUANTILE = 0.90


def parse_json_list(value: Any, field: str, task_id: int) -> list:
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Task {task_id}: invalid {field}: {error}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"Task {task_id}: {field} must be a JSON list.")
    return parsed


def make_features(row: dict[str, Any]) -> dict[str, float]:
    """Features considered potentially available at intake; ID/target/outcomes excluded."""
    task_id = int(row["task_id"])
    created = pd.to_datetime(row["created"], errors="coerce")
    window_start = pd.to_datetime(row["time_window_start"], errors="coerce")
    window_end = pd.to_datetime(row["time_window_end"], errors="coerce")
    if pd.isna(created) or pd.isna(window_start) or pd.isna(window_end):
        raise ValueError(f"Task {task_id}: invalid timestamp in features.")

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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "underprediction_rate": float(np.mean(y_pred < y_true)),
        "one_sided_coverage": float(np.mean(y_true <= y_pred)),
    }


def main() -> None:
    if not TASKS_PATH.exists():
        print(f"Task table not found: {TASKS_PATH}")
        print("Run python src/parse_tdc_records.py first.")
        return

    frame = pd.read_csv(TASKS_PATH, keep_default_na=False)
    frame["created_parsed"] = pd.to_datetime(frame["created"], errors="coerce")
    if frame["created_parsed"].isna().any():
        raise ValueError("Some creation timestamps cannot be parsed.")
    frame["task_id_int"] = frame["task_id"].astype(int)
    frame["duration_target"] = pd.to_numeric(frame["duration"], errors="raise").astype(float)
    frame = frame.sort_values(["created_parsed", "task_id_int"]).reset_index(drop=True)

    n = len(frame)
    train_end = int(np.floor(n * TRAIN_FRACTION))
    calibration_end = int(np.floor(n * (TRAIN_FRACTION + CALIBRATION_FRACTION)))
    train = frame.iloc[:train_end]
    calibration = frame.iloc[train_end:calibration_end]
    test = frame.iloc[calibration_end:]

    x_train = [make_features(row) for row in train.to_dict(orient="records")]
    x_cal = [make_features(row) for row in calibration.to_dict(orient="records")]
    x_test = [make_features(row) for row in test.to_dict(orient="records")]
    y_train = train["duration_target"].to_numpy(dtype=float)
    y_cal = calibration["duration_target"].to_numpy(dtype=float)
    y_test = test["duration_target"].to_numpy(dtype=float)

    model = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("scaler", StandardScaler(with_mean=False)),
        ("regressor", Ridge(alpha=10.0)),
    ])
    model.fit(x_train, y_train)
    raw_cal = np.asarray(model.predict(x_cal), dtype=float)
    residuals = y_cal - raw_cal
    calibrated_quantile = float(np.quantile(residuals, BUFFER_QUANTILE, method="higher"))
    buffer = max(0, int(np.ceil(calibrated_quantile)))

    raw_test = np.asarray(model.predict(x_test), dtype=float)
    buffered_test = np.ceil(np.maximum(1.0, raw_test + buffer))
    y_test_int = y_test.astype(int)

    raw_metrics = regression_metrics(y_test, raw_test)
    buffered_metrics = regression_metrics(y_test, buffered_test)

    prediction_rows = []
    for task_id, actual, raw_pred, buffered_pred in zip(
        test["task_id_int"].astype(int), y_test_int, raw_test, buffered_test
    ):
        prediction_rows.append({
            "task_id": int(task_id),
            "model": "Ridge regression",
            "actual_duration_minutes": int(actual),
            "raw_predicted_duration_minutes": round(float(raw_pred), 3),
            # The scheduling-impact script uses this field as the selected estimate.
            "predicted_duration_minutes": int(buffered_pred),
            "calibration_buffer_minutes": buffer,
        })

    results: dict[str, Any] = {
        "experiment": "chronological_duration_prediction_with_one_sided_calibration",
        "target": "source-recorded task duration in minutes",
        "split": {
            "method": "chronological train/calibration/test",
            "train_fraction": TRAIN_FRACTION,
            "calibration_fraction": CALIBRATION_FRACTION,
            "test_fraction": 1 - TRAIN_FRACTION - CALIBRATION_FRACTION,
            "train_rows": int(len(train)),
            "calibration_rows": int(len(calibration)),
            "test_rows": int(len(test)),
            "train_created_min": train["created_parsed"].min().isoformat(),
            "train_created_max": train["created_parsed"].max().isoformat(),
            "calibration_created_min": calibration["created_parsed"].min().isoformat(),
            "calibration_created_max": calibration["created_parsed"].max().isoformat(),
            "test_created_min": test["created_parsed"].min().isoformat(),
            "test_created_max": test["created_parsed"].max().isoformat(),
        },
        "calibration": {
            "method": "nonnegative higher-quantile residual buffer",
            "target_quantile": BUFFER_QUANTILE,
            "raw_residual_quantile_minutes": calibrated_quantile,
            "applied_buffer_minutes": buffer,
            "calibration_actual_coverage": float(np.mean(y_cal <= raw_cal + buffer)),
        },
        "test_metrics_raw_ridge": raw_metrics,
        "test_metrics_buffered_ridge": buffered_metrics,
        "features_excluded": ["task_id", "duration target", "cancelled outcome", "modified timestamp"],
        "caveats": [
            "The dataset is telecom field-service data, not plumbing-company data.",
            "The source paper treats duration as a scheduling input; the prediction use case assumes duration is unavailable at intake.",
            "Features must be confirmed available at the intended prediction time.",
            "A residual buffer may help with underprediction but may increase overprediction and reduce schedulability.",
            "Temporal coverage on a small static dataset is not a deployment guarantee.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    comparison = [
        {"approach": "Raw Ridge", **raw_metrics},
        {"approach": f"Ridge + {buffer}-minute buffer", **buffered_metrics},
    ]
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    names = ["Raw Ridge", f"Ridge + {buffer} min buffer"]
    maes = [raw_metrics["mae_minutes"], buffered_metrics["mae_minutes"]]
    bars = ax.bar(names, maes)
    ax.set_title("Duration-estimate calibration: test MAE")
    ax.set_ylabel("MAE (minutes; lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    print("CALIBRATED RIDGE DURATION EXPERIMENT")
    print("====================================")
    print(f"Training rows: {len(train):,}")
    print(f"Calibration rows: {len(calibration):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Calibration residual quantile: {calibrated_quantile:.2f} minutes")
    print(f"Applied nonnegative buffer: {buffer} minutes")
    print("\nChronological test-set metrics:")
    for name, scores in (("Raw Ridge", raw_metrics), ("Buffered Ridge", buffered_metrics)):
        print(f"\n{name}")
        print(f"  MAE: {scores['mae_minutes']:.2f} minutes")
        print(f"  RMSE: {scores['rmse_minutes']:.2f} minutes")
        print(f"  R-squared: {scores['r2']:.4f}")
        print(f"  Underprediction rate: {scores['underprediction_rate']:.2%}")
        print(f"  One-sided coverage: {scores['one_sided_coverage']:.2%}")

    print("\nSaved outputs:")
    print(f"- {METRICS_PATH.relative_to(ROOT)}")
    print(f"- {COMPARISON_PATH.relative_to(ROOT)}")
    print(f"- {PREDICTIONS_PATH.relative_to(ROOT)}")
    print(f"- {PLOT_PATH.relative_to(ROOT)}")
    print("\nUse the buffered prediction file for the next scheduling-impact experiment.")
    print("This is an exploratory counterfactual, not production proof.")


if __name__ == "__main__":
    main()
