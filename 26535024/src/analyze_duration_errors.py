"""Analyze duration-prediction errors by actual task-duration range.

Run from repository root:
    python src/analyze_duration_errors.py

Reads temporal test predictions from the duration experiment and compares errors
by duration bucket. It does not modify model outputs or task data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TASKS_PATH = DATA_DIR / "tasks.csv"
PREDICTIONS_PATH = RESULTS_DIR / "duration_temporal_predictions.csv"
OUTPUT_PATH = RESULTS_DIR / "duration_error_by_range.csv"
PLOT_PATH = FIGURES_DIR / "duration_mae_by_range.png"

# Intuitive, increasing ranges for error diagnosis; labels use actual duration.
DURATION_BINS = [0, 15, 30, 60, 120, np.inf]
DURATION_LABELS = ["1–15", "16–30", "31–60", "61–120", "121+"]


def main() -> None:
    for path in (TASKS_PATH, PREDICTIONS_PATH):
        if not path.exists():
            print(f"Required file not found: {path.relative_to(ROOT)}")
            if path == PREDICTIONS_PATH:
                print("Run python src/train_duration_model_temporal.py first.")
            return

    tasks = pd.read_csv(TASKS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    actual = tasks[["task_id", "duration"]].copy()
    actual["task_id"] = actual["task_id"].astype(int)
    actual["duration"] = pd.to_numeric(actual["duration"], errors="raise")
    source_ids = set(actual["task_id"])

    predictions["task_id"] = predictions["task_id"].astype(int)
    missing_ids = sorted(set(predictions["task_id"]) - source_ids)
    if missing_ids:
        raise ValueError(f"Prediction IDs missing from task source: {missing_ids[:10]}")

    data = predictions.copy()
    if "actual_duration_minutes" not in data.columns:
        data = data.merge(
            actual.rename(columns={"duration": "actual_duration_minutes"}),
            on="task_id",
            how="left",
            validate="many_to_one",
        )
    data["actual_duration_minutes"] = pd.to_numeric(
        data["actual_duration_minutes"], errors="raise"
    )
    if data["actual_duration_minutes"].isna().any():
        raise ValueError("Some prediction rows are missing actual durations.");

    data["predicted_duration_minutes"] = pd.to_numeric(
        data["predicted_duration_minutes"], errors="raise"
    )
    data["error_minutes"] = (
        data["predicted_duration_minutes"] - data["actual_duration_minutes"]
    )
    data["absolute_error_minutes"] = data["error_minutes"].abs()
    data["actual_duration_range"] = pd.cut(
        data["actual_duration_minutes"],
        bins=DURATION_BINS,
        labels=DURATION_LABELS,
        include_lowest=True,
        right=True,
    )

    rows: list[dict[str, object]] = []
    for (model_name, duration_range), group in data.groupby(
        ["model", "actual_duration_range"], observed=False
    ):
        if group.empty:
            continue
        rows.append({
            "model": model_name,
            "actual_duration_range_minutes": str(duration_range),
            "test_tasks": int(len(group)),
            "mae_minutes": float(group["absolute_error_minutes"].mean()),
            "rmse_minutes": float(np.sqrt(np.mean(group["error_minutes"] ** 2))),
            "mean_signed_error_minutes": float(group["error_minutes"].mean()),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in summary.groupby("model"):
        group = group.set_index("actual_duration_range_minutes").reindex(DURATION_LABELS)
        ax.plot(
            DURATION_LABELS,
            group["mae_minutes"],
            marker="o",
            label=model_name,
        )
    ax.set_xlabel("Actual task duration (minutes)")
    ax.set_ylabel("Mean absolute error (minutes)")
    ax.set_title("Chronological duration-prediction error by actual duration")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    print("DURATION ERROR ANALYSIS")
    print("=======================")
    print("Positive mean signed error = overprediction; negative = underprediction.")
    print(summary.to_string(index=False, formatters={
        "mae_minutes": "{:.2f}".format,
        "rmse_minutes": "{:.2f}".format,
        "mean_signed_error_minutes": "{:.2f}".format,
    }))
    print("\nSaved:")
    print(f"- {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"- {PLOT_PATH.relative_to(ROOT)}")
    print("\nUse this to identify where predictions are weak before testing scheduling impact.")


if __name__ == "__main__":
    main()
