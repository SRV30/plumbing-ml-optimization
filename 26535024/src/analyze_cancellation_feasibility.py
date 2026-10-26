"""Audit whether cancellation prediction is plausible without target leakage.

Run from the repository root:
    python src/analyze_cancellation_feasibility.py

This performs exploratory checks only; it does not train a model. It writes
readable summary tables and plots under data/processed/ and results/figures/.
The 'modified' field is audited as a possible leakage variable and is not
accepted as a model feature by this script.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
TASKS_PATH = DATA / "tasks.csv"
FIGURES = ROOT / "results" / "figures"

CROSSTAB_PATH = DATA / "cancellation_modified_crosstab.csv"
SKILL_SUMMARY_PATH = DATA / "cancellation_by_skill.csv"
SUMMARY_PATH = DATA / "cancellation_feasibility_summary.json"


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized cancellation value: {value!r}")


def main() -> None:
    if not TASKS_PATH.exists():
        print(f"Missing input: {TASKS_PATH.relative_to(ROOT)}")
        print("Run python src/parse_tdc_records.py first.")
        return

    tasks = pd.read_csv(TASKS_PATH, keep_default_na=False)
    if "cancelled" not in tasks or "modified" not in tasks:
        print("Expected cancelled and modified columns were not found.")
        return

    tasks["cancelled_bool"] = tasks["cancelled"].map(parse_bool)
    tasks["modified_recorded"] = tasks["modified"].astype(str).str.strip().ne("")
    tasks["created_dt"] = pd.to_datetime(tasks["created"], errors="coerce")
    tasks["window_start_dt"] = pd.to_datetime(tasks["time_window_start"], errors="coerce")
    tasks["window_end_dt"] = pd.to_datetime(tasks["time_window_end"], errors="coerce")
    tasks["duration_num"] = pd.to_numeric(tasks["duration"], errors="coerce")
    tasks["priority_num"] = pd.to_numeric(tasks["priority"], errors="coerce")
    tasks["penalty_num"] = pd.to_numeric(tasks["penalty"], errors="coerce")

    tasks["window_length_minutes"] = (
        tasks["window_end_dt"] - tasks["window_start_dt"]
    ).dt.total_seconds() / 60
    tasks["creation_lead_hours"] = (
        tasks["window_start_dt"] - tasks["created_dt"]
    ).dt.total_seconds() / 3600
    tasks["created_hour"] = tasks["created_dt"].dt.hour
    tasks["created_weekday"] = tasks["created_dt"].dt.dayofweek

    target_counts = tasks["cancelled_bool"].value_counts().to_dict()
    canceled = int(target_counts.get(True, 0))
    not_canceled = int(target_counts.get(False, 0))
    prevalence = canceled / len(tasks) if len(tasks) else 0.0

    crosstab = pd.crosstab(
        tasks["modified_recorded"].map({True: "modified_recorded", False: "no_modification_recorded"}),
        tasks["cancelled_bool"].map({True: "cancelled_true", False: "cancelled_false"}),
        dropna=False,
    )
    crosstab = crosstab.reindex(
        index=["no_modification_recorded", "modified_recorded"],
        columns=["cancelled_false", "cancelled_true"],
        fill_value=0,
    )
    crosstab.index.name = "modified_field_status"
    crosstab.to_csv(CROSSTAB_PATH)

    
    records: list[dict[str, Any]] = []
    for _, row in tasks.iterrows():
        skills = json.loads(row["skills_json"])
        for skill in skills:
            records.append({"skill": skill, "cancelled": bool(row["cancelled_bool"])})
    skill_frame = pd.DataFrame(records, columns=["skill", "cancelled"])
    if not skill_frame.empty:
        skill_summary = (
            skill_frame.groupby("skill")["cancelled"]
            .agg(tasks="size", cancelled_tasks="sum", cancellation_rate="mean")
            .reset_index()
            .sort_values("tasks", ascending=False)
        )
    else:
        skill_summary = pd.DataFrame(
            columns=["skill", "tasks", "cancelled_tasks", "cancellation_rate"]
        )
    skill_summary.to_csv(SKILL_SUMMARY_PATH, index=False)

    print("CANCELLATION PREDICTION FEASIBILITY AUDIT")
    print("========================================")
    print(f"Total task records: {len(tasks):,}")
    print(f"Cancelled: {canceled:,} ({prevalence:.2%})")
    print(f"Not cancelled: {not_canceled:,} ({1-prevalence:.2%})")
    print(f"Simple majority-class baseline accuracy: {max(canceled, not_canceled)/len(tasks):.2%}")
    print("\nCancellation vs. whether a modified timestamp is recorded:")
    print(crosstab.to_string())
    print(
        "\nModified timestamp recorded for cancelled tasks: "
        f"{int((tasks['modified_recorded'] & tasks['cancelled_bool']).sum()):,} / {canceled:,}"
    )
    print(
        "Modified timestamp recorded for non-cancelled tasks: "
        f"{int((tasks['modified_recorded'] & ~tasks['cancelled_bool']).sum()):,} / {not_canceled:,}"
    )
    print("\nCandidate pre-outcome features to review (not yet approved for modeling):")
    for column in [
        "duration_num", "priority_num", "penalty_num", "window_length_minutes",
        "creation_lead_hours", "created_hour", "created_weekday"
    ]:
        series = tasks[column]
        print(
            f"  {column}: missing={int(series.isna().sum()):,}; "
            f"min={series.min() if series.notna().any() else 'NA'}; "
            f"median={series.median() if series.notna().any() else 'NA'}; "
            f"max={series.max() if series.notna().any() else 'NA'}"
        )

    created_after_window = int(
        (tasks["created_dt"] > tasks["window_start_dt"]).fillna(False).sum()
    )
    print(f"\nTasks where creation time is after window start: {created_after_window:,}")
    print("Interpretation: this is a data-quality/availability question, not automatically an error.")
    print("\nLeakage warning:")
    print(
        "  Do NOT use the modified timestamp as a cancellation predictor unless the "
        "prediction time is explicitly after that modification."
    )
    print(
        "  Do NOT use the cancellation flag, any direct encoding of it, or post-outcome "
        "fields as input features."
    )
    print(
        "  Before ML training, define a prediction cutoff and verify that every feature "
        "would be known by that cutoff."
    )

    FIGURES.mkdir(parents=True, exist_ok=True)

    class_labels = ["Not cancelled", "Cancelled"]
    class_values = [not_canceled, canceled]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(class_labels, class_values)
    ax.set_title("Task cancellation label distribution")
    ax.set_ylabel("Task count")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.0f")
    fig.tight_layout()
    class_plot = FIGURES / "cancellation_class_distribution.png"
    fig.savefig(class_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    crosstab.plot(kind="bar", ax=ax)
    ax.set_title("Cancellation vs. modified timestamp availability")
    ax.set_xlabel("Modified field status")
    ax.set_ylabel("Task count")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    modification_plot = FIGURES / "cancellation_modified_crosstab.png"
    fig.savefig(modification_plot, dpi=180)
    plt.close(fig)

    summary = {
        "total_tasks": int(len(tasks)),
        "cancelled_tasks": canceled,
        "not_cancelled_tasks": not_canceled,
        "cancellation_prevalence": round(prevalence, 6),
        "majority_class_accuracy_baseline": round(max(canceled, not_canceled) / len(tasks), 6),
        "tasks_with_modified_timestamp": int(tasks["modified_recorded"].sum()),
        "cancelled_with_modified_timestamp": int((tasks["modified_recorded"] & tasks["cancelled_bool"]).sum()),
        "not_cancelled_with_modified_timestamp": int((tasks["modified_recorded"] & ~tasks["cancelled_bool"]).sum()),
        "created_after_window_start": created_after_window,
        "modified_field_is_excluded_from_features": True,
        "model_training_started": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAudit outputs:")
    print(f"- {CROSSTAB_PATH.relative_to(ROOT)}")
    print(f"- {SKILL_SUMMARY_PATH.relative_to(ROOT)}")
    print(f"- {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"- {class_plot.relative_to(ROOT)}")
    print(f"- {modification_plot.relative_to(ROOT)}")
    print("\nNo ML model was trained by this script.")


if __name__ == "__main__":
    main()
