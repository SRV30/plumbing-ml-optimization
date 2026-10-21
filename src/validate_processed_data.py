"""Validate the locally parsed technician and task CSV files.

Run from the repository root:
    python src/validate_processed_data.py

This report is diagnostic only; it does not modify the CSV files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TECHNICIANS_PATH = PROCESSED_DIR / "technicians.csv"
TASKS_PATH = PROCESSED_DIR / "tasks.csv"


def parse_json_list_column(series: pd.Series, column_name: str) -> tuple[list[list], int]:
    """Decode a JSON-list column and return decoded lists plus invalid count."""
    decoded: list[list] = []
    invalid = 0

    for value in series.fillna(""):
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("Not a JSON list")
            decoded.append(parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            invalid += 1
            decoded.append([])

    if invalid:
        print(f"WARNING: {invalid} invalid JSON list value(s) in {column_name}.")
    return decoded, invalid


def print_missing_summary(frame: pd.DataFrame, label: str) -> None:
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    print(f"\n{label} missing-value counts:")
    if missing.empty:
        print("  No pandas-missing values detected.")
    else:
        for name, count in missing.items():
            print(f"  {name}: {count}")


def main() -> None:
    missing_files = [
        path for path in (TECHNICIANS_PATH, TASKS_PATH) if not path.exists()
    ]
    if missing_files:
        print("Required processed CSV file(s) not found:")
        for path in missing_files:
            print(f"- {path.relative_to(PROJECT_ROOT)}")
        print("Run python src/parse_tdc_records.py first.")
        return

    technicians = pd.read_csv(TECHNICIANS_PATH)
    tasks = pd.read_csv(TASKS_PATH)

    print("PROCESSED DATA VALIDATION")
    print("=========================")
    print(f"Technician table shape: {technicians.shape}")
    print(f"Task table shape:       {tasks.shape}")

    required_technician_columns = {
        "technician_id", "skills_json", "time_window_start",
        "time_window_end", "investment_option"
    }
    required_task_columns = {
        "task_id", "skills_json", "duration", "time_window_start",
        "time_window_end", "priority", "penalty", "created", "modified",
        "cancelled", "relevant_technician_ids_json"
    }

    print("\nRequired-column checks:")
    missing_technician_columns = sorted(required_technician_columns - set(technicians.columns))
    missing_task_columns = sorted(required_task_columns - set(tasks.columns))
    print(f"  Technician missing columns: {missing_technician_columns or 'None'}")
    print(f"  Task missing columns:       {missing_task_columns or 'None'}")
    if missing_technician_columns or missing_task_columns:
        print("STOP: required columns are missing; fix the export before analysis.")
        return

    print("\nIdentifier checks:")
    print(f"  Duplicate technician IDs: {technicians['technician_id'].duplicated().sum()}")
    print(f"  Duplicate task IDs:       {tasks['task_id'].duplicated().sum()}")
    print(f"  Missing technician IDs:  {technicians['technician_id'].isna().sum()}")
    print(f"  Missing task IDs:        {tasks['task_id'].isna().sum()}")

    print_missing_summary(technicians, "Technician table")
    print_missing_summary(tasks, "Task table")

    technician_skills, invalid_technician_skills = parse_json_list_column(
        technicians["skills_json"], "technicians.skills_json"
    )
    task_skills, invalid_task_skills = parse_json_list_column(
        tasks["skills_json"], "tasks.skills_json"
    )
    eligible_lists, invalid_eligible = parse_json_list_column(
        tasks["relevant_technician_ids_json"], "tasks.relevant_technician_ids_json"
    )

    print("\nSkill and eligibility checks:")
    print(f"  Unique technician skill names: {len({s for values in technician_skills for s in values})}")
    print(f"  Unique task-required skill names: {len({s for values in task_skills for s in values})}")
    print(f"  Tasks with empty relevant-technician list: {sum(len(values) == 0 for values in eligible_lists)}")
    print(f"  Tasks marked cancelled: {tasks['cancelled'].astype(str).str.lower().eq('true').sum()}")
    print(f"  Tasks not marked cancelled: {tasks['cancelled'].astype(str).str.lower().eq('false').sum()}")

    cancelled_flags = tasks["cancelled"].astype(str).str.lower()
    empty_eligible = pd.Series([len(values) == 0 for values in eligible_lists], index=tasks.index)
    print(
        "  Not-cancelled tasks with empty eligible list: "
        f"{((cancelled_flags == 'false') & empty_eligible).sum()}"
    )
    print(
        "  Cancelled tasks with empty eligible list:     "
        f"{((cancelled_flags == 'true') & empty_eligible).sum()}"
    )

    tech_start = pd.to_datetime(technicians["time_window_start"], errors="coerce")
    tech_end = pd.to_datetime(technicians["time_window_end"], errors="coerce")
    task_start = pd.to_datetime(tasks["time_window_start"], errors="coerce")
    task_end = pd.to_datetime(tasks["time_window_end"], errors="coerce")

    print("\nTime-window and duration checks:")
    print(f"  Technician invalid start/end timestamps: {int((tech_start.isna() | tech_end.isna()).sum())}")
    print(f"  Technician end before start: {int((tech_end < tech_start).sum())}")
    print(f"  Task invalid start/end timestamps:        {int((task_start.isna() | task_end.isna()).sum())}")
    print(f"  Task end before start:                    {int((task_end < task_start).sum())}")
    print(f"  Tasks with non-positive duration:         {int((pd.to_numeric(tasks['duration'], errors='coerce') <= 0).sum())}")
    print(f"  Tasks with missing/unparseable duration:  {int(pd.to_numeric(tasks['duration'], errors='coerce').isna().sum())}")

    if invalid_technician_skills or invalid_task_skills or invalid_eligible:
        print("\nWARNING: one or more JSON-list columns contain invalid values.")
    else:
        print("\nJSON-list validation: passed.")

    print("\nValidation report complete. No source or processed files were modified.")


if __name__ == "__main__":
    main()
