"""Investigate semantic checks on parsed TDC technician/task tables.

Run from the repository root:
    python src/investigate_dataset_semantics.py

This script reads the processed CSVs and prints diagnostics. It does not alter files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
TECH_PATH = DATA_DIR / "technicians.csv"
TASK_PATH = DATA_DIR / "tasks.csv"


def decode_list(value: object) -> list:
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def main() -> None:
    if not TECH_PATH.exists() or not TASK_PATH.exists():
        print("Processed CSV files are missing. Run:")
        print("  python src/parse_tdc_records.py")
        return

    tech = pd.read_csv(TECH_PATH)
    tasks = pd.read_csv(TASK_PATH)

    tech["skills"] = tech["skills_json"].map(decode_list)
    tasks["skills"] = tasks["skills_json"].map(decode_list)
    tasks["eligible_ids"] = tasks["relevant_technician_ids_json"].map(decode_list)
    tech["start"] = pd.to_datetime(tech["time_window_start"], errors="coerce")
    tech["end"] = pd.to_datetime(tech["time_window_end"], errors="coerce")
    tasks["start"] = pd.to_datetime(tasks["time_window_start"], errors="coerce")
    tasks["end"] = pd.to_datetime(tasks["time_window_end"], errors="coerce")
    tasks["cancelled_bool"] = tasks["cancelled"].astype(str).str.lower().eq("true")

    tech_skill_set = {str(skill) for row in tech["skills"] for skill in row}
    task_skill_set = {str(skill) for row in tasks["skills"] for skill in row}
    all_tech_ids = set(pd.to_numeric(tech["technician_id"], errors="coerce").dropna().astype(int))

    print("DATASET SEMANTICS INVESTIGATION")
    print("===============================")
    print(f"Technician records: {len(tech):,}")
    print(f"Task records: {len(tasks):,}")
    print(f"Technician ID range: {tech['technician_id'].min()} .. {tech['technician_id'].max()}")
    print(f"Technician IDs unique: {tech['technician_id'].is_unique}")
    print(f"Task IDs unique: {tasks['task_id'].is_unique}")

    print("\nTechnician record counts by workday (from time_window_start):")
    counts_by_day = tech["start"].dt.strftime("%Y-%m-%d").value_counts(dropna=False).sort_index()
    for day, count in counts_by_day.items():
        print(f"  {day}: {count}")
    print("Note: records are workday-level rows in this dataset; do not infer distinct master-person count from these rows.")

    print("\nSkill vocabulary:")
    print(f"  Distinct skills in technician records: {len(tech_skill_set)}")
    print(f"  Distinct skills required by tasks:     {len(task_skill_set)}")
    print(f"  Union of both sets:                    {len(tech_skill_set | task_skill_set)}")
    print(f"  Skills only in technician records:     {sorted(tech_skill_set - task_skill_set)}")
    print(f"  Skills only in task records:           {sorted(task_skill_set - tech_skill_set)}")
    print(f"  Full skill vocabulary ({len(tech_skill_set | task_skill_set)}): {sorted(tech_skill_set | task_skill_set)}")

    print("\nEligibility-list diagnostics:")
    empty_eligible = tasks["eligible_ids"].map(len).eq(0)
    print(f"  Empty eligible list: {int(empty_eligible.sum())}")
    print(f"  Not-cancelled + empty list: {int(((~tasks['cancelled_bool']) & empty_eligible).sum())}")
    print(f"  Cancelled + empty list:     {int((tasks['cancelled_bool'] & empty_eligible).sum())}")

    if empty_eligible.any():
        show_columns = [
            "task_id", "skills", "duration", "time_window_start", "time_window_end",
            "priority", "cancelled", "relevant_technician_ids_json",
        ]
        print("\nAll tasks with empty eligible-technician lists:")
        print(tasks.loc[empty_eligible, show_columns].to_string(index=False))

        tech_skill_lookup = {
            int(row.technician_id): set(map(str, row.skills))
            for row in tech.itertuples(index=False)
        }
        for row in tasks.loc[empty_eligible].itertuples(index=False):
            required = set(map(str, row.skills))
            technicians_with_skill = [
                technician_id
                for technician_id, skills in tech_skill_lookup.items()
                if required.issubset(skills)
            ]
            print(
                f"  Task {row.task_id}: required={sorted(required)}, "
                f"workday-tech records with skill={len(technicians_with_skill)}"
            )

    print("\nRelevant-technician ID reference check:")
    referenced_ids = {
        int(technician_id)
        for values in tasks["eligible_ids"]
        for technician_id in values
        if isinstance(technician_id, (int, float)) and float(technician_id).is_integer()
    }
    absent = sorted(referenced_ids - all_tech_ids)
    print(f"  Distinct referenced technician IDs: {len(referenced_ids)}")
    print(f"  IDs not found in technician table: {len(absent)}")
    if absent:
        print(f"  First 30 absent IDs: {absent[:30]}")

    print("\nTask outcome and chronology:")
    print(f"  Cancelled tasks: {int(tasks['cancelled_bool'].sum())}")
    print(f"  Not cancelled:   {int((~tasks['cancelled_bool']).sum())}")
    created = pd.to_datetime(tasks["created"], errors="coerce")
    modified = pd.to_datetime(tasks["modified"], errors="coerce")
    print(f"  Created timestamp parse failures: {int(created.isna().sum())}")
    print(f"  Nonblank modified timestamps parsed: {int(modified.notna().sum())}")
    print(f"  Created after task-window start: {int((created > tasks['start']).sum())}")
    print("  Note: cancelled is a final-state label; predicting it requires careful feature cut-off to avoid leakage.")

    print("\nTravel-time unit reference:")
    print("  The associated paper states that travel times are measured in minutes.")
    print("  Source: https://doi.org/10.1002/net.22188")
    print("  Travel-time rows have not been loaded or modified by this script.")
    print("\nDiagnostic complete; no files were modified.")


if __name__ == "__main__":
    main()
