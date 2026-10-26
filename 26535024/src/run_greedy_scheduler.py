"""Run a transparent greedy baseline technician scheduler.

Run from the repository root:
    python src/run_greedy_scheduler.py

Inputs are the processed task/technician CSVs and validated .npy travel matrices.
Outputs are written locally to data/processed/ (Git-ignored).
This is a deterministic heuristic baseline, not an optimal solver or an ML model.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

TASKS_PATH = DATA_DIR / "tasks.csv"
TECHNICIANS_PATH = DATA_DIR / "technicians.csv"
TASK_TO_TASK_PATH = DATA_DIR / "task_to_task.npy"
TASK_TO_TECH_PATH = DATA_DIR / "task_to_technician.npy"
TECH_TO_TASK_PATH = DATA_DIR / "technician_to_task.npy"

SCHEDULE_PATH = DATA_DIR / "baseline_task_status.csv"
ROUTES_PATH = DATA_DIR / "baseline_routes.csv"
SUMMARY_PATH = DATA_DIR / "baseline_summary.json"

STATUS_FIELDS = [
    "task_id",
    "cancelled",
    "status",
    "reason",
    "technician_id",
    "route_sequence",
    "service_start",
    "service_finish",
    "inbound_travel_minutes",
    "return_travel_minutes",
    "duration_minutes",
    "priority",
    "penalty",
]

ROUTE_FIELDS = [
    "technician_id",
    "route_sequence",
    "task_id",
    "service_start",
    "service_finish",
    "inbound_travel_minutes",
    "return_travel_minutes",
    "duration_minutes",
    "priority",
    "penalty",
]


def json_list(value: Any, field: str, row_id: int) -> list:
    try:
        result = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Row {row_id}: invalid JSON in {field}: {error}") from error
    if not isinstance(result, list):
        raise ValueError(f"Row {row_id}: {field} must contain a JSON list.")
    return result


def boolean_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    required = [
        TASKS_PATH,
        TECHNICIANS_PATH,
        TASK_TO_TASK_PATH,
        TASK_TO_TECH_PATH,
        TECH_TO_TASK_PATH,
    ]
    absent = [path for path in required if not path.exists()]
    if absent:
        print("Required input files are missing:")
        for path in absent:
            print(f"- {path.relative_to(ROOT)}")
        print("\nRun the parser and matrix builder first.")
        return

    tasks = pd.read_csv(TASKS_PATH)
    technicians = pd.read_csv(TECHNICIANS_PATH)

    
    task_to_task = np.load(TASK_TO_TASK_PATH, mmap_mode="r")
    task_to_technician = np.load(TASK_TO_TECH_PATH, mmap_mode="r")
    technician_to_task = np.load(TECH_TO_TASK_PATH, mmap_mode="r")

    expected_shapes = {
        "task_to_task": (len(tasks), len(tasks)),
        "task_to_technician": (len(tasks), len(technicians)),
        "technician_to_task": (len(technicians), len(tasks)),
    }
    actual_shapes = {
        "task_to_task": task_to_task.shape,
        "task_to_technician": task_to_technician.shape,
        "technician_to_task": technician_to_task.shape,
    }
    if expected_shapes != actual_shapes:
        print("Travel-matrix shape mismatch.")
        print("Expected:", expected_shapes)
        print("Actual:  ", actual_shapes)
        return

    tech_by_id: dict[int, dict[str, Any]] = {}
    for row in technicians.to_dict(orient="records"):
        technician_id = int(row["technician_id"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError(f"Invalid time window for technician {technician_id}.")
        tech_by_id[technician_id] = {
            "start": start,
            "end": end,
            "last_task_id": None,
            "last_finish": start,
            "task_ids": [],
            "travel_before_return": 0.0,
            "service_minutes": 0,
        }

    task_records: list[dict[str, Any]] = []
    result_by_task: dict[int, dict[str, Any]] = {}
    cancelled_count = 0
    no_candidate_count = 0

    for row in tasks.to_dict(orient="records"):
        task_id = int(row["task_id"])
        cancelled = boolean_value(row["cancelled"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        duration = int(row["duration"])
        candidates = json_list(
            row["relevant_technician_ids_json"],
            "relevant_technician_ids_json",
            task_id,
        )
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError(f"Invalid time window for task {task_id}.")
        if duration <= 0:
            raise ValueError(f"Non-positive duration for task {task_id}.")
        if not all(isinstance(candidate, int) for candidate in candidates):
            raise ValueError(f"Task {task_id} has a non-integer candidate ID.")

        task = {
            "task_id": task_id,
            "cancelled": cancelled,
            "start": start,
            "end": end,
            "duration": duration,
            "priority": int(row["priority"]),
            "penalty": float(row["penalty"]),
            "candidates": candidates,
        }
        task_records.append(task)

        if cancelled:
            cancelled_count += 1
            result_by_task[task_id] = {
                "task_id": task_id,
                "cancelled": True,
                "status": "excluded_cancelled",
                "reason": "Source record is marked cancelled.",
                "technician_id": "",
                "route_sequence": "",
                "service_start": "",
                "service_finish": "",
                "inbound_travel_minutes": "",
                "return_travel_minutes": "",
                "duration_minutes": duration,
                "priority": task["priority"],
                "penalty": task["penalty"],
            }
        elif not candidates:
            no_candidate_count += 1
            result_by_task[task_id] = {
                "task_id": task_id,
                "cancelled": False,
                "status": "unassigned",
                "reason": "Source record contains no relevant technician IDs.",
                "technician_id": "",
                "route_sequence": "",
                "service_start": "",
                "service_finish": "",
                "inbound_travel_minutes": "",
                "return_travel_minutes": "",
                "duration_minutes": duration,
                "priority": task["priority"],
                "penalty": task["penalty"],
            }

    
    
    
    active_tasks = [
        task for task in task_records
        if not task["cancelled"] and task["candidates"]
    ]
    active_tasks.sort(
        key=lambda task: (
            task["priority"],
            -task["penalty"],
            task["end"],
            task["start"],
            task["task_id"],
        )
    )

    assigned_count = 0
    infeasible_count = 0
    for task in active_tasks:
        task_id = task["task_id"]
        best: dict[str, Any] | None = None

        for technician_id in task["candidates"]:
            state = tech_by_id.get(technician_id)
            if state is None:
                continue

            last_task_id = state["last_task_id"]
            if last_task_id is None:
                incoming = float(technician_to_task[technician_id - 1, task_id - 1])
            else:
                incoming = float(task_to_task[last_task_id - 1, task_id - 1])

            return_travel = float(task_to_technician[task_id - 1, technician_id - 1])
            if not np.isfinite(incoming) or not np.isfinite(return_travel):
                continue
            if incoming < 0 or return_travel < 0:
                continue

            arrival = state["last_finish"] + pd.Timedelta(minutes=incoming)
            service_start = max(task["start"], arrival)
            service_finish = service_start + pd.Timedelta(minutes=task["duration"])

            
            
            if service_finish > task["end"]:
                continue
            if service_finish > state["end"]:
                continue
            if service_finish + pd.Timedelta(minutes=return_travel) > state["end"]:
                continue

            if last_task_id is None:
                travel_delta = incoming + return_travel
            else:
                previous_return = float(
                    task_to_technician[last_task_id - 1, technician_id - 1]
                )
                
                
                travel_delta = incoming + return_travel - previous_return

            
            
            score = (
                travel_delta,
                state["service_minutes"],
                len(state["task_ids"]),
                service_finish,
                technician_id,
            )
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "technician_id": technician_id,
                    "incoming": incoming,
                    "return_travel": return_travel,
                    "service_start": service_start,
                    "service_finish": service_finish,
                }

        if best is None:
            infeasible_count += 1
            result_by_task[task_id] = {
                "task_id": task_id,
                "cancelled": False,
                "status": "unassigned",
                "reason": "No candidate fits the current greedy route and time constraints.",
                "technician_id": "",
                "route_sequence": "",
                "service_start": "",
                "service_finish": "",
                "inbound_travel_minutes": "",
                "return_travel_minutes": "",
                "duration_minutes": task["duration"],
                "priority": task["priority"],
                "penalty": task["penalty"],
            }
            continue

        technician_id = best["technician_id"]
        state = tech_by_id[technician_id]
        state["task_ids"].append(task_id)
        state["last_task_id"] = task_id
        state["last_finish"] = best["service_finish"]
        state["travel_before_return"] += best["incoming"]
        state["service_minutes"] += task["duration"]
        assigned_count += 1

        result_by_task[task_id] = {
            "task_id": task_id,
            "cancelled": False,
            "status": "assigned",
            "reason": "",
            "technician_id": technician_id,
            "route_sequence": len(state["task_ids"]),
            "service_start": best["service_start"].isoformat(),
            "service_finish": best["service_finish"].isoformat(),
            "inbound_travel_minutes": round(best["incoming"], 3),
            "return_travel_minutes": "",
            "duration_minutes": task["duration"],
            "priority": task["priority"],
            "penalty": task["penalty"],
        }

    
    route_rows: list[dict[str, Any]] = []
    total_travel_minutes = 0.0
    route_count = 0

    for technician_id, state in tech_by_id.items():
        task_ids = state["task_ids"]
        if not task_ids:
            continue

        route_count += 1
        last_task_id = task_ids[-1]
        final_return = float(task_to_technician[last_task_id - 1, technician_id - 1])
        total_travel_minutes += state["travel_before_return"] + final_return

        last_row = result_by_task[last_task_id]
        last_row["return_travel_minutes"] = round(final_return, 3)

        for task_id in task_ids:
            row = result_by_task[task_id]
            route_rows.append({
                "technician_id": technician_id,
                "route_sequence": row["route_sequence"],
                "task_id": task_id,
                "service_start": row["service_start"],
                "service_finish": row["service_finish"],
                "inbound_travel_minutes": row["inbound_travel_minutes"],
                "return_travel_minutes": row["return_travel_minutes"],
                "duration_minutes": row["duration_minutes"],
                "priority": row["priority"],
                "penalty": row["penalty"],
            })

    schedule_rows = [result_by_task[task_id] for task_id in sorted(result_by_task)]
    route_rows.sort(key=lambda row: (row["technician_id"], row["route_sequence"]))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(SCHEDULE_PATH, STATUS_FIELDS, schedule_rows)
    write_csv(ROUTES_PATH, ROUTE_FIELDS, route_rows)

    total_tasks = len(task_records)
    active_count = total_tasks - cancelled_count
    unassigned_count = sum(row["status"] == "unassigned" for row in schedule_rows)
    coverage = (assigned_count / active_count) if active_count else 0.0
    travel_minutes_by_route = {
        str(technician_id): round(
            state["travel_before_return"]
            + float(task_to_technician[state["last_task_id"] - 1, technician_id - 1]),
            3,
        )
        for technician_id, state in tech_by_id.items()
        if state["task_ids"]
    }
    summary = {
        "baseline": "deterministic_greedy_append_only",
        "note": "Heuristic baseline, not an optimal solution or ML model.",
        "total_task_records": total_tasks,
        "cancelled_excluded": cancelled_count,
        "active_tasks": active_count,
        "assigned_tasks": assigned_count,
        "unassigned_tasks": unassigned_count,
        "unassigned_with_empty_source_eligibility": no_candidate_count,
        "unassigned_with_no_feasible_greedy_candidate": infeasible_count,
        "technician_routes_used": route_count,
        "active_task_coverage": round(coverage, 6),
        "total_travel_minutes_including_depot_returns": round(total_travel_minutes, 3),
        "total_service_minutes": int(sum(
            row["duration"] for row in task_records
            if result_by_task[row["task_id"]]["status"] == "assigned"
        )),
        "total_route_count_with_assignments": len(travel_minutes_by_route),
        "task_to_task_matrix_shape": list(task_to_task.shape),
        "task_to_technician_matrix_shape": list(task_to_technician.shape),
        "technician_to_task_matrix_shape": list(technician_to_task.shape),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nGREEDY BASELINE SCHEDULER SUMMARY")
    print("=================================")
    print(f"Total task records: {total_tasks:,}")
    print(f"Cancelled tasks excluded: {cancelled_count:,}")
    print(f"Active tasks considered: {active_count:,}")
    print(f"Assigned tasks: {assigned_count:,}")
    print(f"Unassigned tasks: {unassigned_count:,}")
    print(f"  Empty source eligibility list: {no_candidate_count:,}")
    print(f"  No feasible greedy candidate: {infeasible_count:,}")
    print(f"Technician routes used: {route_count:,}")
    print(f"Active-task coverage: {coverage:.2%}")
    print(f"Total travel minutes including depot returns: {total_travel_minutes:,.2f}")
    print(f"Total scheduled service minutes: {summary['total_service_minutes']:,}")
    print("\nLocal outputs:")
    print(f"- {SCHEDULE_PATH.relative_to(ROOT)}")
    print(f"- {ROUTES_PATH.relative_to(ROOT)}")
    print(f"- {SUMMARY_PATH.relative_to(ROOT)}")
    print("\nReminder: this greedy schedule is a baseline to evaluate, not an optimal solution.")


if __name__ == "__main__":
    main()
