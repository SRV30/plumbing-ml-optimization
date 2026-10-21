"""Run a route-insertion heuristic for technician scheduling.

Run from repository root:
    python src/run_insertion_scheduler.py

This experiment is an alternative heuristic to the append-only greedy baseline.
For each task, it tries inserting the task at every position of each candidate
technician-day route, then selects the feasible insertion with the lowest change
in total route travel time. It is not an exact/optimal solver or ML model.

Outputs are local and Git-ignored:
    data/processed/insertion_task_status.csv
    data/processed/insertion_routes.csv
    data/processed/insertion_summary.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

TASKS_PATH = DATA / "tasks.csv"
TECHNICIANS_PATH = DATA / "technicians.csv"
T2T_PATH = DATA / "task_to_task.npy"
T2TECH_PATH = DATA / "task_to_technician.npy"
TECH2T_PATH = DATA / "technician_to_task.npy"

STATUS_PATH = DATA / "insertion_task_status.csv"
ROUTES_PATH = DATA / "insertion_routes.csv"
SUMMARY_PATH = DATA / "insertion_summary.json"

STATUS_FIELDS = [
    "task_id", "cancelled", "status", "reason", "technician_id",
    "route_sequence", "service_start", "service_finish",
    "inbound_travel_minutes", "return_travel_minutes",
    "duration_minutes", "priority", "penalty",
]
ROUTE_FIELDS = [
    "technician_id", "route_sequence", "task_id", "service_start",
    "service_finish", "inbound_travel_minutes", "return_travel_minutes",
    "duration_minutes", "priority", "penalty",
]


def decode_list(value: Any, field: str, row_id: int) -> list:
    try:
        result = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Task {row_id}: invalid JSON in {field}: {error}") from error
    if not isinstance(result, list):
        raise ValueError(f"Task {row_id}: {field} must be a JSON list.")
    return result


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_route(
    technician_id: int,
    route_task_ids: list[int],
    tasks_by_id: dict[int, dict[str, Any]],
    shift_start: pd.Timestamp,
    shift_end: pd.Timestamp,
    task_to_task: np.ndarray,
    task_to_technician: np.ndarray,
    technician_to_task: np.ndarray,
) -> dict[str, Any] | None:
    """Return a feasible route schedule and travel total, or None if infeasible."""
    if not route_task_ids:
        return {"rows": [], "travel_minutes": 0.0}

    previous_task_id: int | None = None
    previous_finish = shift_start
    travel_total = 0.0
    scheduled_rows: list[dict[str, Any]] = []

    for sequence, task_id in enumerate(route_task_ids, start=1):
        task = tasks_by_id[task_id]
        if previous_task_id is None:
            inbound = float(technician_to_task[technician_id - 1, task_id - 1])
        else:
            inbound = float(task_to_task[previous_task_id - 1, task_id - 1])

        if not np.isfinite(inbound) or inbound < 0:
            return None

        arrival = previous_finish + pd.Timedelta(minutes=inbound)
        service_start = max(task["start"], arrival)
        service_finish = service_start + pd.Timedelta(minutes=task["duration"])

        if service_start < task["start"] or service_finish > task["end"]:
            return None
        if service_start < shift_start or service_finish > shift_end:
            return None

        travel_total += inbound
        scheduled_rows.append({
            "technician_id": technician_id,
            "route_sequence": sequence,
            "task_id": task_id,
            "service_start": service_start.isoformat(),
            "service_finish": service_finish.isoformat(),
            "inbound_travel_minutes": round(inbound, 3),
            "return_travel_minutes": "",
            "duration_minutes": task["duration"],
            "priority": task["priority"],
            "penalty": task["penalty"],
        })
        previous_task_id = task_id
        previous_finish = service_finish

    assert previous_task_id is not None
    return_travel = float(task_to_technician[previous_task_id - 1, technician_id - 1])
    if not np.isfinite(return_travel) or return_travel < 0:
        return None
    if previous_finish + pd.Timedelta(minutes=return_travel) > shift_end:
        return None

    travel_total += return_travel
    scheduled_rows[-1]["return_travel_minutes"] = round(return_travel, 3)

    return {"rows": scheduled_rows, "travel_minutes": travel_total}


def main() -> None:
    required = [TASKS_PATH, TECHNICIANS_PATH, T2T_PATH, T2TECH_PATH, TECH2T_PATH]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Required input files are missing:")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        print("Run parsing and matrix-building steps first.")
        return

    tasks_frame = pd.read_csv(TASKS_PATH)
    tech_frame = pd.read_csv(TECHNICIANS_PATH)

    task_to_task = np.load(T2T_PATH, mmap_mode="r")
    task_to_technician = np.load(T2TECH_PATH, mmap_mode="r")
    technician_to_task = np.load(TECH2T_PATH, mmap_mode="r")

    if task_to_task.shape != (len(tasks_frame), len(tasks_frame)):
        raise ValueError("Task-to-task matrix shape does not match task table.")
    if task_to_technician.shape != (len(tasks_frame), len(tech_frame)):
        raise ValueError("Task-to-technician matrix shape does not match input tables.")
    if technician_to_task.shape != (len(tech_frame), len(tasks_frame)):
        raise ValueError("Technician-to-task matrix shape does not match input tables.")

    tasks_by_id: dict[int, dict[str, Any]] = {}
    status_by_id: dict[int, dict[str, Any]] = {}
    cancelled_count = 0
    empty_eligibility_count = 0

    for row in tasks_frame.to_dict(orient="records"):
        task_id = int(row["task_id"])
        cancelled = parse_bool(row["cancelled"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        duration = int(row["duration"])
        candidates = decode_list(row["relevant_technician_ids_json"], "eligible technicians", task_id)
        if pd.isna(start) or pd.isna(end) or end < start or duration <= 0:
            raise ValueError(f"Task {task_id} has invalid scheduling data.")
        if not all(isinstance(value, int) and value > 0 for value in candidates):
            raise ValueError(f"Task {task_id} contains invalid technician IDs.")

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
        tasks_by_id[task_id] = task

        if cancelled:
            cancelled_count += 1
            status_by_id[task_id] = {
                "task_id": task_id, "cancelled": True,
                "status": "excluded_cancelled",
                "reason": "Source record is marked cancelled.",
                "technician_id": "", "route_sequence": "",
                "service_start": "", "service_finish": "",
                "inbound_travel_minutes": "", "return_travel_minutes": "",
                "duration_minutes": duration, "priority": task["priority"],
                "penalty": task["penalty"],
            }
        elif not candidates:
            empty_eligibility_count += 1
            status_by_id[task_id] = {
                "task_id": task_id, "cancelled": False,
                "status": "unassigned",
                "reason": "Source record contains no relevant technician IDs.",
                "technician_id": "", "route_sequence": "",
                "service_start": "", "service_finish": "",
                "inbound_travel_minutes": "", "return_travel_minutes": "",
                "duration_minutes": duration, "priority": task["priority"],
                "penalty": task["penalty"],
            }

    technician_windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    technician_routes: dict[int, list[int]] = {}
    for row in tech_frame.to_dict(orient="records"):
        technician_id = int(row["technician_id"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError(f"Technician {technician_id} has an invalid time window.")
        technician_windows[technician_id] = (start, end)
        technician_routes[technician_id] = []

    active_tasks = [
        task for task in tasks_by_id.values()
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

    assigned = 0
    no_feasible_insertion = 0

    for index, task in enumerate(active_tasks, start=1):
        if index % 250 == 0:
            print(f"Scheduling task {index:,} of {len(active_tasks):,}...")

        best: dict[str, Any] | None = None
        for technician_id in task["candidates"]:
            if technician_id not in technician_routes:
                continue
            shift_start, shift_end = technician_windows[technician_id]
            route = technician_routes[technician_id]
            old_eval = evaluate_route(
                technician_id, route, tasks_by_id, shift_start, shift_end,
                task_to_task, task_to_technician, technician_to_task,
            )
            if old_eval is None:
                raise RuntimeError(f"Existing route for technician {technician_id} is infeasible.")

            for position in range(len(route) + 1):
                proposed_route = route[:position] + [task["task_id"]] + route[position:]
                proposed_eval = evaluate_route(
                    technician_id, proposed_route, tasks_by_id, shift_start, shift_end,
                    task_to_task, task_to_technician, technician_to_task,
                )
                if proposed_eval is None:
                    continue

                marginal_travel = proposed_eval["travel_minutes"] - old_eval["travel_minutes"]
                # Prefer lower marginal travel, then shorter current route, then a
                # stable technician ID and insertion position for reproducibility.
                score = (
                    marginal_travel,
                    len(route),
                    technician_id,
                    position,
                )
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "technician_id": technician_id,
                        "route": proposed_route,
                        "evaluation": proposed_eval,
                    }

        if best is None:
            no_feasible_insertion += 1
            status_by_id[task["task_id"]] = {
                "task_id": task["task_id"], "cancelled": False,
                "status": "unassigned",
                "reason": "No feasible insertion found by this heuristic.",
                "technician_id": "", "route_sequence": "",
                "service_start": "", "service_finish": "",
                "inbound_travel_minutes": "", "return_travel_minutes": "",
                "duration_minutes": task["duration"], "priority": task["priority"],
                "penalty": task["penalty"],
            }
            continue

        technician_id = best["technician_id"]
        technician_routes[technician_id] = best["route"]
        assigned += 1

        # Temporarily set status; route sequence/times are finalized after all insertions.
        status_by_id[task["task_id"]] = {
            "task_id": task["task_id"], "cancelled": False,
            "status": "assigned", "reason": "",
            "technician_id": technician_id, "route_sequence": "",
            "service_start": "", "service_finish": "",
            "inbound_travel_minutes": "", "return_travel_minutes": "",
            "duration_minutes": task["duration"], "priority": task["priority"],
            "penalty": task["penalty"],
        }

    route_rows: list[dict[str, Any]] = []
    total_travel = 0.0
    route_count = 0
    service_minutes = 0

    for technician_id, route in technician_routes.items():
        if not route:
            continue
        shift_start, shift_end = technician_windows[technician_id]
        evaluation = evaluate_route(
            technician_id, route, tasks_by_id, shift_start, shift_end,
            task_to_task, task_to_technician, technician_to_task,
        )
        if evaluation is None:
            raise RuntimeError(f"Final route for technician {technician_id} is infeasible.")
        route_count += 1
        total_travel += evaluation["travel_minutes"]

        for row in evaluation["rows"]:
            route_rows.append(row)
            status = status_by_id[row["task_id"]]
            status["route_sequence"] = row["route_sequence"]
            status["service_start"] = row["service_start"]
            status["service_finish"] = row["service_finish"]
            status["inbound_travel_minutes"] = row["inbound_travel_minutes"]
            status["return_travel_minutes"] = row["return_travel_minutes"]
            service_minutes += int(row["duration_minutes"])

    route_rows.sort(key=lambda row: (row["technician_id"], row["route_sequence"]))
    status_rows = [status_by_id[task_id] for task_id in sorted(status_by_id)]

    # Internal accounting assertions prevent incomplete results from being written.
    if len(status_rows) != len(tasks_by_id):
        raise RuntimeError("Status output does not include exactly one row per source task.")
    if sum(row["status"] == "assigned" for row in status_rows) != assigned:
        raise RuntimeError("Assignment count mismatch while preparing outputs.")

    write_csv(STATUS_PATH, STATUS_FIELDS, status_rows)
    write_csv(ROUTES_PATH, ROUTE_FIELDS, route_rows)

    active_count = len(tasks_by_id) - cancelled_count
    unassigned_count = sum(row["status"] == "unassigned" for row in status_rows)
    summary = {
        "baseline": "priority_ordered_route_insertion",
        "note": "Insertion heuristic; not an optimal solver or ML model.",
        "total_task_records": len(tasks_by_id),
        "cancelled_excluded": cancelled_count,
        "active_tasks": active_count,
        "assigned_tasks": assigned,
        "unassigned_tasks": unassigned_count,
        "unassigned_with_empty_source_eligibility": empty_eligibility_count,
        "unassigned_with_no_feasible_insertion": no_feasible_insertion,
        "technician_routes_used": route_count,
        "active_task_coverage": round(assigned / active_count, 6) if active_count else 0.0,
        "total_travel_minutes_including_depot_returns": round(total_travel, 3),
        "total_service_minutes": service_minutes,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nROUTE-INSERTION SCHEDULER SUMMARY")
    print("=================================")
    print(f"Total task records: {len(tasks_by_id):,}")
    print(f"Cancelled tasks excluded: {cancelled_count:,}")
    print(f"Active tasks considered: {active_count:,}")
    print(f"Assigned tasks: {assigned:,}")
    print(f"Unassigned tasks: {unassigned_count:,}")
    print(f"  Empty source eligibility list: {empty_eligibility_count:,}")
    print(f"  No feasible insertion found: {no_feasible_insertion:,}")
    print(f"Technician routes used: {route_count:,}")
    print(f"Active-task coverage: {summary['active_task_coverage']:.2%}")
    print(f"Total travel minutes including depot returns: {total_travel:,.2f}")
    print(f"Total scheduled service minutes: {service_minutes:,}")
    print("\nLocal outputs:")
    print(f"- {STATUS_PATH.relative_to(ROOT)}")
    print(f"- {ROUTES_PATH.relative_to(ROOT)}")
    print(f"- {SUMMARY_PATH.relative_to(ROOT)}")
    print("\nReminder: insertion is another heuristic, not a proof of optimality.")


if __name__ == "__main__":
    main()
