"""Improve technician routes using intra-route swaps and 2-opt.

Run from the repository root:
    python src/improve_local_search_moves.py

Starts from the saved local-search route schedule when available, otherwise
from the validated insertion schedule. Only changes route order within each
technician-day route. Every candidate is re-evaluated against task windows,
shift bounds, travel matrices, and return-to-depot feasibility.

Outputs (local/Git-ignored):
    data/processed/enhanced_local_search_routes.csv
    data/processed/enhanced_local_search_task_status.csv
    data/processed/enhanced_local_search_summary.json

This is a bounded heuristic, not an exact optimizer. It preserves the assigned
task set and accepts a move only when it reduces total travel.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_local_search_scheduler import evaluate_route, parse_bool, parse_json_list as parse_eligible_list

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

TASKS_PATH = DATA / "tasks.csv"
TECHNICIANS_PATH = DATA / "technicians.csv"
INSERTION_ROUTES_PATH = DATA / "insertion_routes.csv"
INSERTION_STATUS_PATH = DATA / "insertion_task_status.csv"
INSERTION_SUMMARY_PATH = DATA / "insertion_summary.json"
LOCAL_ROUTES_PATH = DATA / "local_search_routes.csv"
LOCAL_STATUS_PATH = DATA / "local_search_task_status.csv"
LOCAL_SUMMARY_PATH = DATA / "local_search_summary.json"

T2T_PATH = DATA / "task_to_task.npy"
T2TECH_PATH = DATA / "task_to_technician.npy"
TECH2T_PATH = DATA / "technician_to_task.npy"

OUT_ROUTES = DATA / "enhanced_local_search_routes.csv"
OUT_STATUS = DATA / "enhanced_local_search_task_status.csv"
OUT_SUMMARY = DATA / "enhanced_local_search_summary.json"

MAX_ACCEPTED_MOVES = 30
EPSILON = 1e-6

ROUTE_FIELDS = [
    "technician_id", "route_sequence", "task_id", "service_start",
    "service_finish", "inbound_travel_minutes", "return_travel_minutes",
    "duration_minutes", "priority", "penalty",
]
STATUS_FIELDS = [
    "task_id", "cancelled", "status", "reason", "technician_id",
    "route_sequence", "service_start", "service_finish",
    "inbound_travel_minutes", "return_travel_minutes",
    "duration_minutes", "priority", "penalty",
]


def parse_json_list(value: Any, field: str, task_id: int) -> list:
    """Parse a JSON list; fields can contain skill labels or technician IDs."""
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Task {task_id}: invalid {field}: {error}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"Task {task_id}: {field} must be a JSON list.")
    if field == "relevant_technician_ids_json" and not all(isinstance(item, int) for item in parsed):
        raise ValueError(f"Task {task_id}: eligible technician IDs must be integers.")
    return parsed


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    required = [
        TASKS_PATH, TECHNICIANS_PATH, INSERTION_ROUTES_PATH,
        INSERTION_STATUS_PATH, INSERTION_SUMMARY_PATH,
        T2T_PATH, T2TECH_PATH, TECH2T_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Required files missing:")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        print("Run the data preparation and validated insertion scheduler first.")
        return

    # Prefer the existing relocation-local-search result as the starting point.
    if LOCAL_ROUTES_PATH.exists() and LOCAL_STATUS_PATH.exists() and LOCAL_SUMMARY_PATH.exists():
        routes_path, status_path, summary_path = LOCAL_ROUTES_PATH, LOCAL_STATUS_PATH, LOCAL_SUMMARY_PATH
        starting_method = "existing_relocation_local_search"
    else:
        routes_path, status_path, summary_path = INSERTION_ROUTES_PATH, INSERTION_STATUS_PATH, INSERTION_SUMMARY_PATH
        starting_method = "route_insertion"

    task_frame = pd.read_csv(TASKS_PATH, keep_default_na=False)
    tech_frame = pd.read_csv(TECHNICIANS_PATH, keep_default_na=False)
    route_frame = pd.read_csv(routes_path)
    status_frame = pd.read_csv(status_path, keep_default_na=False)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    t2t = np.load(T2T_PATH, mmap_mode="r")
    t2tech = np.load(T2TECH_PATH, mmap_mode="r")
    tech2t = np.load(TECH2T_PATH, mmap_mode="r")

    tasks: dict[int, dict[str, Any]] = {}
    eligible: dict[int, list[int]] = {}
    cancelled_count = 0
    for row in task_frame.to_dict(orient="records"):
        task_id = int(row["task_id"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        duration = int(row["duration"])
        if pd.isna(start) or pd.isna(end) or end < start or duration <= 0:
            raise ValueError(f"Task {task_id}: invalid time window or duration.")
        cancelled = parse_bool(row["cancelled"])
        eligible_ids = parse_json_list(row["relevant_technician_ids_json"], task_id)
        tasks[task_id] = {
            "cancelled": cancelled, "start": start, "end": end,
            "duration": duration, "priority": int(row["priority"]),
            "penalty": float(row["penalty"]), "eligible": eligible_ids,
        }
        eligible[task_id] = eligible_ids
        cancelled_count += int(cancelled)

    shifts: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for row in tech_frame.to_dict(orient="records"):
        tech_id = int(row["technician_id"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError(f"Technician {tech_id}: invalid shift.")
        shifts[tech_id] = (start, end)

    routes: dict[int, list[int]] = {tech_id: [] for tech_id in shifts}
    for row in route_frame.sort_values(["technician_id", "route_sequence"]).to_dict(orient="records"):
        tech_id, task_id = int(row["technician_id"]), int(row["task_id"])
        if tech_id not in shifts or task_id not in tasks:
            raise ValueError(f"Unknown route ID: technician={tech_id}, task={task_id}.")
        if tech_id not in eligible[task_id]:
            raise ValueError(f"Task {task_id} is assigned to ineligible technician {tech_id}.")
        routes[tech_id].append(task_id)

    assigned_before = {task_id for route in routes.values() for task_id in route}
    if len(assigned_before) != len(route_frame):
        raise ValueError("Starting route table repeats a task.")

    def evaluate(tech_id: int, route: list[int]) -> dict[str, Any] | None:
        start, end = shifts[tech_id]
        return evaluate_route(tech_id, route, tasks, start, end, t2t, t2tech, tech2t)

    cached: dict[int, dict[str, Any]] = {}
    for tech_id, route in routes.items():
        result = evaluate(tech_id, route)
        if result is None:
            raise RuntimeError(f"Starting route for technician {tech_id} is infeasible.")
        cached[tech_id] = result

    initial_travel = sum(item["travel"] for item in cached.values())
    move_log: list[dict[str, Any]] = []

    # First-improvement descent: accept one feasible improvement, then restart.
    for _ in range(MAX_ACCEPTED_MOVES):
        accepted = False
        for tech_id in sorted(routes):
            route = routes[tech_id]
            n = len(route)
            if n < 2:
                continue
            old_eval = cached[tech_id]
            old_cost = old_eval["travel"]

            # Swap two tasks in the same route.
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = route.copy()
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    result = evaluate(tech_id, candidate)
                    if result is not None and result["travel"] < old_cost - EPSILON:
                        routes[tech_id] = candidate
                        cached[tech_id] = result
                        move_log.append({
                            "move": len(move_log) + 1,
                            "operator": "intra_route_swap",
                            "technician_id": tech_id,
                            "position_a": i + 1,
                            "position_b": j + 1,
                            "travel_change_minutes": round(result["travel"] - old_cost, 3),
                        })
                        accepted = True
                        break
                if accepted:
                    break
            if accepted:
                break

            # 2-opt: reverse every contiguous subsequence of length >= 2.
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                    result = evaluate(tech_id, candidate)
                    if result is not None and result["travel"] < old_cost - EPSILON:
                        routes[tech_id] = candidate
                        cached[tech_id] = result
                        move_log.append({
                            "move": len(move_log) + 1,
                            "operator": "two_opt",
                            "technician_id": tech_id,
                            "segment_start": i + 1,
                            "segment_end": j + 1,
                            "travel_change_minutes": round(result["travel"] - old_cost, 3),
                        })
                        accepted = True
                        break
                if accepted:
                    break
            if accepted:
                break

        if not accepted:
            print(f"No improving swap/2-opt move found after {len(move_log)} accepted move(s).")
            break
        if len(move_log) % 5 == 0:
            current = sum(item["travel"] for item in cached.values())
            print(f"Accepted {len(move_log)} moves; current travel {current:,.2f} minutes.")

    final_rows: list[dict[str, Any]] = []
    final_travel = 0.0
    used_routes = 0
    for tech_id, route in routes.items():
        if not route:
            continue
        result = evaluate(tech_id, route)
        if result is None:
            raise RuntimeError(f"Final route for technician {tech_id} failed validation.")
        used_routes += 1
        final_travel += result["travel"]
        final_rows.extend(result["rows"])

    final_assigned = {int(row["task_id"]) for row in final_rows}
    if final_assigned != assigned_before:
        raise RuntimeError("Enhanced search changed the assigned task set.")
    if final_travel > initial_travel + EPSILON:
        raise RuntimeError("Enhanced search increased total travel; refusing to save.")

    status_by_id = {int(row["task_id"]): row for row in status_frame.to_dict(orient="records")}
    for row in status_by_id.values():
        if row["status"] == "assigned":
            row.update({
                "technician_id": "", "route_sequence": "", "service_start": "",
                "service_finish": "", "inbound_travel_minutes": "", "return_travel_minutes": "",
            })
    for route_row in final_rows:
        status = status_by_id[int(route_row["task_id"])]
        status.update({
            "status": "assigned", "reason": "",
            "technician_id": route_row["technician_id"],
            "route_sequence": route_row["route_sequence"],
            "service_start": route_row["service_start"],
            "service_finish": route_row["service_finish"],
            "inbound_travel_minutes": route_row["inbound_travel_minutes"],
            "return_travel_minutes": route_row["return_travel_minutes"],
        })

    final_rows.sort(key=lambda row: (row["technician_id"], row["route_sequence"]))
    status_rows = [status_by_id[task_id] for task_id in sorted(status_by_id)]
    active_tasks = len(tasks) - cancelled_count
    assigned_count = len(final_assigned)
    summary = {
        "method": "route_order_swap_and_2opt_local_search",
        "starting_method": starting_method,
        "note": "Bounded first-improvement heuristic; only feasible moves that reduce travel are accepted.",
        "total_task_records": len(tasks),
        "cancelled_excluded": cancelled_count,
        "active_tasks": active_tasks,
        "assigned_tasks": assigned_count,
        "unassigned_tasks": sum(row["status"] == "unassigned" for row in status_rows),
        "active_task_coverage": round(assigned_count / active_tasks, 6) if active_tasks else 0.0,
        "technician_routes_used": used_routes,
        "total_travel_minutes_including_depot_returns": round(final_travel, 3),
        "total_service_minutes": sum(int(row["duration_minutes"]) for row in final_rows),
        "initial_travel_minutes_including_depot_returns": round(initial_travel, 3),
        "final_travel_minutes_including_depot_returns": round(final_travel, 3),
        "travel_reduction_minutes": round(initial_travel - final_travel, 3),
        "travel_reduction_percent": round((initial_travel - final_travel) / initial_travel * 100, 4) if initial_travel else 0.0,
        "accepted_moves": len(move_log),
        "move_log": move_log,
        "max_accepted_moves": MAX_ACCEPTED_MOVES,
        "assignment_set_preserved": final_assigned == assigned_before,
        "schedule_feasibility_revalidated": True,
    }

    write_csv(OUT_ROUTES, ROUTE_FIELDS, final_rows)
    write_csv(OUT_STATUS, STATUS_FIELDS, status_rows)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nENHANCED LOCAL SEARCH SUMMARY")
    print("=============================")
    print(f"Starting method: {starting_method}")
    print(f"Starting travel: {initial_travel:,.2f} minutes")
    print(f"Final travel:    {final_travel:,.2f} minutes")
    print(f"Reduction:       {initial_travel - final_travel:,.2f} minutes ({summary['travel_reduction_percent']:.2f}%)")
    print(f"Accepted moves:  {len(move_log)}")
    print(f"Assigned tasks:  {assigned_count:,} (assignment set preserved)")
    print("Every final route was re-evaluated for time-window, shift, and depot-return feasibility.")
    print("Do not report an improvement until the script has been run and the output independently checked.")


if __name__ == "__main__":
    main()
