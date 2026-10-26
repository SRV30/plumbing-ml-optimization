"""Improve the validated route-insertion schedule with relocation local search.

Run from the repository root:
    python src/run_local_search_scheduler.py

Starts from insertion_routes.csv and attempts to relocate assigned tasks between
already-used eligible technician-day routes when the move is feasible and reduces
total travel time. Task coverage is held constant. This is a bounded heuristic,
not an exact or globally optimal optimizer.

Outputs are local and Git-ignored:
    data/processed/local_search_task_status.csv
    data/processed/local_search_routes.csv
    data/processed/local_search_summary.json
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
INSERTION_ROUTES_PATH = DATA / "insertion_routes.csv"
INSERTION_STATUS_PATH = DATA / "insertion_task_status.csv"
INSERTION_SUMMARY_PATH = DATA / "insertion_summary.json"
T2T_PATH = DATA / "task_to_task.npy"
T2TECH_PATH = DATA / "task_to_technician.npy"
TECH2T_PATH = DATA / "technician_to_task.npy"

STATUS_PATH = DATA / "local_search_task_status.csv"
ROUTES_PATH = DATA / "local_search_routes.csv"
SUMMARY_PATH = DATA / "local_search_summary.json"


MAX_ACCEPTED_MOVES = 100
IMPROVEMENT_EPSILON = 1e-6

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


def parse_json_list(value: Any, row_id: int) -> list[int]:
    try:
        result = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Task {row_id}: invalid eligible-technician list: {error}") from error
    if not isinstance(result, list) or not all(isinstance(item, int) for item in result):
        raise ValueError(f"Task {row_id}: eligible technicians must be a JSON integer list.")
    return result


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_route(
    technician_id: int,
    route_ids: list[int],
    tasks: dict[int, dict[str, Any]],
    shift_start: pd.Timestamp,
    shift_end: pd.Timestamp,
    t2t: np.ndarray,
    t2tech: np.ndarray,
    tech2t: np.ndarray,
) -> dict[str, Any] | None:
    """Return a feasible route and total travel including final depot return."""
    if not route_ids:
        return {"rows": [], "travel": 0.0}

    previous_task_id: int | None = None
    previous_finish = shift_start
    travel_total = 0.0
    rows: list[dict[str, Any]] = []

    for sequence, task_id in enumerate(route_ids, start=1):
        task = tasks[task_id]
        inbound = (
            float(tech2t[technician_id - 1, task_id - 1])
            if previous_task_id is None
            else float(t2t[previous_task_id - 1, task_id - 1])
        )
        if not np.isfinite(inbound) or inbound < 0:
            return None

        arrival = previous_finish + pd.Timedelta(minutes=inbound)
        service_start = max(task["start"], arrival)
        service_finish = service_start + pd.Timedelta(minutes=task["duration"])

        if service_finish > task["end"]:
            return None
        if service_start < shift_start or service_finish > shift_end:
            return None

        rows.append({
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
        travel_total += inbound
        previous_task_id = task_id
        previous_finish = service_finish

    assert previous_task_id is not None
    return_travel = float(t2tech[previous_task_id - 1, technician_id - 1])
    if not np.isfinite(return_travel) or return_travel < 0:
        return None
    if previous_finish + pd.Timedelta(minutes=return_travel) > shift_end:
        return None

    rows[-1]["return_travel_minutes"] = round(return_travel, 3)
    travel_total += return_travel
    return {"rows": rows, "travel": travel_total}


def main() -> None:
    required = [
        TASKS_PATH, TECHNICIANS_PATH, INSERTION_ROUTES_PATH,
        INSERTION_STATUS_PATH, INSERTION_SUMMARY_PATH,
        T2T_PATH, T2TECH_PATH, TECH2T_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Required file(s) missing:")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        print("Run and validate the insertion scheduler first.")
        return

    tasks_frame = pd.read_csv(TASKS_PATH, keep_default_na=False)
    technicians_frame = pd.read_csv(TECHNICIANS_PATH, keep_default_na=False)
    insertion_routes = pd.read_csv(INSERTION_ROUTES_PATH)
    insertion_status = pd.read_csv(INSERTION_STATUS_PATH, keep_default_na=False)
    insertion_summary = json.loads(INSERTION_SUMMARY_PATH.read_text(encoding="utf-8"))

    t2t = np.load(T2T_PATH, mmap_mode="r")
    t2tech = np.load(T2TECH_PATH, mmap_mode="r")
    tech2t = np.load(TECH2T_PATH, mmap_mode="r")

    task_by_id: dict[int, dict[str, Any]] = {}
    eligible_by_task: dict[int, list[int]] = {}
    cancelled_count = 0

    for row in tasks_frame.to_dict(orient="records"):
        task_id = int(row["task_id"])
        cancelled = parse_bool(row["cancelled"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        duration = int(row["duration"])
        eligible = parse_json_list(row["relevant_technician_ids_json"], task_id)
        if pd.isna(start) or pd.isna(end) or end < start or duration <= 0:
            raise ValueError(f"Task {task_id}: invalid scheduling data.")
        task_by_id[task_id] = {
            "cancelled": cancelled, "start": start, "end": end,
            "duration": duration, "priority": int(row["priority"]),
            "penalty": float(row["penalty"]), "eligible": eligible,
        }
        eligible_by_task[task_id] = eligible
        cancelled_count += int(cancelled)

    tech_windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for row in technicians_frame.to_dict(orient="records"):
        tech_id = int(row["technician_id"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError(f"Technician {tech_id}: invalid working time window.")
        tech_windows[tech_id] = (start, end)

    routes: dict[int, list[int]] = {tech_id: [] for tech_id in tech_windows}
    for row in insertion_routes.sort_values(["technician_id", "route_sequence"]).to_dict(orient="records"):
        tech_id, task_id = int(row["technician_id"]), int(row["task_id"])
        if tech_id not in routes or task_id not in task_by_id:
            raise ValueError(f"Unknown ID in insertion route: technician={tech_id}, task={task_id}.")
        if tech_id not in eligible_by_task[task_id]:
            raise ValueError(f"Task {task_id} is assigned to ineligible technician {tech_id} in insertion output.")
        routes[tech_id].append(task_id)

    base_assigned = {task_id for route in routes.values() for task_id in route}
    if len(base_assigned) != len(insertion_routes):
        raise ValueError("Insertion route table repeats a task.")

    cached: dict[int, dict[str, Any]] = {}
    for tech_id, route in routes.items():
        start, end = tech_windows[tech_id]
        evaluation = evaluate_route(tech_id, route, task_by_id, start, end, t2t, t2tech, tech2t)
        if evaluation is None:
            raise RuntimeError(f"Initial insertion route for technician {tech_id} is infeasible.")
        cached[tech_id] = evaluation

    initial_travel = sum(entry["travel"] for entry in cached.values())
    accepted_moves: list[dict[str, Any]] = []

    
    
    for move_index in range(MAX_ACCEPTED_MOVES):
        improved = False
        active_techs = [tech_id for tech_id, route in routes.items() if route]

        for source_tech in active_techs:
            source_route = routes[source_tech]
            if not source_route:
                continue
            source_start, source_end = tech_windows[source_tech]

            for task_position, task_id in enumerate(list(source_route)):
                eligible_targets = eligible_by_task[task_id]
                new_source_route = source_route[:task_position] + source_route[task_position + 1:]
                new_source_eval = evaluate_route(
                    source_tech, new_source_route, task_by_id, source_start, source_end,
                    t2t, t2tech, tech2t,
                )
                if new_source_eval is None:
                    continue

                for target_tech in eligible_targets:
                    if target_tech == source_tech or target_tech not in routes:
                        continue
                    target_route = routes[target_tech]
                    
                    if not target_route:
                        continue
                    target_start, target_end = tech_windows[target_tech]
                    target_old_eval = cached[target_tech]

                    for position in range(len(target_route) + 1):
                        new_target_route = target_route[:position] + [task_id] + target_route[position:]
                        new_target_eval = evaluate_route(
                            target_tech, new_target_route, task_by_id, target_start, target_end,
                            t2t, t2tech, tech2t,
                        )
                        if new_target_eval is None:
                            continue

                        old_cost = cached[source_tech]["travel"] + target_old_eval["travel"]
                        new_cost = new_source_eval["travel"] + new_target_eval["travel"]
                        delta = new_cost - old_cost
                        if delta < -IMPROVEMENT_EPSILON:
                            routes[source_tech] = new_source_route
                            routes[target_tech] = new_target_route
                            cached[source_tech] = new_source_eval
                            cached[target_tech] = new_target_eval
                            accepted_moves.append({
                                "move": len(accepted_moves) + 1,
                                "task_id": task_id,
                                "from_technician_id": source_tech,
                                "to_technician_id": target_tech,
                                "travel_change_minutes": round(delta, 3),
                            })
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            print(f"No improving relocation found after {len(accepted_moves)} accepted move(s).")
            break
        if (move_index + 1) % 10 == 0:
            total_now = sum(entry["travel"] for entry in cached.values())
            print(f"Accepted moves: {len(accepted_moves)}; current travel: {total_now:,.2f} minutes")

    
    status_by_id = {
        int(row["task_id"]): row
        for row in insertion_status.to_dict(orient="records")
    }
    route_rows: list[dict[str, Any]] = []
    final_travel = 0.0
    route_count = 0

    for task_id, row in status_by_id.items():
        if row["status"] == "assigned":
            row.update({
                "technician_id": "", "route_sequence": "", "service_start": "",
                "service_finish": "", "inbound_travel_minutes": "",
                "return_travel_minutes": "",
            })

    for tech_id, route in routes.items():
        if not route:
            continue
        start, end = tech_windows[tech_id]
        evaluation = evaluate_route(tech_id, route, task_by_id, start, end, t2t, t2tech, tech2t)
        if evaluation is None:
            raise RuntimeError(f"Final route for technician {tech_id} failed revalidation.")
        route_count += 1
        final_travel += evaluation["travel"]

        for row in evaluation["rows"]:
            route_rows.append(row)
            status = status_by_id[row["task_id"]]
            status.update({
                "status": "assigned",
                "reason": "",
                "technician_id": tech_id,
                "route_sequence": row["route_sequence"],
                "service_start": row["service_start"],
                "service_finish": row["service_finish"],
                "inbound_travel_minutes": row["inbound_travel_minutes"],
                "return_travel_minutes": row["return_travel_minutes"],
            })

    route_rows.sort(key=lambda row: (row["technician_id"], row["route_sequence"]))
    status_rows = [status_by_id[task_id] for task_id in sorted(status_by_id)]
    assigned_tasks = sum(row["status"] == "assigned" for row in status_rows)
    unassigned_tasks = sum(row["status"] == "unassigned" for row in status_rows)
    active_tasks = len(task_by_id) - cancelled_count

    if {int(row["task_id"]) for row in route_rows} != base_assigned:
        raise RuntimeError("Local search changed the set of assigned tasks; this experiment should preserve coverage.")
    if assigned_tasks != int(insertion_summary["assigned_tasks"]):
        raise RuntimeError("Local search changed assigned task count unexpectedly.")

    DATA.mkdir(parents=True, exist_ok=True)
    write_csv(STATUS_PATH, STATUS_FIELDS, status_rows)
    write_csv(ROUTES_PATH, ROUTE_FIELDS, route_rows)

    summary = {
        "baseline": "route_insertion_plus_relocation_local_search",
        "note": "Relocation local search starting from validated insertion schedule; not an exact solver.",
        "total_task_records": len(task_by_id),
        "cancelled_excluded": cancelled_count,
        "active_tasks": active_tasks,
        "assigned_tasks": assigned_tasks,
        "unassigned_tasks": unassigned_tasks,
        "technician_routes_used": route_count,
        "active_task_coverage": round(assigned_tasks / active_tasks, 6) if active_tasks else 0.0,
        "total_travel_minutes_including_depot_returns": round(final_travel, 3),
        "total_service_minutes": int(sum(
            task_by_id[task_id]["duration"] for task_id in base_assigned
        )),
        "initial_insertion_travel_minutes": round(initial_travel, 3),
        "travel_reduction_minutes": round(initial_travel - final_travel, 3),
        "travel_reduction_percent": round(
            (initial_travel - final_travel) / initial_travel * 100, 4
        ) if initial_travel else 0.0,
        "accepted_relocation_moves": len(accepted_moves),
        "move_log": accepted_moves,
        "max_accepted_moves": MAX_ACCEPTED_MOVES,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nLOCAL-SEARCH SCHEDULER SUMMARY")
    print("==============================")
    print(f"Starting insertion travel: {initial_travel:,.2f} minutes")
    print(f"Final local-search travel: {final_travel:,.2f} minutes")
    print(f"Travel reduction: {initial_travel - final_travel:,.2f} minutes ({summary['travel_reduction_percent']:.2f}%)")
    print(f"Accepted relocation moves: {len(accepted_moves)}")
    print(f"Tasks assigned: {assigned_tasks:,}")
    print(f"Tasks unassigned: {unassigned_tasks:,}")
    print(f"Active-task coverage: {summary['active_task_coverage']:.2%}")
    print(f"Technician routes used: {route_count:,}")
    print("\nSaved:")
    print(f"- {STATUS_PATH.relative_to(ROOT)}")
    print(f"- {ROUTES_PATH.relative_to(ROOT)}")
    print(f"- {SUMMARY_PATH.relative_to(ROOT)}")
    print("\nThis relocation heuristic preserves assignments; it is not a proof of global optimality.")


if __name__ == "__main__":
    main()
