"""Add intra-route swap and 2-opt improvement to an existing validated schedule.

Run from the repository root:
    python src/run_route_neighborhood_search.py

Starts from the saved relocation-local-search routes, preserving the exact set
of assigned tasks. It explores two route-order neighborhoods:
- Swap two tasks within the same technician route.
- Reverse a contiguous route segment (2-opt-style reversal).

Every accepted move must reduce total travel and pass the same time-window,
shift, eligibility and depot-return feasibility checks as the baseline scheduler.
This is a bounded heuristic, not an exact optimizer.

Requires the existing local-search output files. It does not overwrite them.
Outputs:
    data/processed/neighborhood_search_routes.csv
    data/processed/neighborhood_search_task_status.csv
    data/processed/neighborhood_search_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_local_search_scheduler import (
    DATA,
    TASKS_PATH,
    TECHNICIANS_PATH,
    T2T_PATH,
    T2TECH_PATH,
    TECH2T_PATH,
    evaluate_route,
    parse_bool,
    parse_json_list,
    write_csv,
)

BASE_ROUTES_PATH = DATA / "local_search_routes.csv"
BASE_STATUS_PATH = DATA / "local_search_task_status.csv"
ROUTES_PATH = DATA / "neighborhood_search_routes.csv"
STATUS_PATH = DATA / "neighborhood_search_task_status.csv"
SUMMARY_PATH = DATA / "neighborhood_search_summary.json"

MAX_ACCEPTED_MOVES = 100
IMPROVEMENT_EPSILON = 1e-6

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


def main() -> None:
    required = [
        TASKS_PATH, TECHNICIANS_PATH, BASE_ROUTES_PATH, BASE_STATUS_PATH,
        T2T_PATH, T2TECH_PATH, TECH2T_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Required file(s) missing:")
        for path in missing:
            print(f"- {path.relative_to(DATA.parent.parent)}")
        print("Run and validate the relocation local-search scheduler first.")
        return

    tasks_frame = pd.read_csv(TASKS_PATH, keep_default_na=False)
    technicians_frame = pd.read_csv(TECHNICIANS_PATH, keep_default_na=False)
    base_routes_frame = pd.read_csv(BASE_ROUTES_PATH)
    status_frame = pd.read_csv(BASE_STATUS_PATH, keep_default_na=False)
    t2t = __import__("numpy").load(T2T_PATH, mmap_mode="r")
    t2tech = __import__("numpy").load(T2TECH_PATH, mmap_mode="r")
    tech2t = __import__("numpy").load(TECH2T_PATH, mmap_mode="r")
    import numpy as np

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
    for row in base_routes_frame.sort_values(
        ["technician_id", "route_sequence"]
    ).to_dict(orient="records"):
        tech_id, task_id = int(row["technician_id"]), int(row["task_id"])
        if tech_id not in routes or task_id not in task_by_id:
            raise ValueError(f"Unknown ID in saved route: technician={tech_id}, task={task_id}.")
        if tech_id not in eligible_by_task[task_id]:
            raise ValueError(f"Task {task_id} is assigned to ineligible technician {tech_id}.")
        routes[tech_id].append(task_id)

    base_assigned = [task_id for route in routes.values() for task_id in route]
    if len(base_assigned) != len(set(base_assigned)):
        raise ValueError("Baseline routes contain duplicate task assignments.")
    assigned_set = set(base_assigned)

    cached: dict[int, dict[str, Any]] = {}
    for tech_id, route in routes.items():
        start, end = tech_windows[tech_id]
        evaluation = evaluate_route(
            tech_id, route, task_by_id, start, end, t2t, t2tech, tech2t
        )
        if evaluation is None:
            raise RuntimeError(f"Baseline route for technician {tech_id} is infeasible.")
        cached[tech_id] = evaluation
    initial_travel = float(sum(item["travel"] for item in cached.values()))
    accepted_moves: list[dict[str, Any]] = []

    # Deterministic first-improvement search; only task order changes.
    for iteration in range(MAX_ACCEPTED_MOVES):
        improved = False
        for tech_id in sorted(routes):
            route = routes[tech_id]
            n = len(route)
            if n < 2:
                continue
            shift_start, shift_end = tech_windows[tech_id]
            old_eval = cached[tech_id]

            # Pairwise swap neighborhood.
            candidates = (
                ("swap", route[:i] + [route[j]] + route[i + 1:j] + [route[i]] + route[j + 1:])
                for i in range(n - 1)
                for j in range(i + 1, n)
            )
            for move_type, candidate in candidates:
                candidate_eval = evaluate_route(
                    tech_id, candidate, task_by_id, shift_start, shift_end,
                    t2t, t2tech, tech2t,
                )
                if candidate_eval is not None and candidate_eval["travel"] < old_eval["travel"] - IMPROVEMENT_EPSILON:
                    delta = candidate_eval["travel"] - old_eval["travel"]
                    routes[tech_id] = candidate
                    cached[tech_id] = candidate_eval
                    accepted_moves.append({
                        "move": len(accepted_moves) + 1,
                        "move_type": move_type,
                        "technician_id": tech_id,
                        "task_ids": [route[i], route[j]],
                        "travel_change_minutes": round(delta, 3),
                    })
                    improved = True
                    break
            if improved:
                break

            # 2-opt-style reversal of every contiguous segment.
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                    candidate_eval = evaluate_route(
                        tech_id, candidate, task_by_id, shift_start, shift_end,
                        t2t, t2tech, tech2t,
                    )
                    if candidate_eval is not None and candidate_eval["travel"] < old_eval["travel"] - IMPROVEMENT_EPSILON:
                        delta = candidate_eval["travel"] - old_eval["travel"]
                        routes[tech_id] = candidate
                        cached[tech_id] = candidate_eval
                        accepted_moves.append({
                            "move": len(accepted_moves) + 1,
                            "move_type": "2opt_reversal",
                            "technician_id": tech_id,
                            "task_ids": route[i:j + 1],
                            "travel_change_minutes": round(delta, 3),
                        })
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            print(f"No improving swap/2-opt move found after {len(accepted_moves)} accepted move(s).")
            break
        if (iteration + 1) % 10 == 0:
            print(
                f"Accepted moves: {len(accepted_moves)}; "
                f"travel: {sum(item['travel'] for item in cached.values()):,.2f} minutes"
            )

    # Re-evaluate every route and verify coverage/eligibility before writing.
    route_rows: list[dict[str, Any]] = []
    final_travel = 0.0
    for tech_id, route in routes.items():
        if not route:
            continue
        start, end = tech_windows[tech_id]
        evaluation = evaluate_route(
            tech_id, route, task_by_id, start, end, t2t, t2tech, tech2t
        )
        if evaluation is None:
            raise RuntimeError(f"Final route for technician {tech_id} failed revalidation.")
        final_travel += evaluation["travel"]
        route_rows.extend(evaluation["rows"])

    final_assigned = [int(row["task_id"]) for row in route_rows]
    if len(final_assigned) != len(set(final_assigned)) or set(final_assigned) != assigned_set:
        raise RuntimeError("Neighborhood search changed the assigned task set.")
    for row in route_rows:
        if int(row["technician_id"]) not in eligible_by_task[int(row["task_id"])]:
            raise RuntimeError("Neighborhood search produced an ineligible assignment.")

    status_by_id = {
        int(row["task_id"]): row for row in status_frame.to_dict(orient="records")
    }
    for task_id, row in status_by_id.items():
        if row["status"] == "assigned":
            row.update({
                "technician_id": "", "route_sequence": "", "service_start": "",
                "service_finish": "", "inbound_travel_minutes": "",
                "return_travel_minutes": "",
            })
    for row in route_rows:
        status_by_id[int(row["task_id"])].update({
            "status": "assigned", "reason": "",
            "technician_id": int(row["technician_id"]),
            "route_sequence": int(row["route_sequence"]),
            "service_start": row["service_start"],
            "service_finish": row["service_finish"],
            "inbound_travel_minutes": row["inbound_travel_minutes"],
            "return_travel_minutes": row["return_travel_minutes"],
        })

    route_rows.sort(key=lambda row: (int(row["technician_id"]), int(row["route_sequence"])))
    status_rows = [status_by_id[task_id] for task_id in sorted(status_by_id)]
    assigned_count = sum(row["status"] == "assigned" for row in status_rows)
    unassigned_count = sum(row["status"] == "unassigned" for row in status_rows)
    active_tasks = len(task_by_id) - cancelled_count
    if assigned_count != len(assigned_set):
        raise RuntimeError("Assigned task count differs from the baseline.")

    write_csv(ROUTES_PATH, ROUTE_FIELDS, route_rows)
    write_csv(STATUS_PATH, STATUS_FIELDS, status_rows)
    summary = {
        "baseline": "validated_relocation_local_search",
        "algorithm": "first_improvement_intra_route_swap_and_2opt_reversal",
        "note": "Bounded route-order neighborhood search; preserves assigned tasks and accepts feasible travel-reducing moves only.",
        "total_task_records": len(task_by_id),
        "cancelled_excluded": cancelled_count,
        "active_tasks": active_tasks,
        "assigned_tasks": assigned_count,
        "unassigned_tasks": unassigned_count,
        "technician_routes_used": sum(bool(route) for route in routes.values()),
        "active_task_coverage": round(assigned_count / active_tasks, 6) if active_tasks else 0.0,
        "initial_travel_minutes": round(initial_travel, 3),
        "final_travel_minutes": round(final_travel, 3),
        "travel_reduction_minutes": round(initial_travel - final_travel, 3),
        "travel_reduction_percent": round((initial_travel - final_travel) / initial_travel * 100, 4) if initial_travel else 0.0,
        "accepted_moves": len(accepted_moves),
        "move_log": accepted_moves,
        "max_accepted_moves": MAX_ACCEPTED_MOVES,
        "assigned_task_set_preserved": set(final_assigned) == assigned_set,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nROUTE NEIGHBORHOOD SEARCH SUMMARY")
    print("=================================")
    print(f"Initial travel: {initial_travel:,.2f} minutes")
    print(f"Final travel:   {final_travel:,.2f} minutes")
    print(f"Travel reduction: {initial_travel - final_travel:,.2f} minutes ({summary['travel_reduction_percent']:.2f}%)")
    print(f"Accepted moves: {len(accepted_moves)}")
    print(f"Assigned tasks preserved: {summary['assigned_task_set_preserved']}")
    print(f"Saved {ROUTES_PATH.relative_to(DATA.parent.parent)}")
    print(f"Saved {STATUS_PATH.relative_to(DATA.parent.parent)}")
    print(f"Saved {SUMMARY_PATH.relative_to(DATA.parent.parent)}")


if __name__ == "__main__":
    main()
