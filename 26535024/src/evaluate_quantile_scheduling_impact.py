"""Compare Q50 and Q80 duration estimates in a counterfactual scheduling experiment.

Run from the repository root:
    python 26535024/src/evaluate_quantile_scheduling_impact.py

Only chronological hold-out tasks use predictions; earlier tasks retain source
durations. The resulting routes are replayed with source-recorded durations to
count feasibility violations. This is a research counterfactual on telecom
field-service data, not evidence of real plumbing-business outcomes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_duration_scheduling_impact import (
    evaluate_predicted_route,
    parse_bool,
    read_json_list,
    replay_with_actual_durations,
    write_csv,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TASKS_PATH = DATA / "tasks.csv"
TECHNICIANS_PATH = DATA / "technicians.csv"
PREDICTIONS_PATH = RESULTS / "duration_quantile_predictions.csv"
BASELINE_SUMMARY_PATH = DATA / "insertion_summary.json"
T2T_PATH = DATA / "task_to_task.npy"
T2TECH_PATH = DATA / "task_to_technician.npy"
TECH2T_PATH = DATA / "technician_to_task.npy"
SUMMARY_PATH = RESULTS / "quantile_scheduling_impact.json"
COMPARISON_PATH = RESULTS / "quantile_scheduling_impact_comparison.csv"
STATUS_FIELDS = [
    "task_id", "cancelled", "status", "reason", "technician_id",
    "route_sequence", "service_start_predicted", "service_finish_predicted",
    "estimated_duration_minutes", "actual_duration_minutes",
    "inbound_travel_minutes", "return_travel_minutes", "priority", "penalty",
]
ROUTE_FIELDS = [
    "technician_id", "route_sequence", "task_id", "service_start_predicted",
    "service_finish_predicted", "estimated_duration_minutes",
    "actual_duration_minutes", "inbound_travel_minutes",
    "return_travel_minutes", "priority", "penalty",
]


def schedule_scenario(
    name: str,
    prediction_column: str,
    tasks: dict[int, dict[str, Any]],
    tech_windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]],
    t2t: np.ndarray,
    t2tech: np.ndarray,
    tech2t: np.ndarray,
) -> dict[str, Any]:
    """Run deterministic route insertion for one duration-estimate scenario."""
    for task_id, task in tasks.items():
        task["scheduled_duration"] = task["scenario_durations"].get(
            name, task["actual_duration"]
        )

    technician_routes = {tech_id: [] for tech_id in tech_windows}
    status_by_id: dict[int, dict[str, Any]] = {}
    for task_id, task in tasks.items():
        common = {
            "task_id": task_id,
            "cancelled": task["cancelled"],
            "technician_id": "",
            "route_sequence": "",
            "service_start_predicted": "",
            "service_finish_predicted": "",
            "estimated_duration_minutes": task["scheduled_duration"],
            "actual_duration_minutes": task["actual_duration"],
            "inbound_travel_minutes": "",
            "return_travel_minutes": "",
            "priority": task["priority"],
            "penalty": task["penalty"],
        }
        if task["cancelled"]:
            status_by_id[task_id] = {
                **common, "status": "excluded_cancelled",
                "reason": "Source record is marked cancelled.",
            }
        elif not task["candidates"]:
            status_by_id[task_id] = {
                **common, "status": "unassigned",
                "reason": "Source record contains no eligible technicians.",
            }

    active = [
        task for task in tasks.values()
        if not task["cancelled"] and task["candidates"]
    ]
    active.sort(key=lambda task: (
        task["priority"], -task["penalty"], task["end"], task["start"], task["task_id"]
    ))
    infeasible_count = 0
    for index, task in enumerate(active, start=1):
        if index % 500 == 0:
            print(f"{name}: scheduling {index:,}/{len(active):,} tasks")
        best = None
        for tech_id in task["candidates"]:
            if tech_id not in technician_routes:
                continue
            shift_start, shift_end = tech_windows[tech_id]
            route = technician_routes[tech_id]
            old = evaluate_predicted_route(
                tech_id, route, tasks, shift_start, shift_end, t2t, t2tech, tech2t
            )
            if old is None:
                raise RuntimeError(f"{name}: existing route for technician {tech_id} is infeasible.")
            for pos in range(len(route) + 1):
                proposed = route[:pos] + [task["task_id"]] + route[pos:]
                result = evaluate_predicted_route(
                    tech_id, proposed, tasks, shift_start, shift_end, t2t, t2tech, tech2t
                )
                if result is None:
                    continue
                score = (result["travel"] - old["travel"], len(route), tech_id, pos)
                if best is None or score < best["score"]:
                    best = {
                        "score": score, "technician_id": tech_id,
                        "route": proposed, "evaluation": result,
                    }
        if best is None:
            infeasible_count += 1
            status_by_id[task["task_id"]] = {
                "task_id": task["task_id"], "cancelled": False,
                "status": "unassigned",
                "reason": "No feasible insertion using estimated duration.",
                "technician_id": "", "route_sequence": "",
                "service_start_predicted": "", "service_finish_predicted": "",
                "estimated_duration_minutes": task["scheduled_duration"],
                "actual_duration_minutes": task["actual_duration"],
                "inbound_travel_minutes": "", "return_travel_minutes": "",
                "priority": task["priority"], "penalty": task["penalty"],
            }
        else:
            tech_id = best["technician_id"]
            technician_routes[tech_id] = best["route"]
            status_by_id[task["task_id"]] = {
                "task_id": task["task_id"], "cancelled": False,
                "status": "assigned", "reason": "", "technician_id": tech_id,
                "route_sequence": "", "service_start_predicted": "",
                "service_finish_predicted": "",
                "estimated_duration_minutes": task["scheduled_duration"],
                "actual_duration_minutes": task["actual_duration"],
                "inbound_travel_minutes": "", "return_travel_minutes": "",
                "priority": task["priority"], "penalty": task["penalty"],
            }

    route_rows = []
    predicted_travel = 0.0
    replay_violation_routes = 0
    replay_violations = []
    for tech_id, route in technician_routes.items():
        if not route:
            continue
        shift_start, shift_end = tech_windows[tech_id]
        result = evaluate_predicted_route(
            tech_id, route, tasks, shift_start, shift_end, t2t, t2tech, tech2t
        )
        if result is None:
            raise RuntimeError(f"{name}: final estimated-duration route is infeasible.")
        predicted_travel += result["travel"]
        for row in result["rows"]:
            route_rows.append(row)
            status = status_by_id[row["task_id"]]
            status["route_sequence"] = row["route_sequence"]
            status["service_start_predicted"] = row["service_start_predicted"]
            status["service_finish_predicted"] = row["service_finish_predicted"]
            status["inbound_travel_minutes"] = row["inbound_travel_minutes"]
            status["return_travel_minutes"] = row["return_travel_minutes"]
        replay = replay_with_actual_durations(
            tech_id, route, tasks, shift_start, shift_end, t2t, t2tech, tech2t
        )
        if replay["violations"]:
            replay_violation_routes += 1
            replay_violations.extend(replay["violations"])

    route_rows.sort(key=lambda row: (row["technician_id"], row["route_sequence"]))
    status_rows = [status_by_id[task_id] for task_id in sorted(status_by_id)]
    status_path = DATA / f"quantile_{name.lower()}_task_status.csv"
    routes_path = DATA / f"quantile_{name.lower()}_routes.csv"
    write_csv(status_path, STATUS_FIELDS, status_rows)
    write_csv(routes_path, ROUTE_FIELDS, route_rows)

    assigned = sum(row["status"] == "assigned" for row in status_rows)
    cancelled = sum(row["status"] == "excluded_cancelled" for row in status_rows)
    active_count = len(tasks) - cancelled
    task_violation_ids = {
        item["task_id"] for item in replay_violations
        if any(kind in {"task_window", "technician_shift"} for kind in item["violation_types"])
    }
    return {
        "scenario": name,
        "prediction_column": prediction_column,
        "predicted_holdout_tasks": sum(
            name in task["scenario_durations"] for task in tasks.values()
        ),
        "assigned_tasks": assigned,
        "unassigned_tasks": sum(row["status"] == "unassigned" for row in status_rows),
        "active_task_coverage": round(assigned / active_count, 6) if active_count else 0.0,
        "routes_used": sum(bool(route) for route in technician_routes.values()),
        "planned_travel_minutes": round(predicted_travel, 3),
        "no_feasible_insertion_count": infeasible_count,
        "routes_with_replay_violations": replay_violation_routes,
        "unique_tasks_with_window_or_shift_violations": len(task_violation_ids),
        "return_after_shift_violations": sum(
            "return_after_shift" in item["violation_types"] for item in replay_violations
        ),
        "route_status_file": str(status_path.relative_to(ROOT)),
        "route_file": str(routes_path.relative_to(ROOT)),
    }


def main() -> None:
    required = [
        TASKS_PATH, TECHNICIANS_PATH, PREDICTIONS_PATH, BASELINE_SUMMARY_PATH,
        T2T_PATH, T2TECH_PATH, TECH2T_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing: {path.relative_to(ROOT)}")
        print("Run the duration quantile model and route-insertion scheduler first.")
        return

    task_frame = pd.read_csv(TASKS_PATH, keep_default_na=False)
    tech_frame = pd.read_csv(TECHNICIANS_PATH, keep_default_na=False)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    baseline = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if predictions["task_id"].duplicated().any():
        raise ValueError("Quantile predictions contain duplicate task IDs.")
    pred_by_id = {int(row["task_id"]): row for row in predictions.to_dict("records")}

    t2t = np.load(T2T_PATH, mmap_mode="r")
    t2tech = np.load(T2TECH_PATH, mmap_mode="r")
    tech2t = np.load(TECH2T_PATH, mmap_mode="r")
    tasks: dict[int, dict[str, Any]] = {}
    for row in task_frame.to_dict("records"):
        task_id = int(row["task_id"])
        actual = int(row["duration"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        candidates = read_json_list(
            row["relevant_technician_ids_json"], "relevant_technician_ids_json", task_id
        )
        if pd.isna(start) or pd.isna(end) or end < start or actual <= 0:
            raise ValueError(f"Task {task_id}: invalid time window or source duration.")
        row_pred = pred_by_id.get(task_id)
        scenario_durations = {}
        if row_pred is not None:
            if not math.isclose(float(actual), float(row_pred["actual_duration_minutes"]), abs_tol=1e-6):
                raise ValueError(f"Prediction/source duration mismatch for task {task_id}.")
            for name, column in (
                ("Q50", "q50_predicted_duration_minutes"),
                ("Q80", "q80_planned_duration_minutes"),
            ):
                estimate = float(row_pred[column])
                if not math.isfinite(estimate) or estimate <= 0:
                    raise ValueError(f"Task {task_id}: invalid {column} value.")
                scenario_durations[name] = int(math.ceil(estimate))
        tasks[task_id] = {
            "task_id": task_id, "cancelled": parse_bool(row["cancelled"]),
            "actual_duration": actual, "scheduled_duration": actual,
            "scenario_durations": scenario_durations,
            "start": start, "end": end, "priority": int(row["priority"]),
            "penalty": float(row["penalty"]), "candidates": candidates,
        }

    tech_windows = {}
    for row in tech_frame.to_dict("records"):
        tech_id = int(row["technician_id"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError(f"Technician {tech_id}: invalid shift.")
        tech_windows[tech_id] = (start, end)

    results = []
    for name, column in (
        ("Q50", "q50_predicted_duration_minutes"),
        ("Q80", "q80_planned_duration_minutes"),
    ):
        print(f"\nRunning {name} scheduling scenario...")
        results.append(schedule_scenario(
            name, column, tasks, tech_windows, t2t, t2tech, tech2t
        ))

    baseline_result = {
        "scenario": "Source durations (insertion baseline)",
        "predicted_holdout_tasks": 0,
        "assigned_tasks": int(baseline["assigned_tasks"]),
        "unassigned_tasks": int(baseline["unassigned_tasks"]),
        "active_task_coverage": float(baseline["active_task_coverage"]),
        "routes_used": int(baseline["technician_routes_used"]),
        "planned_travel_minutes": float(baseline["total_travel_minutes_including_depot_returns"]),
        "routes_with_replay_violations": 0,
        "unique_tasks_with_window_or_shift_violations": 0,
        "return_after_shift_violations": 0,
    }
    output = {
        "experiment": "chronological_q50_q80_duration_scheduling_counterfactual",
        "baseline": baseline_result,
        "scenarios": results,
        "caveats": [
            "Only chronologically held-out tasks use Q50/Q80 predictions; earlier tasks retain source durations.",
            "Predicted durations are rounded upward to whole minutes for scheduling.",
            "Routes are replayed using source-recorded durations; these are not independently verified realized service times.",
            "This is a counterfactual on telecommunications field-service data, not plumbing-business evidence.",
            "A schedule-feasibility improvement is not guaranteed by quantile coverage.",
            "The insertion heuristic is a heuristic, not an exact optimizer.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    rows = [baseline_result] + results
    keys = [
        "scenario", "predicted_holdout_tasks", "assigned_tasks", "unassigned_tasks",
        "active_task_coverage", "routes_used", "planned_travel_minutes",
        "routes_with_replay_violations", "unique_tasks_with_window_or_shift_violations",
        "return_after_shift_violations",
    ]
    write_csv(COMPARISON_PATH, keys, [{key: row.get(key, "") for key in keys} for row in rows])

    print("\nQUANTILE SCHEDULING IMPACT — COUNTERFACTUAL")
    print("===========================================")
    for row in rows:
        print(
            f"{row['scenario']}: assigned={row['assigned_tasks']:,}, "
            f"coverage={row['active_task_coverage']:.2%}, "
            f"routes={row['routes_used']:,}, travel={row['planned_travel_minutes']:,.1f} min, "
            f"replay-violated routes={row.get('routes_with_replay_violations', 0)}"
        )
    print(f"Saved {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Saved {COMPARISON_PATH.relative_to(ROOT)}")
    print("Interpret only as a counterfactual on telecom field-service data.")


if __name__ == "__main__":
    main()
