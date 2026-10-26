"""Test the scheduling impact of duration estimates on held-out tasks.

Run from repository root:
    python src/evaluate_duration_scheduling_impact.py

Experiment design:
- Load Ridge predictions for the chronological hold-out tasks.
- Use ceil(predicted duration) for those held-out tasks; use source durations for
  earlier tasks, whose durations are assumed known in this counterfactual.
- Build a route-insertion schedule using those mixed duration inputs.
- Replay the resulting route sequences using source-recorded durations to find
  service-window, shift-end, and return-to-depot violations.
- Compare against the previously validated route-insertion schedule built using
  source-recorded durations.

This is an exploratory counterfactual on telecommunications field-service data,
not proof that the recorded duration is an unknown real-world plumbing outcome.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

TASKS_PATH = DATA / "tasks.csv"
TECHNICIANS_PATH = DATA / "technicians.csv"
T2T_PATH = DATA / "task_to_task.npy"
T2TECH_PATH = DATA / "task_to_technician.npy"
TECH2T_PATH = DATA / "technician_to_task.npy"

DURATION_PREDICTIONS_PATH = RESULTS / "duration_temporal_predictions.csv"
ACTUAL_BASELINE_SUMMARY_PATH = DATA / "insertion_summary.json"
ACTUAL_BASELINE_STATUS_PATH = DATA / "insertion_task_status.csv"

STATUS_PATH = DATA / "duration_estimate_task_status.csv"
ROUTES_PATH = DATA / "duration_estimate_routes.csv"
SUMMARY_PATH = RESULTS / "duration_scheduling_impact.json"
COMPARISON_PATH = RESULTS / "duration_scheduling_impact_comparison.csv"

STATUS_FIELDS = [
    "task_id", "cancelled", "status", "reason", "technician_id",
    "route_sequence", "service_start_predicted", "service_finish_predicted",
    "estimated_duration_minutes", "actual_duration_minutes",
    "inbound_travel_minutes", "return_travel_minutes", "priority", "penalty",
]
ROUTE_FIELDS = [
    "technician_id", "route_sequence", "task_id",
    "service_start_predicted", "service_finish_predicted",
    "estimated_duration_minutes", "actual_duration_minutes",
    "inbound_travel_minutes", "return_travel_minutes",
    "priority", "penalty",
]


def read_json_list(value: Any, field: str, row_id: int) -> list:
    try:
        result = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Task {row_id}: bad JSON in {field}: {error}") from error
    if not isinstance(result, list):
        raise ValueError(f"Task {row_id}: {field} must be a JSON list.")
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


def evaluate_predicted_route(
    technician_id: int,
    route_ids: list[int],
    tasks: dict[int, dict[str, Any]],
    shift_start: pd.Timestamp,
    shift_end: pd.Timestamp,
    t2t: np.ndarray,
    t2tech: np.ndarray,
    tech2t: np.ndarray,
) -> dict[str, Any] | None:
    """Schedule one ordered route using each task's scheduled duration."""
    if not route_ids:
        return {"rows": [], "travel": 0.0}

    previous_id: int | None = None
    previous_finish = shift_start
    travel_total = 0.0
    rows: list[dict[str, Any]] = []

    for sequence, task_id in enumerate(route_ids, start=1):
        task = tasks[task_id]
        inbound = (
            float(tech2t[technician_id - 1, task_id - 1])
            if previous_id is None
            else float(t2t[previous_id - 1, task_id - 1])
        )
        if not np.isfinite(inbound) or inbound < 0:
            return None
        arrival = previous_finish + pd.Timedelta(minutes=inbound)
        service_start = max(task["start"], arrival)
        service_finish = service_start + pd.Timedelta(minutes=task["scheduled_duration"])

        if service_finish > task["end"] or service_start < shift_start or service_finish > shift_end:
            return None

        travel_total += inbound
        rows.append({
            "technician_id": technician_id,
            "route_sequence": sequence,
            "task_id": task_id,
            "service_start_predicted": service_start.isoformat(),
            "service_finish_predicted": service_finish.isoformat(),
            "estimated_duration_minutes": task["scheduled_duration"],
            "actual_duration_minutes": task["actual_duration"],
            "inbound_travel_minutes": round(inbound, 3),
            "return_travel_minutes": "",
            "priority": task["priority"],
            "penalty": task["penalty"],
        })
        previous_id = task_id
        previous_finish = service_finish

    assert previous_id is not None
    return_travel = float(t2tech[previous_id - 1, technician_id - 1])
    if not np.isfinite(return_travel) or return_travel < 0:
        return None
    if previous_finish + pd.Timedelta(minutes=return_travel) > shift_end:
        return None

    rows[-1]["return_travel_minutes"] = round(return_travel, 3)
    travel_total += return_travel
    return {"rows": rows, "travel": travel_total}


def replay_with_actual_durations(
    technician_id: int,
    route_ids: list[int],
    tasks: dict[int, dict[str, Any]],
    shift_start: pd.Timestamp,
    shift_end: pd.Timestamp,
    t2t: np.ndarray,
    t2tech: np.ndarray,
    tech2t: np.ndarray,
) -> dict[str, Any]:
    """Replay a route sequentially with true/source-recorded durations."""
    previous_id: int | None = None
    previous_finish = shift_start
    violations: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []

    for sequence, task_id in enumerate(route_ids, start=1):
        task = tasks[task_id]
        inbound = (
            float(tech2t[technician_id - 1, task_id - 1])
            if previous_id is None
            else float(t2t[previous_id - 1, task_id - 1])
        )
        arrival = previous_finish + pd.Timedelta(minutes=inbound)
        actual_start = max(task["start"], arrival)
        actual_finish = actual_start + pd.Timedelta(minutes=task["actual_duration"])
        reasons = []
        if actual_finish > task["end"]:
            reasons.append("task_window")
        if actual_start < shift_start or actual_finish > shift_end:
            reasons.append("technician_shift")
        if reasons:
            violations.append({
                "task_id": task_id,
                "technician_id": technician_id,
                "route_sequence": sequence,
                "violation_types": reasons,
                "actual_service_start": actual_start.isoformat(),
                "actual_service_finish": actual_finish.isoformat(),
            })
        actual_rows.append({
            "task_id": task_id,
            "actual_service_start": actual_start.isoformat(),
            "actual_service_finish": actual_finish.isoformat(),
            "violation_types": reasons,
        })
        previous_id = task_id
        previous_finish = actual_finish

    return_violation = False
    return_minutes = 0.0
    if previous_id is not None:
        return_minutes = float(t2tech[previous_id - 1, technician_id - 1])
        return_time = previous_finish + pd.Timedelta(minutes=return_minutes)
        return_violation = return_time > shift_end
        if return_violation:
            violations.append({
                "task_id": previous_id,
                "technician_id": technician_id,
                "route_sequence": len(route_ids),
                "violation_types": ["return_after_shift"],
                "actual_service_start": actual_rows[-1]["actual_service_start"],
                "actual_service_finish": actual_rows[-1]["actual_service_finish"],
            })

    return {
        "violations": violations,
        "return_violation": return_violation,
        "return_minutes": return_minutes,
        "actual_rows": actual_rows,
    }


def main() -> None:
    required = [
        TASKS_PATH, TECHNICIANS_PATH, T2T_PATH, T2TECH_PATH, TECH2T_PATH,
        DURATION_PREDICTIONS_PATH, ACTUAL_BASELINE_SUMMARY_PATH, ACTUAL_BASELINE_STATUS_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Required file(s) missing:")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        print("Run parsing, matrices, route-insertion scheduling, and temporal duration ML first.")
        return

    tasks_frame = pd.read_csv(TASKS_PATH, keep_default_na=False)
    tech_frame = pd.read_csv(TECHNICIANS_PATH, keep_default_na=False)
    predictions_frame = pd.read_csv(DURATION_PREDICTIONS_PATH)
    baseline_summary = json.loads(ACTUAL_BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))

    ridge = predictions_frame[predictions_frame["model"] == "Ridge regression"].copy()
    ridge["task_id"] = ridge["task_id"].astype(int)
    if ridge["task_id"].duplicated().any():
        raise ValueError("Ridge prediction file contains duplicate task IDs.")
    predicted_duration = dict(zip(
        ridge["task_id"],
        np.ceil(pd.to_numeric(ridge["predicted_duration_minutes"], errors="raise").clip(lower=1)).astype(int),
    ))
    actual_duration_from_prediction = dict(zip(
        ridge["task_id"],
        pd.to_numeric(ridge["actual_duration_minutes"], errors="raise").astype(float),
    ))

    t2t = np.load(T2T_PATH, mmap_mode="r")
    t2tech = np.load(T2TECH_PATH, mmap_mode="r")
    tech2t = np.load(TECH2T_PATH, mmap_mode="r")

    task_by_id: dict[int, dict[str, Any]] = {}
    status_by_id: dict[int, dict[str, Any]] = {}
    cancelled_count = 0
    empty_eligibility_count = 0

    for row in tasks_frame.to_dict(orient="records"):
        task_id = int(row["task_id"])
        actual_duration = int(row["duration"])
        is_cancelled = parse_bool(row["cancelled"])
        pred_value = predicted_duration.get(task_id)
        schedule_duration = pred_value if pred_value is not None else actual_duration
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        candidates = read_json_list(row["relevant_technician_ids_json"], "relevant_technician_ids_json", task_id)

        if pd.isna(start) or pd.isna(end) or end < start or actual_duration <= 0 or schedule_duration <= 0:
            raise ValueError(f"Task {task_id}: invalid time window or duration.")
        if task_id in actual_duration_from_prediction and not math.isclose(
            float(actual_duration), actual_duration_from_prediction[task_id], abs_tol=1e-6
        ):
            raise ValueError(f"Duration target mismatch for task {task_id} in prediction file.")

        task_by_id[task_id] = {
            "task_id": task_id,
            "cancelled": is_cancelled,
            "actual_duration": actual_duration,
            "scheduled_duration": int(schedule_duration),
            "prediction_used": pred_value is not None,
            "start": start,
            "end": end,
            "priority": int(row["priority"]),
            "penalty": float(row["penalty"]),
            "candidates": candidates,
        }
        if is_cancelled:
            cancelled_count += 1
            status_by_id[task_id] = {
                "task_id": task_id, "cancelled": True, "status": "excluded_cancelled",
                "reason": "Source record is marked cancelled.", "technician_id": "",
                "route_sequence": "", "service_start_predicted": "",
                "service_finish_predicted": "", "estimated_duration_minutes": schedule_duration,
                "actual_duration_minutes": actual_duration, "inbound_travel_minutes": "",
                "return_travel_minutes": "", "priority": int(row["priority"]),
                "penalty": float(row["penalty"]),
            }
        elif not candidates:
            empty_eligibility_count += 1
            status_by_id[task_id] = {
                "task_id": task_id, "cancelled": False, "status": "unassigned",
                "reason": "Source record contains no relevant technician IDs.",
                "technician_id": "", "route_sequence": "", "service_start_predicted": "",
                "service_finish_predicted": "", "estimated_duration_minutes": schedule_duration,
                "actual_duration_minutes": actual_duration, "inbound_travel_minutes": "",
                "return_travel_minutes": "", "priority": int(row["priority"]),
                "penalty": float(row["penalty"]),
            }

    technician_windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    technician_routes: dict[int, list[int]] = {}
    for row in tech_frame.to_dict(orient="records"):
        technician_id = int(row["technician_id"])
        start = pd.to_datetime(row["time_window_start"], errors="coerce")
        end = pd.to_datetime(row["time_window_end"], errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError(f"Technician {technician_id}: invalid shift.")
        technician_windows[technician_id] = (start, end)
        technician_routes[technician_id] = []

    active = [task for task in task_by_id.values() if not task["cancelled"] and task["candidates"]]
    active.sort(key=lambda task: (task["priority"], -task["penalty"], task["end"], task["start"], task["task_id"]))

    assigned = 0
    infeasible_count = 0
    for index, task in enumerate(active, start=1):
        if index % 250 == 0:
            print(f"Scheduling task {index:,} of {len(active):,}...")
        best: dict[str, Any] | None = None

        for technician_id in task["candidates"]:
            if technician_id not in technician_routes:
                continue
            shift_start, shift_end = technician_windows[technician_id]
            route = technician_routes[technician_id]
            old_eval = evaluate_predicted_route(technician_id, route, task_by_id, shift_start, shift_end, t2t, t2tech, tech2t)
            if old_eval is None:
                raise RuntimeError(f"Existing predicted-duration route for technician {technician_id} is infeasible.")

            for position in range(len(route) + 1):
                proposed = route[:position] + [task["task_id"]] + route[position:]
                evaluation = evaluate_predicted_route(technician_id, proposed, task_by_id, shift_start, shift_end, t2t, t2tech, tech2t)
                if evaluation is None:
                    continue
                marginal = evaluation["travel"] - old_eval["travel"]
                score = (marginal, len(route), technician_id, position)
                if best is None or score < best["score"]:
                    best = {"score": score, "technician_id": technician_id, "route": proposed, "evaluation": evaluation}

        if best is None:
            infeasible_count += 1
            status_by_id[task["task_id"]] = {
                "task_id": task["task_id"], "cancelled": False, "status": "unassigned",
                "reason": "No feasible insertion using duration estimate.", "technician_id": "",
                "route_sequence": "", "service_start_predicted": "", "service_finish_predicted": "",
                "estimated_duration_minutes": task["scheduled_duration"], "actual_duration_minutes": task["actual_duration"],
                "inbound_travel_minutes": "", "return_travel_minutes": "",
                "priority": task["priority"], "penalty": task["penalty"],
            }
            continue

        technician_id = best["technician_id"]
        technician_routes[technician_id] = best["route"]
        assigned += 1
        status_by_id[task["task_id"]] = {
            "task_id": task["task_id"], "cancelled": False, "status": "assigned", "reason": "",
            "technician_id": technician_id, "route_sequence": "", "service_start_predicted": "",
            "service_finish_predicted": "", "estimated_duration_minutes": task["scheduled_duration"],
            "actual_duration_minutes": task["actual_duration"], "inbound_travel_minutes": "",
            "return_travel_minutes": "", "priority": task["priority"], "penalty": task["penalty"],
        }

    route_rows: list[dict[str, Any]] = []
    total_predicted_travel = 0.0
    for technician_id, route in technician_routes.items():
        if not route:
            continue
        shift_start, shift_end = technician_windows[technician_id]
        result = evaluate_predicted_route(technician_id, route, task_by_id, shift_start, shift_end, t2t, t2tech, tech2t)
        if result is None:
            raise RuntimeError(f"Final route for technician {technician_id} is infeasible under estimates.")
        total_predicted_travel += result["travel"]
        for row in result["rows"]:
            route_rows.append(row)
            status = status_by_id[row["task_id"]]
            status["route_sequence"] = row["route_sequence"]
            status["service_start_predicted"] = row["service_start_predicted"]
            status["service_finish_predicted"] = row["service_finish_predicted"]
            status["inbound_travel_minutes"] = row["inbound_travel_minutes"]
            status["return_travel_minutes"] = row["return_travel_minutes"]

    
    route_violations: list[dict[str, Any]] = []
    task_violations: list[dict[str, Any]] = []
    routes_with_violations = 0
    actual_feasible_routes = 0
    for technician_id, route in technician_routes.items():
        if not route:
            continue
        shift_start, shift_end = technician_windows[technician_id]
        replay = replay_with_actual_durations(technician_id, route, task_by_id, shift_start, shift_end, t2t, t2tech, tech2t)
        if replay["violations"]:
            routes_with_violations += 1
            route_violations.extend(replay["violations"])
        else:
            actual_feasible_routes += 1
        for violation in replay["violations"]:
            if any(kind in {"task_window", "technician_shift"} for kind in violation["violation_types"]):
                task_violations.append(violation)

    route_rows.sort(key=lambda row: (row["technician_id"], row["route_sequence"]))
    status_rows = [status_by_id[task_id] for task_id in sorted(status_by_id)]
    write_csv(STATUS_PATH, STATUS_FIELDS, status_rows)
    write_csv(ROUTES_PATH, ROUTE_FIELDS, route_rows)

    active_count = len(task_by_id) - cancelled_count
    unassigned_count = sum(row["status"] == "unassigned" for row in status_rows)
    actual_duration_mae = None
    if actual_duration_from_prediction:
        errors = [
            predicted_duration[task_id] - task_by_id[task_id]["actual_duration"]
            for task_id in predicted_duration
        ]
        actual_duration_mae = round(float(np.mean(np.abs(errors))), 3)

    comparison = {
        "baseline_actual_duration": {
            "assigned_tasks": int(baseline_summary["assigned_tasks"]),
            "unassigned_tasks": int(baseline_summary["unassigned_tasks"]),
            "active_task_coverage": float(baseline_summary["active_task_coverage"]),
            "routes_used": int(baseline_summary["technician_routes_used"]),
            "travel_minutes": float(baseline_summary["total_travel_minutes_including_depot_returns"]),
            "interpretation": "Existing route-insertion schedule built using source-recorded durations.",
        },
        "estimated_duration_schedule": {
            "duration_predictions_used_for_tasks": len(predicted_duration),
            "assigned_tasks": assigned,
            "unassigned_tasks": unassigned_count,
            "empty_source_eligibility_count": empty_eligibility_count,
            "no_feasible_insertion_count": infeasible_count,
            "active_task_coverage": round(assigned / active_count, 6) if active_count else 0.0,
            "routes_used": sum(bool(route) for route in technician_routes.values()),
            "travel_minutes_using_estimated_duration_route_order": round(total_predicted_travel, 3),
            "duration_predictions_mae_minutes_on_holdout": actual_duration_mae,
            "routes_with_any_violation_when_replayed_with_source_durations": routes_with_violations,
            "routes_feasible_when_replayed_with_source_durations": actual_feasible_routes,
            "tasks_with_task-window_or_shift_violations_on_replay": len({row["task_id"] for row in task_violations}),
            "violation_examples": task_violations[:50],
        },
        "caveats": [
            "Counterfactual experiment: only chronologically held-out tasks use Ridge predictions; older tasks retain source durations.",
            "Predictions are rounded up to integer minutes for scheduling.",
            "Replay uses source-recorded duration values to test whether estimated-duration routes would remain feasible.",
            "Source durations are defined as scheduling inputs in the associated paper; this experiment is only relevant if duration can be unknown at intake.",
            "Data represents telecom field service, not actual plumbing operations.",
            "The insertion heuristic is not an exact or optimal solver.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    baseline = comparison["baseline_actual_duration"]
    estimate = comparison["estimated_duration_schedule"]
    comparison_rows = [
        {
            "scenario": "Insertion schedule using source durations",
            "assigned_tasks": baseline["assigned_tasks"],
            "unassigned_tasks": baseline["unassigned_tasks"],
            "active_task_coverage": baseline["active_task_coverage"],
            "routes_used": baseline["routes_used"],
            "travel_minutes": baseline["travel_minutes"],
            "replay_infeasible_routes": 0,
            "replay_task_violations": 0,
        },
        {
            "scenario": "Insertion schedule using Ridge estimates for held-out tasks",
            "assigned_tasks": estimate["assigned_tasks"],
            "unassigned_tasks": estimate["unassigned_tasks"],
            "active_task_coverage": estimate["active_task_coverage"],
            "routes_used": estimate["routes_used"],
            "travel_minutes": estimate["travel_minutes_using_estimated_duration_route_order"],
            "replay_infeasible_routes": estimate["routes_with_any_violation_when_replayed_with_source_durations"],
            "replay_task_violations": estimate["tasks_with_task-window_or_shift_violations_on_replay"],
        },
    ]
    write_csv(COMPARISON_PATH, list(comparison_rows[0].keys()), comparison_rows)

    print("\nDURATION-ESTIMATE SCHEDULING IMPACT")
    print("===================================")
    print(f"Ridge held-out task predictions used: {len(predicted_duration):,}")
    print(f"Hold-out duration MAE: {actual_duration_mae:.2f} minutes" if actual_duration_mae is not None else "Hold-out duration MAE: unavailable")
    print(f"Active tasks: {active_count:,}")
    print(f"Assigned using estimates: {assigned:,}")
    print(f"Unassigned using estimates: {unassigned_count:,}")
    print(f"Coverage using estimates: {estimate['active_task_coverage']:.2%}")
    print(f"Routes used: {estimate['routes_used']:,}")
    print(f"Travel minutes on estimated-duration routes: {total_predicted_travel:,.2f}")
    print(f"Routes with violations when replayed using source durations: {routes_with_violations:,}")
    print(f"Unique tasks with actual-duration window/shift violations: {estimate['tasks_with_task-window_or_shift_violations_on_replay']:,}")
    print("\nBaseline using source durations:")
    print(f"  Assigned: {baseline['assigned_tasks']:,}; coverage: {baseline['active_task_coverage']:.2%}; travel: {baseline['travel_minutes']:,.2f} minutes")
    print("\nSaved outputs:")
    print(f"- {STATUS_PATH.relative_to(ROOT)}")
    print(f"- {ROUTES_PATH.relative_to(ROOT)}")
    print(f"- {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"- {COMPARISON_PATH.relative_to(ROOT)}")
    print("\nTreat this as a counterfactual experiment, not proof of real plumbing-business impact.")


if __name__ == "__main__":
    main()
