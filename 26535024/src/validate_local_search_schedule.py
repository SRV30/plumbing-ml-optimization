"""Validate the output produced by the local-search scheduler.

Run from the repository root:
    python src/validate_local_search_schedule.py

Checks task accounting, source eligibility, route order, time windows, travel legs,
service duration and end-of-shift return travel. Reads matrices via memory-mapping.
No input or output files are modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
TASKS_PATH = DATA / "tasks.csv"
TECHNICIANS_PATH = DATA / "technicians.csv"
SCHEDULE_PATHS = {
    "local": (
        DATA / "local_search_task_status.csv",
        DATA / "local_search_routes.csv",
        DATA / "local_search_summary.json",
    ),
    "enhanced": (
        DATA / "enhanced_local_search_task_status.csv",
        DATA / "enhanced_local_search_routes.csv",
        DATA / "enhanced_local_search_summary.json",
    ),
}
T2T_PATH = DATA / "task_to_task.npy"
T2TECH_PATH = DATA / "task_to_technician.npy"
TECH2T_PATH = DATA / "technician_to_task.npy"

TOLERANCE_MINUTES = 1e-4


def decode_list(value: Any) -> list:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")
    return parsed


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def close(a: float, b: float, tolerance: float = TOLERANCE_MINUTES) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate a generated technician schedule."
    )
    parser.add_argument(
        "--schedule",
        choices=sorted(SCHEDULE_PATHS),
        default="local",
        help="Which schedule to validate: 'local' (default) or 'enhanced'.",
    )
    args = parser.parse_args()
    schedule_path, routes_path, summary_path = SCHEDULE_PATHS[args.schedule]

    required = [
        TASKS_PATH, TECHNICIANS_PATH, schedule_path, routes_path, summary_path,
        T2T_PATH, T2TECH_PATH, TECH2T_PATH,
    ]
    missing_files = [path for path in required if not path.exists()]
    if missing_files:
        print("Required input/output files missing:")
        for path in missing_files:
            print(f"- {path.relative_to(ROOT)}")
        print("Run the parser, matrix builder, and scheduler first.")
        return

    tasks = pd.read_csv(TASKS_PATH)
    technicians = pd.read_csv(TECHNICIANS_PATH)
    schedule = pd.read_csv(schedule_path, keep_default_na=False)
    routes = pd.read_csv(routes_path, keep_default_na=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    t2t = np.load(T2T_PATH, mmap_mode="r")
    t2tech = np.load(T2TECH_PATH, mmap_mode="r")
    tech2t = np.load(TECH2T_PATH, mmap_mode="r")

    task_by_id = {int(row["task_id"]): row for row in tasks.to_dict("records")}
    tech_by_id = {int(row["technician_id"]): row for row in technicians.to_dict("records")}
    status_by_id = {int(row["task_id"]): row for row in schedule.to_dict("records")}

    errors: list[str] = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    print(f"{args.schedule.upper()} SCHEDULE VALIDATION")
    print("=" * (len(args.schedule) + 22))
    print(f"Schedule status file: {schedule_path.name}")
    print(f"Route file: {routes_path.name}")
    print(f"Task source rows: {len(tasks):,}")
    print(f"Schedule status rows: {len(schedule):,}")
    print(f"Route assignment rows: {len(routes):,}")

    # Table-level accounting.
    check(schedule["task_id"].is_unique, "Task-status table has duplicate task IDs.")
    check(
        set(status_by_id) == set(task_by_id),
        "Task-status IDs do not exactly match source task IDs.",
    )

    assigned_rows = schedule[schedule["status"] == "assigned"]
    check(
        not assigned_rows["task_id"].duplicated().any(),
        "A task is marked assigned more than once in the status table.",
    )
    check(
        routes["task_id"].is_unique,
        "A task appears more than once in the route table.",
    )
    check(
        set(routes["task_id"].astype(int)) == set(assigned_rows["task_id"].astype(int)),
        "Route task IDs do not match tasks marked assigned.",
    )

    status_values = set(schedule["status"])
    check(
        status_values.issubset({"assigned", "unassigned", "excluded_cancelled"}),
        f"Unexpected schedule status values: {sorted(status_values)}",
    )

    for task_id, source in task_by_id.items():
        row = status_by_id.get(task_id)
        if row is None:
            continue
        source_cancelled = as_bool(source["cancelled"])
        expected_status = str(row["status"])
        check(
            (source_cancelled and expected_status == "excluded_cancelled")
            or (not source_cancelled and expected_status != "excluded_cancelled"),
            f"Task {task_id}: source cancellation flag disagrees with scheduler status.",
        )
        if expected_status == "assigned":
            try:
                tech_id = int(row["technician_id"])
                eligible = decode_list(source["relevant_technician_ids_json"])
                check(
                    tech_id in eligible,
                    f"Task {task_id}: assigned technician {tech_id} is not in source eligibility list.",
                )
                check(
                    tech_id in tech_by_id,
                    f"Task {task_id}: assigned technician {tech_id} missing from technician table.",
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                errors.append(f"Task {task_id}: could not verify technician eligibility.")

    # Route-level validation in the scheduler's recorded sequence.
    route_groups = 0
    for technician_id, group in routes.groupby("technician_id"):
        try:
            tech_id = int(technician_id)
        except (TypeError, ValueError):
            errors.append(f"Invalid technician ID in route table: {technician_id!r}")
            continue

        if tech_id not in tech_by_id:
            errors.append(f"Route uses unknown technician ID {tech_id}.")
            continue

        route_groups += 1
        tech = tech_by_id[tech_id]
        shift_start = pd.to_datetime(tech["time_window_start"], errors="coerce")
        shift_end = pd.to_datetime(tech["time_window_end"], errors="coerce")
        ordered = group.sort_values("route_sequence")

        seq = ordered["route_sequence"].astype(int).tolist()
        check(seq == list(range(1, len(seq) + 1)), f"Technician {tech_id}: route sequence is not contiguous.")

        previous_task_id: int | None = None
        previous_finish: pd.Timestamp | None = None

        for route_row in ordered.to_dict("records"):
            task_id = int(route_row["task_id"])
            if task_id not in task_by_id:
                errors.append(f"Route references unknown task ID {task_id}.")
                continue
            source = task_by_id[task_id]
            service_start = pd.to_datetime(route_row["service_start"], errors="coerce")
            service_finish = pd.to_datetime(route_row["service_finish"], errors="coerce")
            task_start = pd.to_datetime(source["time_window_start"], errors="coerce")
            task_end = pd.to_datetime(source["time_window_end"], errors="coerce")
            duration = int(source["duration"])

            check(not pd.isna(service_start) and not pd.isna(service_finish),
                  f"Task {task_id}: service timestamps are invalid.")
            if pd.isna(service_start) or pd.isna(service_finish):
                continue

            check(service_start >= task_start, f"Task {task_id}: service starts before task window.")
            check(service_finish <= task_end, f"Task {task_id}: service finishes after task window.")
            check(service_start >= shift_start, f"Task {task_id}: service starts before technician shift.")
            check(service_finish <= shift_end, f"Task {task_id}: service finishes after technician shift.")
            check(service_finish == service_start + pd.Timedelta(minutes=duration),
                  f"Task {task_id}: service duration does not match source duration.")

            if previous_task_id is None:
                inbound = float(tech2t[tech_id - 1, task_id - 1])
                expected_arrival = shift_start + pd.Timedelta(minutes=inbound)
            else:
                inbound = float(t2t[previous_task_id - 1, task_id - 1])
                expected_arrival = previous_finish + pd.Timedelta(minutes=inbound)

            recorded_inbound = float(route_row["inbound_travel_minutes"])
            check(close(recorded_inbound, inbound),
                  f"Task {task_id}: recorded inbound travel differs from matrix value.")
            check(service_start >= expected_arrival,
                  f"Task {task_id}: service starts before arrival after inbound travel.")

            previous_task_id = task_id
            previous_finish = service_finish

        if not ordered.empty:
            last_row = ordered.iloc[-1]
            last_task_id = int(last_row["task_id"])
            final_return = float(t2tech[last_task_id - 1, tech_id - 1])
            recorded_return = pd.to_numeric(
                pd.Series([last_row["return_travel_minutes"]]), errors="coerce"
            ).iloc[0]
            check(
                pd.notna(recorded_return) and close(float(recorded_return), final_return),
                f"Technician {tech_id}: final return-travel value is missing or incorrect.",
            )
            if not pd.isna(previous_finish):
                check(
                    previous_finish + pd.Timedelta(minutes=final_return) <= shift_end,
                    f"Technician {tech_id}: final return trip finishes after shift end.",
                )

            # Return travel should be blank for all earlier route rows.
            for nonfinal in ordered.iloc[:-1].to_dict("records"):
                check(
                    str(nonfinal["return_travel_minutes"]).strip() == "",
                    f"Technician {tech_id}: a non-final task has a return-travel value.",
                )

    # Summary consistency.
    computed_assigned = int((schedule["status"] == "assigned").sum())
    computed_cancelled = int((schedule["status"] == "excluded_cancelled").sum())
    computed_unassigned = int((schedule["status"] == "unassigned").sum())
    check(computed_assigned == int(summary["assigned_tasks"]), "Summary assigned count mismatch.")
    check(computed_cancelled == int(summary["cancelled_excluded"]), "Summary cancelled count mismatch.")
    check(computed_unassigned == int(summary["unassigned_tasks"]), "Summary unassigned count mismatch.")
    check(route_groups == int(summary["technician_routes_used"]), "Summary route count mismatch.")

    # Validate travel totals against the route rows, including depot legs.
    if "inbound_travel_minutes" in routes.columns and "return_travel_minutes" in routes.columns:
        inbound_total = pd.to_numeric(
            routes["inbound_travel_minutes"], errors="coerce"
        ).fillna(0.0).sum()
        return_total = pd.to_numeric(
            routes["return_travel_minutes"], errors="coerce"
        ).fillna(0.0).sum()
        computed_travel = float(inbound_total + return_total)
        expected_travel = summary.get(
            "final_travel_minutes_including_depot_returns",
            summary.get("total_travel_minutes_including_depot_returns"),
        )
        if expected_travel is not None:
            check(
                close(computed_travel, float(expected_travel), tolerance=1e-2),
                "Summary total travel does not match the route-table travel legs.",
            )

    print(f"Route groups validated: {route_groups:,}")
    print(f"Checks performed: {checks:,}")
    print(f"Validation errors: {len(errors):,}")

    if errors:
        print("\nFirst 30 errors:")
        for error in errors[:30]:
            print("-", error)
        print("\nValidation FAILED. Do not cite baseline results as verified yet.")
    else:
        print("\nVALIDATION PASSED")
        print("Task accounting, eligibility, route timing, travel legs, and summary consistency passed.")
        print("This validates internal feasibility checks, not global optimality.")


if __name__ == "__main__":
    main()
