"""Compare validated greedy, route-insertion, and local-search summaries.

Run from the repository root:
    python src/compare_schedulers.py

Reads the three JSON summaries written by the scheduling experiments and produces:
    data/processed/scheduler_comparison.csv
    results/figures/travel_time_comparison.png
    results/figures/task_coverage_comparison.png
    results/figures/routes_used_comparison.png

The source summaries are not modified. Comparisons are meaningful only after
both schedule validators pass.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"

SUMMARY_FILES = {
    "Greedy append-only": DATA / "baseline_summary.json",
    "Route insertion": DATA / "insertion_summary.json",
    "Local search": DATA / "local_search_summary.json",
}
ENHANCED_SUMMARY = DATA / "enhanced_local_search_summary.json"
CSV_OUTPUT = DATA / "scheduler_comparison.csv"


def _normalise_summary(label: str, summary: dict) -> dict:
    """Accept both legacy experiment summaries and the dashboard's unified schema."""
    aliases = {
        "total_travel_minutes_including_depot_returns": (
            "total_travel_minutes_including_depot_returns",
            "final_travel_minutes_including_depot_returns",
            "total_travel_minutes",
            "travel_minutes",
            "total_travel_time_minutes",
            "total_travel_time",
        ),
        "total_service_minutes": (
            "total_service_minutes",
            "service_minutes",
            "total_duration_minutes",
        ),
        "technician_routes_used": (
            "technician_routes_used",
            "routes_used",
            "technician_day_routes",
            "number_of_routes",
        ),
        "active_task_coverage": (
            "active_task_coverage",
            "coverage",
            "task_coverage",
        ),
        "assigned_tasks": ("assigned_tasks", "tasks_assigned", "assigned"),
        "unassigned_tasks": ("unassigned_tasks", "tasks_unassigned", "unassigned"),
    }
    result = dict(summary)
    for canonical, candidates in aliases.items():
        if result.get(canonical) is None:
            for candidate in candidates:
                if result.get(candidate) is not None:
                    result[canonical] = result[candidate]
                    break
    required = (
        "total_travel_minutes_including_depot_returns",
        "technician_routes_used",
        "active_task_coverage",
        "assigned_tasks",
        "unassigned_tasks",
    )
    missing = [key for key in required if result.get(key) is None]
    if missing:
        available = ", ".join(sorted(summary.keys()))
        raise ValueError(
            f"{label} summary cannot be normalised; missing {missing}. "
            f"Available keys: {available}"
        )
    result.setdefault("total_service_minutes", None)
    return result


def main() -> None:
    missing = [path for path in SUMMARY_FILES.values() if not path.exists()]
    if missing:
        print("Scheduler summary file(s) not found:")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        print("Run and validate both scheduling experiments first.")
        return

    summaries: dict[str, dict] = {}
    for label, path in SUMMARY_FILES.items():
        raw_summary = json.loads(path.read_text(encoding="utf-8"))
        summaries[label] = _normalise_summary(label, raw_summary)

    if ENHANCED_SUMMARY.exists():
        enhanced = _normalise_summary(
            "Enhanced local search",
            json.loads(ENHANCED_SUMMARY.read_text(encoding="utf-8")),
        )
        
        if enhanced.get("total_service_minutes") is None:
            enhanced_routes_path = DATA / "enhanced_local_search_routes.csv"
            if enhanced_routes_path.exists():
                with enhanced_routes_path.open("r", encoding="utf-8", newline="") as handle:
                    enhanced_rows = list(csv.DictReader(handle))
                if enhanced_rows and "duration_minutes" in enhanced_rows[0]:
                    enhanced["total_service_minutes"] = sum(
                        float(row.get("duration_minutes") or 0) for row in enhanced_rows
                    )
        summaries["Enhanced local search"] = enhanced

    metrics = [
        ("assigned_tasks", "Tasks assigned", "count"),
        ("unassigned_tasks", "Tasks unassigned", "count"),
        ("active_task_coverage", "Active-task coverage", "fraction"),
        ("technician_routes_used", "Technician routes used", "count"),
        ("total_travel_minutes_including_depot_returns", "Travel time including depot returns", "minutes"),
        ("total_service_minutes", "Scheduled service time", "minutes"),
    ]

    method_columns = {
        "Greedy append-only": "greedy_append_only",
        "Route insertion": "route_insertion",
        "Local search": "local_search",
    }
    if "Enhanced local search" in summaries:
        method_columns["Enhanced local search"] = "enhanced_local_search"
    rows: list[dict[str, object]] = []
    for key, label, unit in metrics:
        row: dict[str, object] = {"metric": label, "unit": unit}
        values = {}
        for method_name, column_name in method_columns.items():
            raw_value = summaries[method_name].get(key)
            value = float(raw_value) if raw_value is not None else None
            row[column_name] = value
            if value is not None:
                values[method_name] = value
        row["local_search_minus_greedy"] = (
            values["Local search"] - values["Greedy append-only"]
            if "Local search" in values and "Greedy append-only" in values else ""
        )
        if "Enhanced local search" in method_columns:
            row["enhanced_local_search_minus_greedy"] = (
                values["Enhanced local search"] - values["Greedy append-only"]
                if "Enhanced local search" in values and "Greedy append-only" in values else ""
            )
        if key == "total_travel_minutes_including_depot_returns" and values:
            greedy_value = values.get("Greedy append-only", 0.0)
            best_method = min(values, key=values.get)
            row["travel_reduction_percent_vs_greedy"] = (
                (greedy_value - values[best_method]) / greedy_value * 100
                if greedy_value else 0.0
            )
        else:
            row["travel_reduction_percent_vs_greedy"] = ""
        rows.append(row)

    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = list(summaries.keys())
    short_label_map = {
        "Greedy append-only": "Greedy",
        "Route insertion": "Insertion",
        "Local search": "Local search",
        "Enhanced local search": "Enhanced local search",
    }
    short_labels = [short_label_map[label] for label in labels]

    
    travel_values = [
        float(summaries[label]["total_travel_minutes_including_depot_returns"])
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(short_labels, travel_values)
    ax.set_title("Estimated total travel time")
    ax.set_ylabel("Minutes (including return to depot)")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.0f")
    fig.tight_layout()
    travel_figure = FIGURES / "travel_time_comparison.png"
    fig.savefig(travel_figure, dpi=180)
    plt.close(fig)

    coverage_values = [
        float(summaries[label]["active_task_coverage"]) * 100
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(short_labels, coverage_values)
    ax.set_title("Active-task coverage")
    ax.set_ylabel("Tasks assigned (% of non-cancelled tasks)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.2f%%")
    fig.tight_layout()
    coverage_figure = FIGURES / "task_coverage_comparison.png"
    fig.savefig(coverage_figure, dpi=180)
    plt.close(fig)

    route_values = [
        int(summaries[label]["technician_routes_used"])
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(short_labels, route_values)
    ax.set_title("Technician routes used")
    ax.set_ylabel("Number of technician-day routes")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.0f")
    fig.tight_layout()
    routes_figure = FIGURES / "routes_used_comparison.png"
    fig.savefig(routes_figure, dpi=180)
    plt.close(fig)

    greedy_travel = float(summaries["Greedy append-only"]["total_travel_minutes_including_depot_returns"])
    insertion_travel = float(summaries["Route insertion"]["total_travel_minutes_including_depot_returns"])
    local_search_travel = float(summaries["Local search"]["total_travel_minutes_including_depot_returns"])
    best_label = min(
        labels,
        key=lambda label: float(summaries[label]["total_travel_minutes_including_depot_returns"]),
    )
    best_travel = float(summaries[best_label]["total_travel_minutes_including_depot_returns"])
    reduction_vs_insertion = insertion_travel - best_travel
    reduction_vs_greedy = greedy_travel - best_travel
    reduction_vs_greedy_pct = (
        reduction_vs_greedy / greedy_travel * 100 if greedy_travel else 0.0
    )

    print("SCHEDULER COMPARISON")
    print("====================")
    print(f"Greedy travel time:       {greedy_travel:,.2f} minutes")
    print(f"Insertion travel time:    {insertion_travel:,.2f} minutes")
    print(f"Local-search travel time: {local_search_travel:,.2f} minutes")
    print(f"Best method ({best_label}) travel time: {best_travel:,.2f} minutes")
    print(
        f"Best-method reduction vs greedy: {reduction_vs_greedy:,.2f} minutes "
        f"({reduction_vs_greedy_pct:.2f}%)"
    )
    print(
        f"{best_label} reduction vs insertion: {reduction_vs_insertion:,.2f} minutes "
        f"({(reduction_vs_insertion / insertion_travel * 100) if insertion_travel else 0.0:.2f}%)"
    )
    print("\nPer-method results:")
    for label in labels:
        item = summaries[label]
        print(
            f"  {label}: assigned={int(item['assigned_tasks']):,}, "
            f"coverage={float(item['active_task_coverage']):.2%}, "
            f"unassigned={int(item['unassigned_tasks']):,}, "
            f"routes={int(item['technician_routes_used']):,}, "
            f"travel={float(item['total_travel_minutes_including_depot_returns']):,.2f} min"
        )
    print("\nCreated local files:")
    print(f"- {CSV_OUTPUT.relative_to(ROOT)}")
    print(f"- {travel_figure.relative_to(ROOT)}")
    print(f"- {coverage_figure.relative_to(ROOT)}")
    print(f"- {routes_figure.relative_to(ROOT)}")
    print("\nReminder: report only after both scheduler validators pass.")


if __name__ == "__main__":
    main()
