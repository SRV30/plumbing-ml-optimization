"""Compare validated greedy and route-insertion scheduler summaries.

Run from the repository root:
    python src/compare_schedulers.py

Reads the JSON summaries written by the two scheduler scripts and produces:
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
}
CSV_OUTPUT = DATA / "scheduler_comparison.csv"


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
        summaries[label] = json.loads(path.read_text(encoding="utf-8"))

    metrics = [
        ("assigned_tasks", "Tasks assigned", "count"),
        ("unassigned_tasks", "Tasks unassigned", "count"),
        ("active_task_coverage", "Active-task coverage", "fraction"),
        ("technician_routes_used", "Technician routes used", "count"),
        ("total_travel_minutes_including_depot_returns", "Travel time including depot returns", "minutes"),
        ("total_service_minutes", "Scheduled service time", "minutes"),
    ]

    rows: list[dict[str, object]] = []
    for key, label, unit in metrics:
        greedy_value = float(summaries["Greedy append-only"][key])
        insertion_value = float(summaries["Route insertion"][key])
        delta = insertion_value - greedy_value
        if key == "total_travel_minutes_including_depot_returns":
            reduction_pct = (
                (greedy_value - insertion_value) / greedy_value * 100
                if greedy_value else 0.0
            )
        else:
            reduction_pct = ""
        rows.append({
            "metric": label,
            "unit": unit,
            "greedy_append_only": greedy_value,
            "route_insertion": insertion_value,
            "insertion_minus_greedy": delta,
            "travel_reduction_percent_if_applicable": reduction_pct,
        })

    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = list(SUMMARY_FILES.keys())
    short_labels = ["Greedy", "Insertion"]

    # Separate figures make each comparison readable in the final report.
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

    greedy_travel = travel_values[0]
    insertion_travel = travel_values[1]
    reduction = greedy_travel - insertion_travel
    reduction_pct = (reduction / greedy_travel * 100) if greedy_travel else 0.0

    print("SCHEDULER COMPARISON")
    print("====================")
    print(f"Greedy travel time:    {greedy_travel:,.2f} minutes")
    print(f"Insertion travel time: {insertion_travel:,.2f} minutes")
    print(f"Travel-time reduction: {reduction:,.2f} minutes ({reduction_pct:.2f}%)")
    print(
        "Assigned tasks:       "
        f"{int(summaries['Greedy append-only']['assigned_tasks']):,} (greedy), "
        f"{int(summaries['Route insertion']['assigned_tasks']):,} (insertion)"
    )
    print(
        "Coverage:             "
        f"{float(summaries['Greedy append-only']['active_task_coverage']):.2%} (greedy), "
        f"{float(summaries['Route insertion']['active_task_coverage']):.2%} (insertion)"
    )
    print(
        "Routes used:          "
        f"{int(summaries['Greedy append-only']['technician_routes_used']):,} (greedy), "
        f"{int(summaries['Route insertion']['technician_routes_used']):,} (insertion)"
    )
    print("\nCreated local files:")
    print(f"- {CSV_OUTPUT.relative_to(ROOT)}")
    print(f"- {travel_figure.relative_to(ROOT)}")
    print(f"- {coverage_figure.relative_to(ROOT)}")
    print(f"- {routes_figure.relative_to(ROOT)}")
    print("\nReminder: report only after both scheduler validators pass.")


if __name__ == "__main__":
    main()
