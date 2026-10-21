from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"

SUMMARY_PATHS = {
    "Greedy append-only": DATA / "baseline_summary.json",
    "Route insertion": DATA / "insertion_summary.json",
    "Local search": DATA / "local_search_summary.json",
}

st.set_page_config(
    page_title="Field-Service Scheduling Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Field-Service Scheduling Optimizer")
st.caption("Offline dashboard for the three scheduling experiments in this project.")

st.warning(
    "Dataset limitation: these experiments use telecommunications field-service data, "
    "not real plumbing-company records. The results are a research prototype and are "
    "not evidence of actual savings for a plumbing business.",
    icon="⚠️",
)

missing = [path for path in SUMMARY_PATHS.values() if not path.exists()]
if missing:
    st.error("One or more saved scheduler summaries are missing.")
    st.write("The dashboard reads saved results and does not run the slow scheduling algorithm.")
    st.write("Missing files:")
    for path in missing:
        st.code(str(path.relative_to(ROOT)))
    st.markdown("From the project root, check that these experiments have already been run:")
    st.code(
        "python src/run_greedy_scheduler.py\n"
        "python src/run_insertion_scheduler.py\n"
        "python src/run_local_search_scheduler.py\n"
        "python src/compare_schedulers.py",
        language="powershell",
    )
    st.info(
        "Each scheduler must be validated before its results are reported. "
        "The local-search result shown in this project uses the bounded 100-move run."
    )
    st.stop()

try:
    summaries = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in SUMMARY_PATHS.items()
    }
    required_keys = {
        "assigned_tasks",
        "unassigned_tasks",
        "active_task_coverage",
        "technician_routes_used",
        "total_travel_minutes_including_depot_returns",
    }
    for label, summary in summaries.items():
        absent = required_keys.difference(summary)
        if absent:
            raise ValueError(f"{label} summary is missing keys: {', '.join(sorted(absent))}")
except (OSError, json.JSONDecodeError, ValueError) as error:
    st.error(f"Unable to read a scheduler summary: {error}")
    st.stop()

travel_key = "total_travel_minutes_including_depot_returns"
method_order = ["Greedy append-only", "Route insertion", "Local search"]
records: list[dict[str, object]] = []

for label in method_order:
    item = summaries[label]
    assigned = int(item["assigned_tasks"])
    unassigned = int(item["unassigned_tasks"])
    active = int(item.get("active_tasks", assigned + unassigned))
    records.append(
        {
            "Scheduler": label,
            "Tasks assigned": assigned,
            "Active tasks": active,
            "Coverage (%)": float(item["active_task_coverage"]) * 100,
            "Tasks unassigned": unassigned,
            "Routes used": int(item["technician_routes_used"]),
            "Travel (min)": float(item[travel_key]),
            "Service (min)": int(item.get("total_service_minutes", 0)),
        }
    )

comparison = pd.DataFrame(records)
greedy = summaries["Greedy append-only"]
insertion = summaries["Route insertion"]
local_search = summaries["Local search"]

greedy_travel = float(greedy[travel_key])
insertion_travel = float(insertion[travel_key])
local_travel = float(local_search[travel_key])
saved_vs_greedy = greedy_travel - local_travel
saved_vs_insertion = insertion_travel - local_travel
saved_pct = (saved_vs_greedy / greedy_travel * 100) if greedy_travel else 0.0
insertion_saved_pct = (
    saved_vs_insertion / insertion_travel * 100 if insertion_travel else 0.0
)

tab_overview, tab_routes, tab_ml = st.tabs(
    ["Scheduling overview", "Route explorer", "ML experiments and limitations"]
)

with tab_overview:
    st.subheader("Current best scheduling result")
    active_tasks = int(local_search.get("active_tasks", 2233))
    assigned_tasks = int(local_search["assigned_tasks"])
    unassigned_tasks = int(local_search["unassigned_tasks"])
    routes_used = int(local_search["technician_routes_used"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated travel", f"{local_travel:,.0f} min")
    m2.metric(
        "Travel reduction vs greedy",
        f"{saved_pct:.2f}%",
        delta=f"{saved_vs_greedy:,.0f} minutes saved",
        delta_color="off",
    )
    m3.metric("Active tasks assigned", f"{assigned_tasks:,} / {active_tasks:,}")
    m4.metric("Technician-day routes", f"{routes_used:,}")

    st.markdown(
        f"Local search also reduces travel by **{saved_vs_insertion:,.0f} minutes "
        f"({insertion_saved_pct:.2f}%)** compared with route insertion, with the same "
        f"number of assigned tasks."
    )
    st.caption(
        "Travel includes the final return-to-depot leg. Service duration is reported "
        "separately. Scheduling methods are heuristics, not guaranteed optimal solutions."
    )

    st.subheader("Compare all three methods")
    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Coverage (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Travel (min)": st.column_config.NumberColumn(format="%,.0f"),
            "Service (min)": st.column_config.NumberColumn(format="%,.0f"),
        },
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Estimated travel time")
        travel_chart = comparison.set_index("Scheduler")[["Travel (min)"]]
        st.bar_chart(travel_chart)
    with right:
        st.markdown("#### Active-task coverage")
        coverage_chart = comparison.set_index("Scheduler")[["Coverage (%)"]]
        st.bar_chart(coverage_chart)

    st.markdown("#### Saved comparison plots")
    plot_columns = st.columns(3)
    plots = [
        ("travel_time_comparison.png", "Travel time"),
        ("task_coverage_comparison.png", "Task coverage"),
        ("routes_used_comparison.png", "Routes used"),
    ]
    for column, (filename, label) in zip(plot_columns, plots):
        image_path = FIGURES / filename
        with column:
            if image_path.exists():
                st.image(str(image_path), caption=label, use_container_width=True)
            else:
                st.info(f"{filename} not found. Run python src/compare_schedulers.py to regenerate it.")

    st.download_button(
        "Download comparison table (CSV)",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="scheduler_comparison_dashboard.csv",
        mime="text/csv",
    )

with tab_routes:
    st.subheader("Inspect the saved local-search schedule")
    routes_path = DATA / "local_search_routes.csv"
    status_path = DATA / "local_search_task_status.csv"

    if routes_path.exists():
        try:
            routes_df = pd.read_csv(routes_path, keep_default_na=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read local-search route assignments: {error}")
        else:
            if routes_df.empty:
                st.info("The local-search route file is empty.")
            else:
                routes_df["technician_id"] = pd.to_numeric(
                    routes_df["technician_id"], errors="coerce"
                )
                route_ids = sorted(
                    int(value)
                    for value in routes_df["technician_id"].dropna().unique()
                )
                chosen_technician = st.selectbox(
                    "Select technician-day ID",
                    route_ids,
                    format_func=lambda value: f"Technician-day {value}",
                )
                one_route = routes_df[
                    routes_df["technician_id"] == chosen_technician
                ].copy()
                one_route["route_sequence"] = pd.to_numeric(
                    one_route["route_sequence"], errors="coerce"
                )
                one_route = one_route.sort_values("route_sequence")

                inbound = pd.to_numeric(
                    one_route["inbound_travel_minutes"], errors="coerce"
                ).fillna(0).sum()
                returns = pd.to_numeric(
                    one_route["return_travel_minutes"], errors="coerce"
                ).fillna(0).sum()
                r1, r2, r3 = st.columns(3)
                r1.metric("Tasks in selected route", f"{len(one_route):,}")
                r2.metric("Inbound travel total", f"{inbound:,.0f} min")
                r3.metric("Return travel", f"{returns:,.0f} min")

                display_columns = [
                    column
                    for column in [
                        "route_sequence",
                        "task_id",
                        "service_start",
                        "service_finish",
                        "duration_minutes",
                        "inbound_travel_minutes",
                        "return_travel_minutes",
                    ]
                    if column in one_route.columns
                ]
                st.dataframe(
                    one_route[display_columns],
                    hide_index=True,
                    use_container_width=True,
                )

                st.download_button(
                    "Download selected route (CSV)",
                    data=one_route[display_columns].to_csv(index=False).encode("utf-8"),
                    file_name=f"local_search_route_technician_{chosen_technician}.csv",
                    mime="text/csv",
                )
    else:
        st.info(
            "Route assignments are not present yet. The dashboard can still show the "
            "saved aggregate comparison, but this tab needs data/processed/local_search_routes.csv."
        )

    if status_path.exists():
        st.markdown("#### Task assignment status")
        try:
            status_df = pd.read_csv(status_path, keep_default_na=False)
            status_values = sorted(status_df["status"].astype(str).unique().tolist())
            selected_status = st.selectbox(
                "Filter tasks by status",
                status_values,
                key="task_status_filter",
            )
            filtered = status_df[status_df["status"].astype(str) == selected_status]
            status_columns = [
                column
                for column in [
                    "task_id",
                    "status",
                    "reason",
                    "technician_id",
                    "service_start",
                    "service_finish",
                ]
                if column in filtered.columns
            ]
            st.caption(f"Showing {len(filtered):,} tasks with status '{selected_status}'.")
            st.dataframe(
                filtered[status_columns].head(200),
                hide_index=True,
                use_container_width=True,
            )
            if len(filtered) > 200:
                st.caption("Only the first 200 matching rows are shown here.")
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, KeyError) as error:
            st.error(f"Could not load task status records: {error}")
    else:
        st.info("Task-status details are unavailable because the local status CSV is missing.")

with tab_ml:
    st.subheader("What the ML experiments currently show")
    st.write(
        "The ML work is exploratory. The best measured improvement so far comes from "
        "the scheduling heuristics above, not from an ML model changing the schedule."
    )

    ml_left, ml_right = st.columns(2)
    with ml_left:
        st.markdown("#### Task-duration regression")
        st.metric("Ridge model MAE", "25.83 minutes")
        st.metric("Median-baseline MAE", "44.55 minutes")
        st.write(
            "On the chronological holdout, Ridge predicted the recorded duration more "
            "accurately than the median baseline. However, the research formulation "
            "treats task duration as a scheduling input, so we still need to justify "
            "when this prediction would be available and useful in real operations."
        )
    with ml_right:
        st.markdown("#### Cancellation prediction")
        st.metric("Cancelled test tasks detected by Random Forest", "1 of 82")
        st.write(
            "The chronological test was weak for cancellation detection. This model "
            "should not be presented as a reliable cancellation-warning system."
        )

    st.warning(
        "Do not claim that ML has improved business performance yet. The current "
        "evidence supports a scheduling-method comparison and documents limitations "
        "of the first ML experiments.",
        icon="⚠️",
    )

with st.sidebar:
    st.header("About this prototype")
    st.write("**Project:** ML-Based Business Optimization for Plumbing Service Chains")
    st.write("**Scheduling methods:** greedy, route insertion, relocation local search")
    st.write("**Data:** telecommunications field-service instance")
    st.divider()
    st.markdown("**How to interpret the results**")
    st.write("Travel values are estimates from the research instance, not actual plumbing-company costs.")
    st.write("The dashboard reads locally generated result files; it does not rerun the schedulers.")
