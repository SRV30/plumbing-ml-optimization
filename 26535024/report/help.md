# Project Help and Reproduction Guide

**Project:** ML-Based Business Optimization for Plumbing Service Chains  
**Module:** 26535024 — Technician Routing, Scheduling, and Duration Prediction  
**Dataset:** TDC Net telecommunications field-service benchmark (not plumbing-company data)

## 1. Important interpretation

This repository is an academic research prototype. The source data describes telecommunications field service, not real plumbing operations. Do not present the reported travel minutes as measured fuel, payroll, cost, or customer-service savings for a plumbing business.

The downloaded archive's parsed technician/skill counts (1,430 technician-day records and 32 observed skills) differ from the paper's reported counts (720 workday technicians and 31 skills). This discrepancy is documented but unresolved.

## 2. Environment setup

Use Python 3.10 or a compatible version supported by the dependencies. From the repository root, enter the module directory:

```powershell
cd 26535024
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use the environment according to your institution's machine policy or run the installed interpreter directly.

## 3. Prepare local data

The raw benchmark and generated processed data are intentionally excluded from Git because of their size. Follow the data-source and preprocessing instructions in `26535024/README.md`. Keep original source data unchanged and do not commit large raw data or private credentials.

The scheduling and dashboard scripts expect locally generated files under `26535024/data/processed/`, including task/technician tables and travel-time matrices. A fresh clone cannot reproduce task-level schedules until these inputs have been prepared.

## 4. Validate and compare schedulers

Run from `26535024/` after the source tables, matrices, and schedules exist:

```powershell
python -m compileall -q .\src
python .\src\validate_greedy_schedule.py
python .\src\validate_insertion_schedule.py
python .\src\validate_local_search_schedule.py
python .\src\validate_local_search_schedule.py --schedule enhanced
python .\src\compare_schedulers.py
```

The enhanced local-search validator should report 2,677 source task rows, 2,677 status rows, 2,209 assigned route rows, 637 route groups, 28,261 checks, and zero validation errors for the verified run documented on 26 October 2026. This is an observed result, not a guarantee that a fresh environment or rerun will produce identical output. Validation establishes consistency with the implemented rules, not global optimality.

The baseline comparison script compares greedy append-only, route insertion, and relocation local search. The enhanced route-order result is reported separately in the final report and this guide.

## 5. Run the enhanced route-order pass

From `26535024/`, with the local scheduling inputs available:

```powershell
python -m py_compile .\src\improve_local_search_moves.py
python .\src\improve_local_search_moves.py
python .\src\validate_local_search_schedule.py --schedule enhanced
```

The script uses feasible intra-route swaps and 2-opt reversals that reduce travel. It preserves the assigned task set and is a bounded heuristic, not an exact optimizer. It writes its enhanced route/status/summary files to `data/processed/`, which is Git-ignored.

## 6. Run the dashboard

From `26535024/`:

```powershell
python -m py_compile .\app.py
streamlit run .\app.py
```

Open the local address printed by Streamlit (usually `http://localhost:8501`). The dashboard displays aggregate baseline and enhanced results. If prepared local schedule files are present, it can also inspect the saved enhanced route order; otherwise the route explorer explains that local task-level files are required. The dashboard does not rerun scheduling algorithms automatically.

## 7. Main reported results

| Method | Assigned tasks | Routes | Estimated travel including depot returns |
|---|---:|---:|---:|
| Greedy append-only | 2,209 | 748 | 17,783 min |
| Route insertion | 2,209 | 643 | 14,627 min |
| Relocation local search | 2,209 | 637 | 14,239 min |
| Enhanced route-order local search | 2,209 | 637 | 14,174 min |

The enhanced result reduces estimated travel by 3,609 minutes (20.30%) relative to greedy scheduling and by 453 minutes (3.10%) relative to route insertion. Relative to its 14,239-minute starting schedule, the enhanced route-order pass reduces travel by 65 minutes (0.46%). The verified enhanced schedule passed 28,261 implemented validation checks with zero errors.

The ML duration and cancellation experiments are exploratory. Duration-model counterfactuals expose trade-offs between prediction error, planned travel, coverage, and replay feasibility; they do not establish that ML improves operational outcomes.

## 8. Submission checklist

- Fill in institution/course, faculty guide, submission date, and all five group members' names, enrollment numbers, and verified contributions in `report/final_report.md`.
- Export the report to the required format and add final page references after pagination.
- Prepare the course presentation and include any required source-code screenshots only in the designated appendix/help materials, not the report body if prohibited by course instructions.
- Verify that the report, dashboard, and presentation consistently state the telecommunications-data limitation.
- Re-run validators after any schedule-changing code or data modification and retain the actual outputs.
- Do not claim global optimality, production readiness, or real plumbing-business savings from these experiments.
