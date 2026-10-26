from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "26535023" / "datasets"
TASKS_FILE = DATA / "Tasks.txt"
TECHNICIANS_FILE = DATA / "Technicians.txt"
CONFIG_FILE = DATA / "TDCData_investment.txt"

st.set_page_config(page_title="26535023 | Strategic Resource Optimization", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#176b87 58%,#24a19c); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0 0 .3rem 0;}
.hero p {color:#e6f6ff; margin:0; font-size:1rem;}
.metric-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:1rem;}
</style>
<div class="hero">
<h1>Strategic Resource Optimization</h1>
<p>Enrollment number: 26535023 · ML-based cancellation risk and budget allocation</p>
</div>
""", unsafe_allow_html=True)

def parse_config(path: Path):
    config = {"skill_cost": 175.0, "skill_groups": {}}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if "Skill investment cost:" in line:
            config["skill_cost"] = float(line.split(":", 1)[1].strip())
        elif "Skill investment group:" in line:
            match = re.search(r"Skill investment group:\s*(\d+):\s*Set\(\[(.*?)\]\)", line)
            if match:
                config["skill_groups"][int(match.group(1))] = [
                    s.strip().replace('"', "") for s in match.group(2).split(",") if s.strip()
                ]
    return config

def parse_tasks(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("Task id"):
            continue
        get = lambda pattern: re.search(pattern, line)
        tid = int(get(r"Task id (\d+)").group(1))
        skills_raw = re.findall(r"skills \[(.*?)\]", line)[0]
        skills = [s.strip().replace('"', "") for s in skills_raw.split(",") if s.strip()]
        tw = re.findall(r"time window \[(.*?), (.*?)\]", line)[0]
        tw_start, tw_end = datetime.fromisoformat(tw[0]), datetime.fromisoformat(tw[1])
        created = datetime.fromisoformat(get(r"created ([\d\-T:]+)").group(1))
        tech_match = get(r"relevant technician ids \[(.*?)\]")
        rows.append({
            "task_id": tid,
            "primary_skill": skills[0] if skills else "None",
            "skills": skills,
            "num_skills": len(skills),
            "duration": int(get(r"duration (\d+)").group(1)),
            "tw_duration": (tw_end - tw_start).total_seconds() / 60,
            "priority": int(get(r"priority (\d+)").group(1)),
            "penalty": float(get(r"penalty ([\d\.]+)").group(1)),
            "lead_time": (tw_start - created).total_seconds() / 3600,
            "num_techs": len(tech_match.group(1).split(",")) if tech_match and tech_match.group(1).strip() else 0,
            "cancelled": int(get(r"cancelled (true|false)").group(1) == "true"),
        })
    return pd.DataFrame(rows)

def parse_technicians(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("Technician id"):
            continue
        tid = int(re.search(r"Technician id (\d+)", line).group(1))
        skills_raw = re.findall(r"skills \[(.*?)\]", line)[0]
        skills = [s.strip().replace('"', "") for s in skills_raw.split(",") if s.strip()]
        rows.append({"tech_id": tid, "skills": skills, "can_invest": bool(re.search(r"investment option true", line))})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner="Loading records and fitting the risk model…")
def run_analysis(budget: float, unit_cost: float, savings_rate: float):
    tasks = parse_tasks(TASKS_FILE)
    technicians = parse_technicians(TECHNICIANS_FILE)
    mlb = MultiLabelBinarizer()
    skills = pd.DataFrame(mlb.fit_transform(tasks["skills"]), columns=[f"skill_{s}" for s in mlb.classes_])
    feature_cols = ["duration", "tw_duration", "priority", "penalty", "lead_time", "num_techs", "num_skills"]
    X = pd.concat([tasks[feature_cols], skills], axis=1)
    y = tasks["cancelled"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    tasks["predicted_risk"] = model.predict_proba(X)[:, 1]
    tasks["expected_loss"] = tasks["predicted_risk"] * tasks["penalty"]
    tasks["intervention_cost"] = unit_cost
    tasks["expected_saved_penalty"] = tasks["expected_loss"] * savings_rate
    result = linprog(
        -tasks["expected_saved_penalty"].to_numpy(),
        A_ub=[tasks["intervention_cost"].to_numpy()],
        b_ub=[budget],
        bounds=[(0, 1)] * len(tasks),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Budget optimization failed: {result.message}")
    tasks["optimization_fraction"] = result.x
    tasks["is_targeted"] = result.x > 0.5
    return tasks, technicians, auc, float(np.sum(result.x * tasks["intervention_cost"].to_numpy())), float(-result.fun), len(y_test)

if not all(p.exists() for p in [TASKS_FILE, TECHNICIANS_FILE, CONFIG_FILE]):
    st.error("Required dataset files were not found. Run the combined app from the repository checkout so the 26535023/datasets folder is available.")
    st.stop()

config = parse_config(CONFIG_FILE)
with st.sidebar:
    st.header("Scenario controls")
    budget = st.number_input("Investment budget ($)", min_value=500.0, max_value=500000.0, value=20000.0, step=2500.0)
    unit_cost = st.number_input("Cost per intervention ($)", min_value=1.0, max_value=5000.0, value=float(config["skill_cost"]), step=25.0)
    savings_percent = st.slider("Assumed penalty reduction", min_value=5, max_value=90, value=60, step=5)
    st.caption("Change a value to recalculate the scenario.")

try:
    tasks, technicians, auc, spent, saved, test_n = run_analysis(float(budget), float(unit_cost), savings_percent / 100)
except Exception as exc:
    st.error(f"Could not calculate the optimization scenario: {exc}")
    st.stop()

targeted = tasks[tasks["is_targeted"]].copy()
untargeted = tasks[~tasks["is_targeted"]].copy()
net_value = saved - spent
roi = (net_value / spent * 100) if spent else 0.0
risk_count = int((tasks["predicted_risk"] > 0.5).sum())

st.caption("Explore cancellation-risk estimates and compare budget-allocation scenarios.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Backlog tasks", f"{len(tasks):,}")
c2.metric("Technicians", f"{len(technicians):,}")
c3.metric("Test ROC-AUC", f"{auc:.3f}")
c4.metric("High-risk tasks", f"{risk_count:,}")

st.subheader("Budget allocation overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Budget used by optimizer", f"${spent:,.0f}", f"{spent / budget * 100:.1f}% of budget")
m2.metric("Estimated penalty reduction", f"${saved:,.0f}")
m3.metric("Estimated net value", f"${net_value:,.0f}")
m4.metric("Scenario ROI", f"{roi:,.1f}%")

left, right = st.columns(2)
with left:
    st.markdown("#### Investment budget vs. estimated savings")
    chart = pd.DataFrame({"Category": ["Budget allocated", "Estimated penalty reduction"], "Amount ($)": [spent, saved]})
    st.plotly_chart(px.bar(chart, x="Category", y="Amount ($)", text_auto=",.0f"), use_container_width=True)
with right:
    st.markdown("#### Targeted vs. other tasks")
    counts = pd.DataFrame({"Group": ["Targeted", "Not targeted"], "Tasks": [len(targeted), len(untargeted)]})
    st.plotly_chart(px.pie(counts, names="Group", values="Tasks", hole=0.55), use_container_width=True)

st.subheader("Cancellation-risk analysis")
left, right = st.columns(2)
with left:
    st.markdown("#### Predicted risk distribution")
    risk_plot = px.histogram(tasks, x="predicted_risk", color="is_targeted", nbins=25,
                             labels={"predicted_risk": "Predicted cancellation risk", "is_targeted": "Targeted"})
    st.plotly_chart(risk_plot, use_container_width=True)
with right:
    st.markdown("#### Most influential model features")
    feature_cols = ["duration", "tw_duration", "priority", "penalty", "lead_time", "num_techs", "num_skills"]
    mlb = MultiLabelBinarizer()
    skill_matrix = pd.DataFrame(mlb.fit_transform(tasks["skills"]), columns=[f"skill_{s}" for s in mlb.classes_])
    X = pd.concat([tasks[feature_cols], skill_matrix], axis=1)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    y = tasks["cancelled"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).nlargest(10, "Importance")
    st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h"), use_container_width=True)

st.subheader("Skill allocation")
skill_summary = targeted.groupby("primary_skill").agg(
    Tasks=("task_id", "count"),
    Capital=("intervention_cost", "sum"),
    Estimated_savings=("expected_saved_penalty", "sum"),
).reset_index()
if not skill_summary.empty:
    skill_summary["Estimated net value"] = skill_summary["Estimated_savings"] - skill_summary["Capital"]
    st.plotly_chart(px.bar(skill_summary.nlargest(10, "Estimated_savings"), x="primary_skill", y="Estimated_savings",
                           hover_data=["Tasks", "Capital", "Estimated net value"],
                           labels={"primary_skill": "Primary skill", "Estimated_savings": "Estimated penalty reduction ($)"}),
                   use_container_width=True)
    st.dataframe(skill_summary.sort_values("Estimated_savings", ascending=False), use_container_width=True, hide_index=True)
else:
    st.info("No tasks cross the optimizer's targeting threshold under these settings.")

st.subheader("Priority task review")
show_targeted = st.toggle("Show targeted tasks only", value=True)
review = targeted if show_targeted else tasks
review = review.sort_values(["expected_saved_penalty", "predicted_risk"], ascending=False)
st.dataframe(
    review[["task_id", "primary_skill", "priority", "duration", "penalty", "predicted_risk",
            "expected_saved_penalty", "is_targeted"]].head(100).rename(columns={
                "task_id": "Task ID", "primary_skill": "Primary skill", "priority": "Priority",
                "duration": "Duration (min)", "penalty": "Penalty ($)", "predicted_risk": "Predicted risk",
                "expected_saved_penalty": "Estimated penalty reduction ($)", "is_targeted": "Targeted",
            }),
    use_container_width=True,
    hide_index=True,
)
st.download_button("Download task-level scenario (CSV)", tasks.drop(columns=["skills"]).to_csv(index=False).encode("utf-8"),
                   file_name="26535023_resource_optimization_scenario.csv", mime="text/csv")

st.subheader("Module files")
st.markdown(
    "- [Optimization notebook](https://github.com/SRV30/plumbing-ml-optimization/blob/main/26535023/Strategic_Resource_Optimization_ML.ipynb)\n"
    "- [Project report](https://github.com/SRV30/plumbing-ml-optimization/blob/main/26535023/Mr_Rooter_Plumbing_ML_Project_Report.docx)\n"
    "- [Presentation](https://github.com/SRV30/plumbing-ml-optimization/blob/main/26535023/Mr_Rooter_Plumbing_ML_Presentation.pptx)"
)
st.caption("This dashboard displays model-based estimates under the selected assumptions; savings and ROI are projections, not measured business outcomes.")
