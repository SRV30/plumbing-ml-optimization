from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Mr. Rooter | Integrated ML Suite",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.hero {
  padding: 1.6rem 1.8rem; border-radius: 18px;
  background: linear-gradient(120deg, #102a43 0%, #176b87 55%, #24a19c 100%);
  color: white; margin-bottom: 1.2rem;
}
.hero h1 {color: white; margin-bottom: .3rem;}
.hero p {color: #e6f6ff; font-size: 1.05rem;}
.member-card {
  border: 1px solid rgba(128,128,128,.3); border-radius: 14px;
  padding: 1rem 1.1rem; min-height: 190px;
}
.enrollment {font-size: .82rem; letter-spacing: .08em; font-weight: 700; color: #48a9a6;}
.status {font-size: .82rem; opacity: .8;}
</style>
<div class="hero">
  <h1>Mr. Rooter · Integrated ML Suite</h1>
  <p>One portal for the five team modules — each clearly identified by enrollment number.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Use the navigation menu on the left to open a member's module. The individual "
    "Streamlit applications are launched inside this shared portal; notebook-only modules "
    "have dedicated overview pages and links to their source notebooks."
)

st.subheader("Project modules")
cards = [
    ("25920036", "Water-Leak Detection", "Classifies accelerometer signal windows as leak / no leak.", "Notebook overview", "pages/01_25920036_Water_Leak_Detection.py"),
    ("26535001", "Plumbing Demand Forecasting", "Forecasts service demand, explores what-if scenarios, and supports staffing plans.", "Interactive app", "pages/02_26535001_Demand_Forecasting.py"),
    ("26535016", "Customer Feedback Analysis", "Analyzes review sentiment, complaint category, severity, and follow-up priority.", "Interactive app", "pages/03_26535016_Feedback_Analysis.py"),
    ("26535023", "Strategic Resource Optimization", "Explores resource/skill investment decisions using an optimization notebook.", "Notebook overview", "pages/04_26535023_Resource_Optimization.py"),
    ("26535024", "Technician Scheduling & Optimization", "Compares scheduling heuristics and duration/cancellation experiments.", "Interactive app", "pages/05_26535024_Technician_Scheduling.py"),
]
cols = st.columns(2)
for i, (enrollment, title, description, kind, page) in enumerate(cards):
    with cols[i % 2]:
        st.markdown(
            f'<div class="member-card"><div class="enrollment">ENROLLMENT {enrollment}</div>'
            f'<h3>{title}</h3><p>{description}</p><div class="status">{kind}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")

st.subheader("How to navigate")
st.markdown(
    "- Open a module from the **left sidebar**.\n"
    "- Use the enrollment number shown in each page title to identify its owner/module.\n"
    "- The two notebook-based modules are presented as project overview pages; their notebooks "
    "remain the source for executing those experiments."
)
