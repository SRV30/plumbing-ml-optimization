from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "25920036" / "code" / "ml_subratdang_25920036.ipynb"

st.set_page_config(page_title="25920036 | Water-Leak Detection", page_icon="💧", layout="wide")
st.title("Water-Leak Detection")
st.caption("Enrollment number: 25920036")
st.write(
    "This module uses accelerometer signals and a Random Forest classifier to label "
    "signal windows as No Leak (0) or Orifice Leak (1). The repository currently provides "
    "a notebook workflow rather than a standalone Streamlit application."
)
a, b, c = st.columns(3)
a.metric("Window size", "25,600 samples")
b.metric("Signal classes", "2")
c.metric("Original recordings", "2")
st.warning(
    "The notebook's current 70 windows come from only two original recordings, one per class. "
    "Perfect exploratory scores do not establish performance on independent unseen recordings."
)
if NOTEBOOK.exists():
    st.success(f"Notebook found: {NOTEBOOK.relative_to(ROOT)}")
else:
    st.error("The source notebook is missing from this checkout.")
st.markdown(
    "[Open the water-leak detection notebook on GitHub](https://github.com/SRV30/plumbing-ml-optimization/blob/main/25920036/code/ml_subratdang_25920036.ipynb)"
)
st.subheader("Workflow")
st.markdown(
    "1. Load the combined leak-signal CSV in the notebook.\n"
    "2. Segment the signals into non-overlapping windows.\n"
    "3. Extract signal features and train the Random Forest classifier.\n"
    "4. Review classification metrics, confusion matrix, ROC curve, and feature importance."
)
