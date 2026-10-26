from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "26535023" / "Strategic_Resource_Optimization_ML.ipynb"

st.set_page_config(page_title="26535023 | Strategic Resource Optimization", page_icon="📈", layout="wide")
st.title("Strategic Resource Optimization")
st.caption("Enrollment number: 26535023")
st.write(
    "This module explores strategic resource allocation and skill-investment decisions "
    "using machine-learning estimates and optimization. The repository currently provides "
    "a Jupyter notebook and project report rather than a standalone Streamlit application."
)
st.subheader("Module artifacts")
st.markdown(
    "- [Open the optimization notebook on GitHub](https://github.com/SRV30/plumbing-ml-optimization/blob/main/26535023/Strategic_Resource_Optimization_ML.ipynb)\n"
    "- [Open the project report](https://github.com/SRV30/plumbing-ml-optimization/blob/main/26535023/Mr_Rooter_Plumbing_ML_Project_Report.docx)\n"
    "- [Open the presentation](https://github.com/SRV30/plumbing-ml-optimization/blob/main/26535023/Mr_Rooter_Plumbing_ML_Presentation.pptx)"
)
if NOTEBOOK.exists():
    st.success(f"Notebook found: {NOTEBOOK.relative_to(ROOT)}")
else:
    st.error("The source notebook is missing from this checkout.")
st.info(
    "Run the notebook to reproduce the experiments. This overview page does not rerun "
    "the optimization or claim results beyond the notebook and report."
)
