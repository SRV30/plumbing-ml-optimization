from pathlib import Path
import runpy
import streamlit as st

APP = Path(__file__).resolve().parents[2] / "26535024" / "app.py"
if not APP.exists():
    st.error(f"Technician scheduling app was not found: {APP}")
    st.stop()

runpy.run_path(str(APP), run_name="__main__")
