from pathlib import Path
import os
import runpy
import sys
import streamlit as st

MODULE_DIR = Path(__file__).resolve().parents[2] / "26535001" / "Plumbing-Demand-Forecast-Model-main"
APP = MODULE_DIR / "app.py"

if not APP.exists():
    st.error(f"Demand forecasting app was not found: {APP}")
    st.stop()

sys.path.insert(0, str(MODULE_DIR))
old_cwd = Path.cwd()
try:
    os.chdir(MODULE_DIR)
    runpy.run_path(str(APP), run_name="__main__")
finally:
    os.chdir(old_cwd)
