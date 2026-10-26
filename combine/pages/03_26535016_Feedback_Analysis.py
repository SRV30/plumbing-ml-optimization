from pathlib import Path
import os
import runpy
import sys
import streamlit as st

MODULE_DIR = Path(__file__).resolve().parents[2] / "26535016"
APP = MODULE_DIR / "feedback_app.py"

if not APP.exists():
    st.error(f"Feedback analysis app was not found: {APP}")
    st.stop()

sys.path.insert(0, str(MODULE_DIR))
old_cwd = Path.cwd()
try:
    os.chdir(MODULE_DIR)
    runpy.run_path(str(APP), run_name="__main__")
finally:
    os.chdir(old_cwd)
