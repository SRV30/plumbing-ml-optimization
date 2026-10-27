# Group 5 — ML for Plumbing Service Operations

This repository combines five student ML modules and a shared Streamlit portal.

## Project modules

| Enrollment | Module | Main workflow |
|---|---|---|
| 25920036 | Water-leak detection | Notebook-based accelerometer signal classification |
| 26535001 | Plumbing demand forecasting | Forecasting pipeline and Streamlit interface |
| 26535016 | Customer feedback analysis | Text classification and Streamlit interface |
| 26535023 | Strategic resource optimization | Notebook-based resource/skill investment analysis |
| 26535024 | Technician scheduling and optimization | Field-service scheduling experiments and interface |

## Launch the integrated application

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r combine/requirements.txt
python -m streamlit run combine/app.py
```

Individual modules may have additional dependencies and data requirements. Check their local README/help files before running a module separately. Notebook-based modules must be run from their notebooks to reproduce the experiments.

## Data and evaluation notes

- Demand-forecasting data is synthetic; its metrics should not be represented as observed business performance.
- The water-leak workflow currently derives 70 windows from two original recordings. Splitting windows from the same recording across train/test can inflate scores; independent recordings are needed for a strong generalization claim.
- Technician scheduling uses a telecommunications field-service benchmark, not actual plumbing-company records.
- Resource-optimization results depend on the assumptions and input data defined in the notebook. Estimated savings are not measured savings unless validated against real operations.

## Reproducibility and attribution

See [CODE_PROVENANCE.md](CODE_PROVENANCE.md) for known data/research sources and the checks needed to trace adapted code, notebooks, model artifacts, and licenses. Keep required acknowledgements for third-party code and datasets. Common ML algorithms do not establish originality by themselves; the actual implementation and its source history must be reviewed.

## Feedback module tests

From the feedback module directory:

```bash
cd 26535016
python -m unittest discover -s tests
```

Run module-level and integrated workflows after changing dependencies, model training, or data processing.
