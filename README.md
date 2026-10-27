# Group 5 — ML for Plumbing Service Operations

This repository brings together five student project modules and an integrated Streamlit portal.

## Modules

| Enrollment | Module | Implementation |
|---|---|---|
| 25920036 | Water-leak detection | Notebook-based signal classification |
| 26535001 | Plumbing demand forecasting | LightGBM forecasting and Streamlit app |
| 26535016 | Customer feedback analysis | Text classification and Streamlit app |
| 26535023 | Strategic resource optimization | Notebook/optimization workflow |
| 26535024 | Technician scheduling and optimization | Field-service scheduling experiments |

## Run the integrated app

Use Python 3.10+ in a virtual environment. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r combine/requirements.txt
python -m streamlit run combine/app.py
```

Some modules also require their own dependencies and local datasets. Follow the module-specific README/help files before running an individual module.

## Originality and source attribution

Read [CODE_PROVENANCE.md](CODE_PROVENANCE.md) before submission. It records known dataset/research sources and the remaining attribution checks. Common algorithms do not by themselves prove copying, and formatting or renaming alone does not make copied code original. Keep licenses and citations for any third-party code that was adapted.

## Known limitations

- Demand forecasting uses synthetic plumbing-demand data; reported performance is not evidence of real-world business performance.
- The water-leak dataset currently derives 70 windows from only two original recordings, limiting claims about generalization.
- Technician scheduling uses a telecommunications field-service benchmark rather than plumbing-company records.
- Resource-optimization assumptions and scenario estimates must not be presented as measured savings without supporting experiments.

## Tests

Feedback helper unit tests can be run from the module directory:

```bash
cd 26535016
python -m unittest discover -s tests
```

A successful unit-test run does not establish that all applications or models have been end-to-end tested.
