# Group 5 — Plumbing ML Project

Integrated Streamlit portal for five ML/optimization modules.

## Modules

- **25920036:** Water-Leak Detection
- **26535001:** Plumbing Demand Forecasting
- **26535016:** Customer Feedback Analysis
- **26535023:** Strategic Resource Optimization
- **26535024:** Technician Scheduling & Duration Prediction

## Setup and Run (Windows)

**Requirements:** [Python](https://www.python.org/downloads/) and
[Git](https://git-scm.com/downloads).

``` bash
git clone https://github.com/SRV30/plumbing-ml-optimization.git
cd plumbing-ml-optimization

py -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r .\combine\requirements.txt

python -m streamlit run .\combine\app.py
```

Open the local URL shown in the terminal (usually
`http://localhost:8501`). Keep the terminal open while using the app;
press **Ctrl+C** to stop it.

If you already cloned the repository, run `git pull` from the project
folder before reinstalling dependencies if needed.

## Notebooks

The Water-Leak Detection and Strategic Resource Optimization modules
include notebook-based experiments. Open their `.ipynb` files in Jupyter
or Google Colab and run cells in order. Colab notebooks may require
file-path or data-upload adjustments for local execution.

## Notes

- Demand forecasting uses synthetic data.
- Technician scheduling uses telecommunications field-service benchmark
  data, not real plumbing records.
- Leak-detection results use windows from only two original recordings.
- Resource-optimization financial results are estimates based on
  assumptions, not verified savings.
- The five modules are complementary prototypes, not a validated
  end-to-end production system.

## Repository

[Open the project on
GitHub](https://github.com/SRV30/plumbing-ml-optimization)
