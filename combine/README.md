# Integrated ML Suite

A shared Streamlit portal for the five enrollment-number modules in the Group 5 plumbing-services ML project.

## Modules

- **25920036** — Water-Leak Detection (notebook overview page; notebook-based workflow)
- **26535001** — Plumbing Demand Forecasting (existing Streamlit app)
- **26535016** — Customer Feedback Analysis (existing Streamlit app)
- **26535023** — Strategic Resource Optimization (notebook overview page; notebook-based workflow)
- **26535024** — Technician Scheduling & Optimization (existing Streamlit app)

The existing apps are reused in place rather than copied, so their module data/model paths remain separate.

## Run locally

From the repository root:

```powershell
python -m pip install -r .\combine\requirements.txt
python -m streamlit run .\combine\app.py
```

If one module needs an additional dependency, install that module's own requirements as well. Demand forecasting uses the model and data files in `26535001/Plumbing-Demand-Forecast-Model-main`; feedback analysis uses files in `26535016`; scheduling uses generated files in `26535024`.

## Notes

- The two notebook-only modules have overview pages, not newly implemented prediction interfaces.
- Each module keeps its own dataset, model, and validation limitations.
- Run the portal from the repository root so all original module paths resolve.
