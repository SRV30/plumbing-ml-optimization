# ML-Based Business Optimization for Plumbing Service Chains

**Group 5 project** — inspired by plumbing service businesses such as Mr. Rooter Plumbing.

## Project objective

Investigate how machine learning and technician scheduling can help a plumbing service chain make better operational decisions. The project will use publicly available field-service technician-routing data as a starting point, subject to dataset validation.

The intended workflow is:

1. Validate and document an open dataset.
2. Explore technician, task, skill, location, duration, and scheduling fields available in the data.
3. Build a defensible baseline for technician assignment or routing.
4. Investigate whether the data supports a useful ML prediction task.
5. Compare approaches using measurable metrics and explain the business implications.

> **Scope note:** The candidate dataset describes telecommunications field service, not plumbing operations. Any adaptation to plumbing will be treated as a research prototype, and the report will state this limitation clearly. We will not claim a reduction in costs or delays unless experiments support it.

## Dataset

**Candidate source:** Gamst, M. and Pisinger, D., *Decision Support for the Technician Routing and Scheduling Problem* (Zenodo).

- Record: https://zenodo.org/records/7904015
- Candidate file for initial inspection: `TDCData_instance.zip`
- Other record files will be considered only if needed.

Download the data from Zenodo yourself and store it under `data/raw/`. Do not commit the large raw dataset to GitHub. Keep the downloaded originals unchanged, and document any preprocessing steps.

## Tools

- Python
- pandas and NumPy for data handling
- Matplotlib for plots
- scikit-learn for ML models if the validated fields support a prediction task
- An optimization library if needed for technician scheduling
- VS Code and Google Colab

Exact dependencies will be recorded in `requirements.txt` once the first implementation is defined.

## Planned repository structure

```text
plumbing-ml-optimization/
├── data/
│   ├── raw/              # Downloaded source data (not committed)
│   └── processed/        # Generated / cleaned data (not committed)
├── notebooks/
│   └── 01_dataset_analysis.ipynb
├── src/
├── results/
│   └── figures/
├── report/
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started

The first milestone is dataset validation, not model training.

1. Download `TDCData_instance.zip` from the Zenodo record above.
2. Put it in `data/raw/`.
3. Inspect the ZIP contents and file formats.
4. Record the file list, fields, units, missing values, and any constraints.
5. Decide which research question is actually supported by the data.

Once the structure is verified, setup commands and reproducible analysis instructions will be added here.

## Evaluation plan

The final evaluation will depend on the validated data and selected research question. Possible scheduling measures include travel time, waiting / lateness, workload balance, or unassigned tasks. If a prediction task is supported, an appropriate held-out evaluation metric will also be reported. We will compare against a clearly defined baseline and explain the results with plots and written interpretation.

## Academic deliverables

The course submission requires a project report, presentation, code, openly available data-source links, and a help file. The report should include each member's contribution and relevant page numbers on its first page, explain every result visualization, and end with evidence-based recommendations. Code and code screenshots should be kept out of the report.

## Research source

Gamst, M. and Pisinger, D. *Decision Support for the Technician Routing and Scheduling Problem*. Dataset and results: https://zenodo.org/records/7904015

## Team

Group 5 — add all five members' names and enrollment numbers here once confirmed.
