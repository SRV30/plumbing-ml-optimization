# ML-Based Business Optimization for Plumbing Service Chains

**Group 5 project** — inspired by plumbing service businesses such as Mr. Rooter Plumbing.

## Project objective

Investigate how machine learning and technician scheduling can help a plumbing service chain make better operational decisions. The project uses publicly available field-service technician-routing data as a starting point, subject to dataset validation.

The intended workflow is:

1. Validate and document an open dataset.
2. Explore technician, task, skill, location, duration, and scheduling fields available in the data.
3. Build a clear baseline for technician assignment or routing.
4. Investigate whether the data supports a useful ML prediction task.
5. Compare approaches using measurable metrics and explain the business implications.

> **Scope note:** The candidate dataset describes telecommunications field service, not plumbing operations. Any adaptation to plumbing will be treated as a research prototype, and the report will state this limitation clearly. We will not claim a reduction in costs or delays unless experiments support it.

## Dataset

**Candidate source:** Gamst, M. and Pisinger, D., *Decision Support for the Technician Routing and Scheduling Problem* (Zenodo).

- Record: https://zenodo.org/records/7904015
- Initial archive: `TDCData_instance.zip`
- The archive contains text files, including `TDCData.txt` and `TDCData_investment.txt`. The main text file is large, so inspect it before loading it into memory.
- Research paper: https://doi.org/10.1002/net.22188

The paper describes the TDC Net instance as containing 720 workday technicians, 2,677 tasks, 31 skills and a five-day horizon. These are paper-reported instance facts; our next step is to verify how the downloaded files encode the data.

Download the data from Zenodo yourself and store it under `data/raw/`. Do not commit the large raw dataset to GitHub. Keep the downloaded originals unchanged, and document any preprocessing steps.

## Tools

- Python
- pandas and NumPy for data handling
- Matplotlib for plots
- scikit-learn for ML models if the validated fields support a prediction task
- An optimization library if needed for technician scheduling
- VS Code and Google Colab

Install the initial dependencies using:

```powershell
python -m pip install -r requirements.txt
```

## Planned repository structure

```text
plumbing-ml-optimization/
├── data/
│   ├── raw/              # Downloaded source data (not committed)
│   └── processed/        # Generated / cleaned data (not committed)
├── notebooks/
├── src/
│   ├── inspect_dataset_zip.py
│   └── inspect_tdc_text.py
├── results/
│   └── figures/
├── report/
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset validation: current step

The archive inspection has confirmed that the ZIP is readable and contains two text members:

- `TDCData.txt` (approximately 366.5 MB uncompressed)
- `TDCData_investment.txt` (approximately 1.3 KB uncompressed)

These are text files, not CSV/Excel files; that is expected. Do not assume a delimiter or load the main file with pandas until the file structure has been inspected.

Run from the repository root:

```powershell
python src/inspect_dataset_zip.py
python src/inspect_tdc_text.py
```

The second script previews a small number of lines directly from the ZIP without extracting the 366 MB text file. Review the sample to determine the encoding, field separators, headers and record structure before writing a parser.

## Evaluation plan

The final evaluation depends on the validated data and selected research question. Possible scheduling measures include estimated travel time, unserved tasks, task coverage, time-window feasibility and workload balance. If a prediction task is supported by the source data, we will evaluate it using an appropriate held-out metric. Results will be compared with a clearly defined baseline and explained with plots and written interpretation.

## Academic deliverables

The course submission requires a project report, presentation, code, openly available data-source links and a help file. The report should include each member's contribution and relevant page numbers on its first page, explain every result visualization and end with evidence-based recommendations. Code and code screenshots should be kept out of the report.

## Research source

Gamst, M. and Pisinger, D. *Decision Support for the Technician Routing and Scheduling Problem*. Networks, 83(1), 169–196. https://doi.org/10.1002/net.22188  
Dataset and results: https://zenodo.org/records/7904015

## Team

Group 5 — add all five members' names and enrollment numbers here once confirmed.
