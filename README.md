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

The paper describes workday technicians as technician-day entities: it states that a workday technician is generated for each workday for each technician. The paper's reported instance summary gives 720 workday technicians, whereas this downloaded archive has 1,430 technician records. This discrepancy is unresolved and must be investigated rather than silently reconciled. The archive's technician time-window dates will be summarized to understand the rows before claims about technician counts are made. The paper reports 2,677 tasks, 31 skills and a five-day horizon.

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

The preview of `TDCData.txt` confirms that it contains technician records with an ID, a list of skills, a working time window, and an investment-option flag. The small investment file describes settings such as skill investment, technician investment, overtime investment and digitization. This preview does **not** yet show the task records or travel-time structure.

Run these commands from the repository root:

```powershell
python src/inspect_dataset_zip.py
python src/inspect_tdc_text.py
python src/summarize_tdc_structure.py
```

The structure-summary script reads the main text file line by line from inside the ZIP. It reports recognized section headings, approximate line counts and a few example records per section without extracting the large file or loading it all into memory. Inspect its output before writing a parser or training any model.
## Dataset validation: current findings

The initial archive scan found these section counts in `TDCData.txt`:

- `Technicians:` 1,430 data lines
- `Tasks:` 2,677 data lines
- `Travel time:` 14,823,979 data lines
- Total main-file lines: 14,828,089

Observed fields in the sample technician/task records include technician IDs, technician skills, working time windows, task IDs, required skills, duration, task time windows, priority, penalty, created/modified timestamps, cancellation status, and relevant technician IDs. Travel-time examples use a key such as `task1000task1: 186.0`.

These counts and patterns come from a preliminary scan. Before parsing, inspect complete sample records because fields can vary (for example, some sample lines spell the time-window label as `timewindow` without a space) and the travel-time key semantics still need verification.

Run:

```powershell
python src/preview_tdc_records.py
```

This prints three technician examples, three task examples and eight travel-time entries from the ZIP without extracting the 366 MB file. The output will help us define and validate the parser. Do not treat the travel-time values as minutes until the source documentation confirms their units.

## Parse technician and task records

After reviewing the sample records, run from the repository root:

```powershell
python src/parse_tdc_records.py
```

The parser streams the ZIP and writes local outputs to:

- `data/processed/technicians.csv`
- `data/processed/tasks.csv`

It validates record formats and duplicate IDs. If a row does not match, it reports the first parsing issues and deliberately does not write partial CSV files. Travel-time rows are not expanded in this step because that section contains nearly 14.8 million entries. The raw ZIP and generated processed files are excluded from Git.

## Data semantics investigation

After running the parser and validating the CSVs, pull the latest repository and run:

```powershell
git pull origin main
python src/investigate_dataset_semantics.py
```

This diagnostic reports technician records by time-window date, the full skill vocabulary and any skills that occur on only one side (technician skills or task requirements), all tasks with an empty relevant-technician list, and whether the referenced technician IDs exist in the technician table. It also summarizes cancellation timestamps to inform whether cancellation prediction can be designed without target leakage.

The associated paper explicitly states that travel times are measured in minutes. The dataset's observed 1,430 technician rows still do not match the paper's summary of 720 workday technicians; this remains an open data-version/representation question pending the diagnostic output and further inspection.

## Travel-time validation

The streaming scan of the travel-time section reported:

- Total classified entries: 14,823,979
- Task-to-task entries: 7,166,329 (`2677 × 2677`)
- Task-to-technician entries: 3,828,110 (`2677 × 1430`)
- Technician-to-task entries: 3,828,110 (`1430 × 2677`)
- Technician-to-technician entries: 1,430, all observed as technician self-pairs
- Unrecognized entries: 0
- Negative values: 0
- Invalid typed ID references: 0

The earlier count of 4,107 self-pairs includes 2,677 task self-pairs plus 1,430 technician self-pairs; it does not indicate 4,107 technician-to-technician relationships.

### Build travel-time matrices

Run from the repository root after parsing the CSV files:

```powershell
python src/build_travel_matrices.py
```

The script streams the source archive and builds three local NumPy files in `data/processed/`:

- `task_to_task.npy` (2,677 × 2,677)
- `task_to_technician.npy` (2,677 × 1,430)
- `technician_to_task.npy` (1,430 × 2,677)

It uses NaN placeholders to verify that every expected pair appears exactly once. It publishes the final matrix files only when category counts, ID references, duplicate checks, missing-pair checks, and the technician self-pair checks pass. The matrix files are generated locally and excluded from Git.

## First scheduling baseline

After the travel matrices have passed validation, run the deterministic greedy baseline:

```powershell
python -m py_compile src/run_greedy_scheduler.py
python src/run_greedy_scheduler.py
```

The scheduler:
- Excludes tasks already marked cancelled in the source.
- Uses the task's source-provided relevant-technician ID list as the candidate set.
- Orders tasks by numeric priority, higher penalty, earlier window end, then stable ID.
- Appends each accepted task to one technician-day route if task and shift time windows remain feasible, including inbound travel, service duration and return travel.
- Writes a status record for every task and a route table for assigned tasks.

Outputs (local, Git-ignored):
- `data/processed/baseline_task_status.csv`
- `data/processed/baseline_routes.csv`
- `data/processed/baseline_summary.json`

This is a deterministic, append-only greedy heuristic. It is not an ML model and not guaranteed to be optimal. Unassigned tasks must be interpreted as tasks this particular heuristic could not schedule under its rules, not proof that no feasible schedule exists. The baseline results are a reference point for later comparisons.

## Evaluation plan

The final evaluation depends on the validated data and selected research question. Possible scheduling measures include estimated travel time, unserved tasks, task coverage, time-window feasibility and workload balance. If a prediction task is supported by the source data, we will evaluate it using an appropriate held-out metric. Results will be compared with a clearly defined baseline and explained with plots and written interpretation.

## Academic deliverables

The course submission requires a project report, presentation, code, openly available data-source links and a help file. The report should include each member's contribution and relevant page numbers on its first page, explain every result visualization and end with evidence-based recommendations. Code and code screenshots should be kept out of the report.

## Research source

Gamst, M. and Pisinger, D. *Decision Support for the Technician Routing and Scheduling Problem*. Networks, 83(1), 169–196. https://doi.org/10.1002/net.22188  
Dataset and results: https://zenodo.org/records/7904015

## Team

Group 5 — add all five members' names and enrollment numbers here once confirmed.
