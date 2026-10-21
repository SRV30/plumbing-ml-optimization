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

## Validate baseline scheduler output

After building the travel matrices and running the scheduler, validate the generated assignments:

```powershell
python -m py_compile src/validate_greedy_schedule.py
python src/validate_greedy_schedule.py
```

The validator checks that:
- every source task has exactly one status row;
- tasks marked cancelled are excluded and assigned tasks are not marked cancelled;
- assigned tasks use a technician listed in the source task's eligible-technician list;
- assigned route rows correspond exactly to tasks marked assigned;
- route sequence numbers, inbound travel values, service durations and time windows are consistent;
- technician shifts and final return travel are respected; and
- summary counts match the output tables.

A passing result validates internal consistency and the constraints included in the validator; it does not prove global optimality or that the heuristic finds every feasible assignment.

## Route-insertion scheduling experiment

After the greedy baseline has been generated and validated, run the alternative route-insertion heuristic:

```powershell
python -m py_compile src/run_insertion_scheduler.py
python src/run_insertion_scheduler.py
python -m py_compile src/validate_insertion_schedule.py
python src/validate_insertion_schedule.py
```

The insertion heuristic tries placing each eligible, non-cancelled task at different positions within a candidate technician-day route. It keeps an insertion only if the full route remains feasible and selects the feasible option with the lowest marginal travel time. This is a different heuristic, not an exact optimizer and not a Machine Learning model.

The insertion output has separate files so it will not overwrite the greedy baseline:
- `data/processed/insertion_task_status.csv`
- `data/processed/insertion_routes.csv`
- `data/processed/insertion_summary.json`

The validator checks the generated table and recorded scheduling constraints. Compare the two methods only after the insertion run passes validation. A higher task-coverage number or lower travel time should be interpreted against the same constraint definitions; neither alone proves the method is globally optimal.

## Compare the scheduling methods

After both scheduling outputs have been generated and both validators pass, run:

```powershell
python -m py_compile src/compare_schedulers.py
python src/compare_schedulers.py
```

The comparison script reads `baseline_summary.json` and `insertion_summary.json`, writes a side-by-side CSV, and creates separate PNG plots for travel time, task coverage and routes used. It does not modify either scheduler's outputs.

## Initial Machine Learning experiment: cancellation prediction

The parsed task table has a `cancelled` field (444 true, 2,233 false in the observed data). We will treat cancellation prediction as an exploratory ML candidate, not as a guaranteed final model.

Before running the experiment, note a serious data-leakage risk: all 444 cancelled tasks have a nonblank count of 444 in the `modified` column, while the 2,233 non-cancelled tasks have blank `modified` values. Because modification can be recorded after an outcome, the model deliberately excludes `modified`. It also excludes `task_id` and the target label from features.

Run:

```powershell
python -m py_compile src/train_cancellation_model.py
python src/train_cancellation_model.py
```

The script compares:
- a majority-class baseline;
- class-weighted logistic regression; and
- a class-weighted random forest.

It reports accuracy, balanced accuracy, cancellation precision/recall/F1, average precision, ROC-AUC and a confusion matrix, and saves JSON/CSV results plus ROC and precision–recall plots under `results/`.

This is a stratified random hold-out on one static telecommunications field-service dataset. Even a high score would not by itself establish production cancellation prediction or plumbing-business performance. We must interpret cancellation labels and feature availability at the intended prediction time, compare against the majority baseline, and report limitations honestly.

## Initial cancellation-model result (exploratory)

The first stratified random hold-out run used 2,141 training rows and 536 test rows. The cancelled class has 444 rows overall, compared with 2,233 non-cancelled rows.

Initial test metrics reported before fixing logistic-regression scaling:

| Model | Accuracy | Balanced accuracy | Cancellation precision | Cancellation recall | Cancellation F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.8340 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic regression (not yet converged) | 0.6884 | 0.6827 | 0.3030 | 0.6742 | 0.4181 | 0.7118 |
| Random forest | 0.8451 | 0.6732 | 0.5441 | 0.4157 | 0.4713 | 0.7540 |

The initial logistic regression run emitted a convergence warning. The script has therefore been updated to scale the sparse feature matrix before fitting logistic regression, and this model must be rerun before its final metrics are reported. The random-forest result is from the original exploratory run and remains a preliminary result.

The majority baseline's 83.40% accuracy comes from predicting every test case as non-cancelled; it detects no cancellations. For the class of interest, precision/recall/F1 and average precision are more informative than accuracy alone.

Run the updated experiment after pulling:

```powershell
git pull origin main
python -m py_compile src/train_cancellation_model.py
python src/train_cancellation_model.py
```

Limitations: this is a single random/stratified split on a small, static telecommunications service dataset. It is not a future-time evaluation, and the availability of the final task fields at the intended prediction time remains unproven. Treat the numbers as exploratory, not as evidence of production or plumbing-company cancellation prediction.

## Updated random-holdout cancellation model results

After scaling the sparse feature matrix, Logistic Regression completed without a convergence warning in the latest run. The results below are from a stratified random 80/20 hold-out (2,141 train rows, 536 test rows):

| Model | Accuracy | Balanced accuracy | Cancellation precision | Cancellation recall | Cancellation F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.8340 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic regression (scaled) | 0.6959 | 0.6917 | 0.3112 | 0.6854 | 0.4281 | 0.7138 |
| Random forest | 0.8451 | 0.6732 | 0.5441 | 0.4157 | 0.4713 | 0.7540 |

The random forest has the stronger F1 and ROC-AUC in this split; logistic regression catches more cancellations but produces more false positives. The majority baseline predicts every task as not cancelled, so its accuracy is not useful as a cancellation-detection score.

These remain exploratory numbers. The random hold-out can mix older and newer task records, and the dataset is a static telecommunications instance rather than a plumbing-company dataset. The `modified` field and `task_id` remain excluded due to target-leakage and identity concerns.

### Next evaluation: chronological hold-out

To test whether results hold when predicting tasks created later than the training examples, run:

```powershell
git pull origin main
python -m py_compile src/train_cancellation_model_temporal.py
python src/train_cancellation_model_temporal.py
```

The temporal experiment sorts tasks by `created`, trains on the older 80%, and tests on the newest 20%. It keeps the `modified` timestamp and `task_id` out of the features. This is a stricter exploratory test, but not proof of production performance: target timing and the availability of features at the actual decision point still need to be justified.

## Chronological cancellation result and model decision

The chronological hold-out sorted task records by `created`, trained on the older 80%, and tested on the newest 20% (2,141 training rows; 536 test rows). The test set contained 82 cancelled tasks.

| Model | Accuracy | Balanced accuracy | Cancellation precision | Cancellation recall | Cancellation F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.8470 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic regression | 0.4832 | 0.5201 | 0.1626 | 0.5732 | 0.2534 | 0.5227 |
| Random forest | 0.8470 | 0.5050 | 0.5000 | 0.0122 | 0.0238 | 0.6036 |

At the default threshold, Random Forest identified only 1 of 82 cancelled test tasks. Accuracy matches the majority baseline, which identifies no cancellations. This makes cancellation prediction too weak to present as the primary ML result at this stage. Keep the experiment as a negative finding, and investigate an alternative target rather than claiming cancellation-risk prediction is reliable.

## Chronological task-duration prediction

The recorded task `duration` is a candidate regression target. Run:

```powershell
git pull origin main
python -m py_compile src/train_duration_model_temporal.py
python src/train_duration_model_temporal.py
```

The script sorts tasks by their creation timestamp, trains on the older 80%, and tests on the newest 20%. It compares a training-set median baseline, Ridge regression and Random Forest regression. It reports MAE, RMSE, median absolute error and R-squared, and creates a comparison plot.

The script excludes the task ID, cancellation outcome and `modified` timestamp from its features. However, before interpreting results for a real business, confirm whether the source `duration` represents an estimate known before service or realized duration, and whether window/eligibility information is available at prediction time. The dataset represents telecommunications field service, not plumbing; even strong metrics would remain a proxy-data result.

## Evaluation plan

The final evaluation depends on the validated data and selected research question. Possible scheduling measures include estimated travel time, unserved tasks, task coverage, time-window feasibility and workload balance. If a prediction task is supported by the source data, we will evaluate it using an appropriate held-out metric. Results will be compared with a clearly defined baseline and explained with plots and written interpretation.

## Academic deliverables

The course submission requires a project report, presentation, code, openly available data-source links and a help file. The report should include each member's contribution and relevant page numbers on its first page, explain every result visualization and end with evidence-based recommendations. Code and code screenshots should be kept out of the report.

## Research source

Gamst, M. and Pisinger, D. *Decision Support for the Technician Routing and Scheduling Problem*. Networks, 83(1), 169–196. https://doi.org/10.1002/net.22188  
Dataset and results: https://zenodo.org/records/7904015

## Team

Group 5 — add all five members' names and enrollment numbers here once confirmed.
