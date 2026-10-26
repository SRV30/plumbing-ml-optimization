# Quantile Duration ML for Risk-Aware Technician Scheduling

## Purpose

This experiment extends the `26535024` technician-scheduling module with a job-level machine-learning approach designed for schedule planning. Rather than returning only one expected duration, quantile regression estimates both a typical duration (the 50th percentile) and a conservative planning duration (the 80th percentile).

This is a distinct contribution from branch-level demand forecasting: it predicts the duration of an individual task and explores how duration uncertainty can be represented in technician scheduling.

## Run

From the repository root, after parsing the source data:

```bash
python src/parse_tdc_records.py
python src/train_duration_quantile_model.py
```

The input file is `data/processed/tasks.csv`. The script creates:

- `results/duration_quantile_metrics.json`
- `results/duration_quantile_predictions.csv`

The script does not modify the existing duration-model outputs or scheduling baselines.

## Method

1. Sort task records by creation timestamp.
2. Use the oldest 80% for training and the newest 20% for testing.
3. Build candidate features from priority, penalty, task time-window length, creation/window timing, required skills, and number of eligible technicians.
4. Exclude task ID, duration target, cancellation outcome, and modification timestamp from the duration features.
5. Compare a median baseline and Ridge regression with gradient-boosting quantile regressors at the 50th and 80th percentiles.
6. Report MAE/RMSE for point predictions, pinball loss for quantile predictions, and empirical coverage for the 80th-percentile estimate.

## How to interpret the output

- **Q50** is a typical-duration estimate.
- **Q80** is a conservative planning estimate: ideally, around 80% of held-out recorded durations should be at or below it.
- If Q80 has low empirical coverage, the model is underestimating the upper tail more often than intended.
- A wider Q80-to-Q50 gap indicates greater estimated uncertainty for the planning horizon.
- The Q80 estimate is not a guarantee and should be tested through the existing schedule-feasibility replay before being used by a scheduler.

Do not claim the model improves routing unless a controlled experiment shows an improvement in schedule feasibility or another predefined operational metric.

## Limitations

- The benchmark is telecommunications field-service data, not data from a plumbing company.
- The model predicts the dataset's recorded duration field. Confirm its operational meaning before describing it as actual service time or planned duration.
- The chronological hold-out is a useful first check but is only one test on one benchmark instance.
- Use only features genuinely available when a job is scheduled.
- Quantile predictions can cross or be miscalibrated; this implementation enforces Q80 >= Q50 and reports empirical coverage, but it does not establish calibration across future datasets.
- No performance results are claimed until the script is run against the locally prepared data.

## Next experiment

Use Q50 and Q80 estimates as separate duration-planning scenarios in the existing scheduler, then replay the resulting routes with source-recorded durations. Compare coverage, total travel, route feasibility violations, and unassigned jobs against the validated source-duration baseline. Keep all scenarios on the same task set and do not describe the source-recorded duration as a true future outcome unless the dataset semantics support that interpretation.
