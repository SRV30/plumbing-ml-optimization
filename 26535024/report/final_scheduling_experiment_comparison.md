# Final Scheduling and Duration-Model Evaluation

**Module:** 26535024  
**Evaluation date:** 26 October 2026  
**Dataset:** TDC Net telecommunications field-service benchmark

## Executive summary

The route-optimization experiments reduce travel from 17,783 minutes for the greedy baseline to 14,174 minutes for the enhanced local-search schedule, a reduction of 3,609 minutes (approximately 20.3%) with 2,209 assigned tasks retained. The enhanced schedule passed independent validation with 28,261 checks and zero errors.

The duration-model experiments reveal a separate trade-off. Raw Ridge has lower prediction error, while buffered Ridge reduces the underprediction rate from 38.99% to 10.63%. However, buffered Ridge increases MAE from 24.80 to 57.29 minutes and produces a schedule with more travel than the source-duration insertion baseline. The Q50 scenario lowers planned travel but has more replay violations. No single duration-estimation strategy dominates all reported measures.

All results are exploratory and use telecommunications field-service data, not a real plumbing dataset.

## 1. Route optimization

| Method | Assigned tasks | Routes used | Travel minutes |
|---|---:|---:|---:|
| Greedy append-only | 2,209 | 748 | 17,783 |
| Route insertion | 2,209 | 643 | 14,627 |
| Relocation local search | 2,209 | 637 | 14,239 |
| Enhanced local search | 2,209 | 637 | 14,174 |

Route insertion reduces travel by 3,156 minutes (17.75%) relative to greedy. Relocation local search reduces travel by a further 388 minutes (2.65%) relative to insertion. Enhanced local search reduces travel by another 65 minutes (0.46%) relative to the 14,239-minute starting schedule.

The enhanced schedule's assignment set was preserved. Independent validation reported 2,677 source tasks, 2,677 task-status rows, 2,209 assignment rows, 637 route groups, 28,261 checks, zero errors, and a passing result. These checks establish consistency under the implemented rules; they do not prove global optimality.

## 2. Duration prediction

### Chronological test-set metrics

| Metric | Raw Ridge | Buffered Ridge |
|---|---:|---:|
| MAE | 24.80 min | 57.29 min |
| RMSE | 48.32 min | 64.70 min |
| R-squared | 0.4603 | 0.0326 |
| Underprediction rate | 38.99% | 10.63% |
| One-sided coverage | 61.01% | 89.37% |

A nonnegative 46-minute buffer was added using a calibration residual quantile. This improved one-sided coverage on the test set to 89.37%, near the 90% target, but worsened point-prediction error. Test-set coverage is not a guarantee of future coverage under distribution shift.

The separate quantile experiment reported Q80 empirical coverage of 79.29%, close to its 80% target. The Ridge point model's MAE was 25.83 minutes in that experiment, while Gradient Boosting Q50 MAE was 27.54 minutes. Because the experimental splits and model variants differ, these metrics should not be merged as though they were one common test.

## 3. Counterfactual scheduling scenarios

| Scenario | Assigned tasks | Coverage | Routes | Travel minutes | Routes with replay violations |
|---|---:|---:|---:|---:|---:|
| Source-duration insertion baseline | 2,209 | 98.93% | 643 | 14,627 | 0 |
| Q50 | 2,214 | 99.15% | 626 | 14,340 | 57 |
| Q80 | 2,204 | 98.70% | 648 | 14,765 | 26 |
| Buffered Ridge | 2,198 | 98.43% | 668 | 15,855 | 11 |

The calibrated-duration scenario used predictions for 536 chronologically held-out tasks, retained source durations for earlier tasks, assigned 2,198 tasks, and achieved 98.43% coverage. Its route order used 15,855 travel minutes. Replay with source-recorded durations found 11 routes with violations and 11 unique tasks with task-window or shift violations.

Q50 has the lowest travel time and highest coverage among these counterfactual scenarios, but its 57 replay-violated routes are a substantial caution. Q80 lowers the number of replay-violated routes to 26 but has more travel and lower coverage than the baseline. Buffered Ridge has 11 replay-violated routes, but it also uses 1,228 more travel minutes than the source-duration baseline and assigns 11 fewer tasks.

These outcomes illustrate the risk-efficiency trade-off. They do not establish that replay-violated routes would fail in real operations: replay uses the source-recorded duration values, whose relationship to realized service time needs to be understood.

## 4. Recommended interpretation

1. Retain enhanced local search as the best measured route-travel result in the current runs, subject to its documented validation scope.
2. Do not claim that duration ML has improved scheduling overall. The experiments show trade-offs rather than a consistent win.
3. Choose duration estimates according to the operational objective. If underestimation is costly, buffered estimates may be worth investigating, but their impact on coverage and travel must also be measured.
4. Use a common held-out task set and consistent schedule constraints for the next comparison. Report assignment coverage, travel, route count, task-window/shift/depot-return violations, and prediction metrics separately.
5. Confirm the semantic meaning of the duration field and which features are available at scheduling time before describing the target as service duration or proposing deployment.

## 5. Limitations

- The dataset is a telecommunications field-service benchmark, not plumbing-company data.
- The scheduling algorithms are heuristics, not exact optimizers.
- The duration experiments are counterfactuals; the source-recorded durations are used for replay and are not independently verified realized outcomes.
- A single chronological holdout does not establish robustness across future periods or other businesses.
- No real plumbing-business cost savings, delay reductions, or productivity gains have been demonstrated.

## 6. Reproduction

Run from the repository root after preparing the local source data, processed CSVs and travel-time matrices:

```powershell
python 26535024/src/train_duration_quantile_model.py
python 26535024/src/improve_local_search_moves.py
python 26535024/src/validate_local_search_schedule.py --schedule enhanced
python 26535024/src/evaluate_quantile_scheduling_impact.py
python 26535024/src/train_calibrated_duration_model.py
python 26535024/src/evaluate_calibrated_duration_scheduling_impact.py
```

The generated datasets, matrices, schedules and detailed result CSVs are local artifacts and may be excluded from version control. The tables above document the reported local runs as of 26 October 2026.
