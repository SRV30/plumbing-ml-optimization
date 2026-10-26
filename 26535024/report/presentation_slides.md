# Presentation Slides — ML-Based Business Optimization for Plumbing Service Chains

**Group 5 | Module 26535024**  
**Status:** Slide content draft; add verified team/course details and transfer to the required presentation template before submission.

---

## Slide 1 — Title

**ML-Based Business Optimization for Plumbing Service Chains**  
Technician Routing, Scheduling, and Duration Prediction

- Group 5
- Institution/course: [Complete]
- Team members: [Complete]
- Faculty guide: [Complete]
- Submission date: [Complete]

*Scope note: the experiments use a telecommunications field-service benchmark, not plumbing-company data.*

## Slide 2 — Business Motivation and Problem

- Field-service operations must assign jobs to qualified technicians.
- Decisions are constrained by technician shifts, task time windows, service durations, and travel times.
- Poor route order can increase estimated travel.
- Research question: how do simple scheduling heuristics compare, and do the available data support useful ML predictions?

**Boundary:** plumbing is the motivating application; the benchmark itself is telecommunications field service.

## Slide 3 — Dataset and Validation

- Source: TDC Net technician-routing and scheduling benchmark associated with Gamst and Pisinger.
- 2,677 task records across a five-day horizon.
- 444 cancelled tasks; 2,233 active tasks in the scheduling experiments.
- The downloaded archive parses into 1,430 technician-day records and 32 observed skill labels.
- The paper reports 720 workday technicians and 31 skills; the discrepancy remains unresolved.

**Implication:** report observed data as parsed and do not claim exact reproduction of the published counts.

## Slide 4 — Scheduling Methods

1. **Greedy append-only:** add feasible tasks in sequence without revisiting earlier route order.
2. **Route insertion:** consider inserting tasks into route positions rather than only appending.
3. **Relocation local search:** refine assignments with feasible task relocations.
4. **Enhanced route-order search:** try intra-route swaps and 2-opt reversals; accept only feasible moves that reduce travel.

All are heuristics; none guarantees a globally optimal schedule.

## Slide 5 — Scheduling Results

| Method | Assigned tasks | Coverage | Routes | Estimated travel |
|---|---:|---:|---:|---:|
| Greedy append-only | 2,209 | 98.93% | 748 | 17,783 min |
| Route insertion | 2,209 | 98.93% | 643 | 14,627 min |
| Relocation local search | 2,209 | 98.93% | 637 | 14,239 min |
| Enhanced route-order local search | 2,209 | 98.93% | 637 | **14,174 min** |

Travel includes the final return-to-depot legs.

## Slide 6 — What the Scheduling Results Mean

- Enhanced local search reduced estimated travel by **3,609 minutes (20.30%)** compared with greedy scheduling.
- It reduced travel by **453 minutes (3.10%)** compared with route insertion.
- The final route-order pass saved a further **65 minutes (0.46%)** compared with its 14,239-minute starting schedule.
- The same 2,209 tasks remained assigned; coverage did not improve or decline between methods.

These are estimated travel minutes on one benchmark instance—not measured financial savings, fuel reductions, or plumbing-business outcomes.

## Slide 7 — Validation and Reliability

Enhanced schedule validator reported:

- Source task rows: 2,677
- Schedule status rows: 2,677
- Route assignment rows: 2,209
- Route groups: 637
- Validation checks: 28,261
- Validation errors: 0

The checks cover task accounting, eligibility, route timing, travel legs, and summary consistency under the implemented rules. They do not prove global optimality or guarantee real-world feasibility.

## Slide 8 — Machine-Learning Experiments

**Duration prediction**
- Chronological holdout Ridge MAE: 25.83 minutes.
- Median-baseline MAE: 44.55 minutes.
- The recorded duration field's operational meaning and real prediction-time availability need confirmation.

**Cancellation prediction**
- Chronological performance was weak.
- The evaluated Random Forest detected 1 of 82 cancelled test tasks at the reported threshold.

**Interpretation:** ML predictions have not demonstrated an overall scheduling improvement.

## Slide 9 — Counterfactual Duration Scheduling

| Scenario | Assigned | Coverage | Routes | Travel | Routes with replay violations |
|---|---:|---:|---:|---:|---:|
| Source-duration insertion baseline | 2,209 | 98.93% | 643 | 14,627 min | 0 |
| Q50 | 2,214 | 99.15% | 626 | 14,340 min | 57 |
| Q80 | 2,204 | 98.70% | 648 | 14,765 min | 26 |
| Buffered Ridge | 2,198 | 98.43% | 668 | 15,855 min | 11 |

Replay violations are stress-test findings from evaluating predicted route orders with source-recorded durations; they are not observed real-world failures. No single duration strategy dominates all measures.

## Slide 10 — Conclusion and Future Work

**Supported conclusion**
- The tested scheduling heuristics improved estimated travel relative to the greedy baseline on this telecommunications benchmark.
- Enhanced local search achieved 14,174 estimated travel minutes with 98.93% active-task coverage and passed the implemented validator.
- Duration and cancellation ML remain exploratory; no overall ML-driven scheduling benefit has been established.

**Next steps**
- Resolve or further document the data-count discrepancy.
- Obtain suitable plumbing-company data before making plumbing-specific claims.
- Add reproducibility tests and confirm duration semantics and feature availability.
- Complete team/course metadata, final report pagination, and required presentation export.

## Slide 11 — References

1. Gamst, M., & Pisinger, D. (2024). *Decision support for the technician routing and scheduling problem*. Networks, 83(1), 169–196. https://doi.org/10.1002/net.22188
2. Gamst, M., & Pisinger, D. (2023). TDC Net technician routing and scheduling dataset. Zenodo. https://zenodo.org/records/7904015
3. Project repository: https://github.com/SRV30/plumbing-ml-optimization

---

## Presenter notes

- Explain that 14,174 minutes is the best measured travel result among the tested methods, not an optimal solution.
- Do not describe the benchmark as plumbing data.
- Distinguish baseline scheduler comparison figures from the separately validated enhanced route-order result.
- Explain that replay violations expose risk in the counterfactual ML scheduling scenarios; they do not represent observed service failures.
- Add team contributions and any course-required acknowledgements before presenting.
