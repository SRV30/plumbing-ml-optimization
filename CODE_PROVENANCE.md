# Code, data, and research provenance

This document records known external inputs and remaining checks for this repository. It does not certify every file or establish that all code has been independently verified.

## General guidance

For every adapted source file, notebook, code snippet, formula, dataset, or model artifact, record the upstream URL, author/owner, license, and the parts used. Preserve copyright/license notices and explain substantial adaptations. The team should be able to explain each implementation's inputs, outputs, design choices, and tests.

Standard techniques such as TF-IDF, linear SVMs, tree-based models, signal statistics, and routing heuristics are established methods. Their use alone does not imply that a specific implementation was copied.

## Technician routing and scheduling — 26535024

- Research: Gamst, M. & Pisinger, D. (2024), “Decision support for the technician routing and scheduling problem,” *Networks*, 83(1), 169–196. DOI: https://doi.org/10.1002/net.22188
- Dataset: TDC Net field-service instance, Zenodo record https://zenodo.org/records/7904015
- The benchmark describes telecommunications field service, not a plumbing business. Present results as a related field-service prototype.
- The project report notes a mismatch between counts in the paper and the downloaded data archive. Keep the discrepancy documented until independently reconciled.

## Demand forecasting — 26535001

- The module README identifies the demand dataset as synthetic and cites https://www.kaggle.com/datasets/adityaraiiiii/plumbing-demand-dataset.
- Keep generated data, model outputs, and measured business outcomes clearly distinguished.
- Check the original notebook/repository history and add the source and license of any reused starter code or notebook material.

## Water-leak detection — 25920036

- The notebook uses accelerometer recordings for leak/no-leak classification. Verify the original recording source and its terms from the notebook's source/download cells and record them here.
- Current documentation describes 70 windows derived from two original recordings (one per class). Window-level scores are not evidence of performance on independent recordings.

## Customer feedback analysis — 26535016

- The implementation uses established NLP/ML techniques, including TF-IDF and calibrated linear SVM classification. Cite external implementation examples if code was adapted from them.
- Verify the origin, license, and generation method of `plumbing_reviews.csv`. Clearly label synthetic/demo reviews.
- Treat the complaint playbook and suggested replies as project-authored policy only if the team actually developed and reviewed them.

## Strategic resource optimization — 26535023

- Identify and cite the source of each dataset, benchmark, notebook, formula, and optimization method after checking the notebook's source cells.
- Separate assumptions and scenario estimates from validated operational results.

## Final reproducibility checklist

- [ ] Review code, Markdown, and output cells in each notebook.
- [ ] Compare module implementations with upstream repositories, tutorials, papers, and licenses.
- [ ] Preserve required notices for adapted code and third-party assets.
- [ ] Record how datasets and trained artifacts were produced.
- [ ] Run tests and reproduce reported metrics after behavioral changes.
- [ ] Do not treat formatting or variable renaming as substantive refactoring.
