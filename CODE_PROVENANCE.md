# Code and data provenance

This file records known external inputs and the attribution checks required for this project. It is not a claim that every file has already passed a plagiarism audit.

## Project-specific implementations

The five modules and the integrated Streamlit portal are maintained as one academic group project. Standard library calls, common algorithms (for example TF-IDF, linear SVMs, Random Forests, gradient boosting, and routing heuristics), and common mathematical definitions are not by themselves evidence of copied code. Each module should still document any copied or adapted implementation, snippet, tutorial, notebook, or third-party component used to build it.

For every code file, the team should be able to explain its purpose, inputs, outputs, design choices, and tests. If code was adapted from another source, retain the source attribution and comply with that source's license; do not remove attribution to make the implementation appear original.

## Known external data and research

### Technician routing and scheduling — enrollment 26535024

- Research paper: Gamst, M. & Pisinger, D. (2024), “Decision support for the technician routing and scheduling problem,” *Networks*, 83(1), 169–196. DOI: 10.1002/net.22188.
- Dataset: TDC Net technician-routing instance published with the research, Zenodo record 7904015: https://zenodo.org/records/7904015.
- The dataset describes telecommunications field service, not a plumbing company. Results from it must be described as a related field-service prototype, not measured performance at Mr. Rooter or another plumbing chain.
- The repository's report notes discrepancies between the paper's instance summary and the downloaded archive. Preserve that note until the counts are independently reconciled.

## Known external data and model artifacts

### Plumbing demand forecasting — enrollment 26535001

- The module README identifies the demand dataset as synthetic and cites its public Kaggle dataset page: https://www.kaggle.com/datasets/adityaraiiiii/plumbing-demand-dataset.
- Model files and reported metrics should be traced to the training code/notebook and a reproducible evaluation. Do not describe synthetic-data results as real-world company performance.
- Confirm whether any source notebook or starter repository was reused; add its URL and license here if applicable.

### Water-leak detection — enrollment 25920036

- The notebook uses accelerometer recordings for leak/no-leak classification. Record the original recording/dataset source and license here after checking the notebook's download/source cells.
- The current documentation says 70 windows came from only two original recordings (one per class). Keep this limitation prominent; window-level scores do not establish generalization to new recordings.

### Customer feedback analysis — enrollment 26535016

- The implementation uses standard NLP/ML techniques including TF-IDF and a calibrated linear SVM. These are established methods and should be cited in the report where relevant, but their use alone does not imply code copying.
- Verify and record the origin, license, and generation method of `plumbing_reviews.csv`. Clearly distinguish synthetic/demo reviews from real customer feedback.
- The complaint playbook and response templates should be reviewed by the project team and edited to reflect its own decisions. Do not represent generated or illustrative text as actual company policy.

### Strategic resource optimization — enrollment 26535023

- Before submission, identify the source of each dataset, benchmark, notebook, formula, and optimization method in this module. Add direct URLs, publication details, and license information after checking the actual source files.
- Clearly separate measured outputs from assumptions and scenario estimates.

## Audit checklist

- [ ] Check notebooks' Markdown, code cells, and output text for copied passages and source links.
- [ ] Compare each module with its cited repositories, papers, datasets, and starter code.
- [ ] Add license notices for reused third-party code and preserve required notices.
- [ ] Identify synthetic data, downloaded data, model artifacts, and generated reports separately.
- [ ] Run tests and record results after each behavioral refactor.
- [ ] Do not use variable renaming, comment changes, or formatting alone as evidence of originality.
- [ ] Do not claim a numerical plagiarism score unless an actual similarity report was produced and interpreted.
