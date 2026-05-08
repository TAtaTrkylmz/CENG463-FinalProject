# Project Progress

Date: 9 May 2026

## Current Progress Summary
- Problem framing and pipeline setup are complete for the first pass of the study (epistemic uncertainty and hallucination detection).
- The default dataset (HaluEval QA) is integrated with a deterministic preprocessing step that creates train/val/test splits.
- Local causal LM scoring is implemented (token log-probability features) with memory and context prompting modes.
- Four methods are implemented and runnable: three baselines plus one LM scoring stage used as features.
- Baselines: lexical SVM (TF-IDF + linear SVM), entropy classifier (logprob features + logistic regression), and RAG compare (context-vs-memory delta thresholding).
- Scoring stage: local causal LM log-probability extraction for uncertainty features (memory and context prompts).
- Evaluation and report-asset generation (metrics table + plots) are implemented and connected to the results folder.

## Repository Structure and Behavior

### data/
- raw/: Reserved for raw dataset downloads.
- external/: Placeholder for any external resources that are not part of the dataset but may be used later.
- processed/halueval_qa/: Output of dataset normalization and splitting. Each original QA sample becomes two supervised rows (factual and hallucinated).

### scripts/
- prepare_dataset.py: Downloads HaluEval QA, normalizes it into supervised rows, and writes train/val/test JSONL splits.
- run_inference.py: Scores processed samples with a local causal LM and writes scored JSONL (memory or context mode).
- run_baseline.py: Runs one of the baselines and saves predictions + metrics.
- run_rag.py: Runs the RAG comparison baseline end-to-end (prepare, score, compare) with optional skip flags.
- evaluate.py: Computes metrics from a predictions CSV.
- make_report_assets.py: Creates a summary CSV of all metrics and generates plots in reports/.

### src/
- main.py: Full pipeline runner (prepare -> inference -> baselines -> report assets) with skip flags.
- llm_uncertainty/data.py: Dataset normalization, splitting, and record loading.
- llm_uncertainty/local_lm.py: Local causal LM scoring with token log-probability extraction.
- llm_uncertainty/features.py: Feature computation for log-probabilities and RAG comparison deltas.
- llm_uncertainty/baselines.py: Baseline implementations (lexical SVM, entropy classifier, RAG compare).
- llm_uncertainty/metrics.py: Common classification metric computation.
- llm_uncertainty/reporting.py: Aggregates metrics to a report table and generates plots.
- llm_uncertainty/prompts.py: Memory and context prompt templates.
- llm_uncertainty/io.py: JSONL utilities and safe path creation.

### results/
- baseline outputs are stored under per-method folders (e.g., entropy/, lexical_svm/, rag/compare/).
- scored/: Scored JSONL files for memory and context modes.
- metrics/: Aggregated metrics JSON files.
- predictions/: Consolidated prediction CSVs for reporting.

### reports/
- tables/: Report-ready CSV tables (e.g., initial_results.csv).
- figures/: Generated plots (confusion matrices, ROC curves, calibration plots).

### docs/
- progress_report_template.md: Template outlining required progress report sections and evaluation criteria.
- papers/: Literature review storage location.

## Current Outputs in the Workspace
- Processed dataset splits exist under data/processed/halueval_qa/.
- Baseline results exist under results/ (entropy, lexical_svm, rag compare).
- Initial reporting assets exist under reports/tables/ and reports/figures/.

## Immediate Next Steps
- Add a short literature review summary under docs/papers/ and link key findings.
- Run additional experiments (larger limits or full dataset) to stabilize metrics.
- Expand evaluation metrics (ECE, Brier score, coverage-risk) if needed.
- Add error analysis examples to prepare for the final report.
