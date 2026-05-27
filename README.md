# CENG463 Final Project

**Title:** Epistemic Uncertainty in LLM Hallucinations

This repository supports a reproducible ML pipeline for studying whether uncertainty signals help detect hallucinations in LLM answers.

## Project Goals

We study whether uncertainty signals can help identify or explain hallucinations produced by large language models. The repository is structured so that we can:
- prepare and version datasets,
- run multiple baseline methods,
- compare hallucination and uncertainty signals,
- save reproducible experiment outputs,
- generate figures and tables for the progress report and final report.

## Default Dataset and Model

The first implementation uses **HaluEval QA**. Each original sample becomes two supervised examples:
- factual answer: `label=0`
- hallucinated answer: `label=1`

The default inference backend is a local Hugging Face causal LM (`distilgpt2`) so the project can extract token-level log probabilities without paying for an API.

## Setup

Create an environment and install dependencies:

```powershell
pip install -r requirements.txt
```

## Reproduce a Small End-to-End Run

Start with a small limit so the workflow is fast and cheap:

```powershell
python scripts/prepare_dataset.py --limit 100
```

Score candidate answers without retrieved context:

```powershell
python scripts/run_inference.py --input data/processed/halueval_qa/train.jsonl --output results/generations/halueval_qa_train_memory.jsonl --mode memory --limit 80
python scripts/run_inference.py --input data/processed/halueval_qa/val.jsonl --output results/generations/halueval_qa_val_memory.jsonl --mode memory --limit 20
```

Score the same validation examples with HaluEval knowledge as context:

```powershell
python scripts/run_inference.py --input data/processed/halueval_qa/val.jsonl --output results/generations/halueval_qa_val_context.jsonl --mode context --limit 20
```

Run the three baselines:

```powershell
python scripts/run_baseline.py --baseline lexical_svm --train data/processed/halueval_qa/train.jsonl --eval data/processed/halueval_qa/val.jsonl --predictions-output results/predictions/lexical_svm_val.csv --metrics-output results/metrics/lexical_svm_val.json
python scripts/run_baseline.py --baseline entropy --train results/generations/halueval_qa_train_memory.jsonl --eval results/generations/halueval_qa_val_memory.jsonl --predictions-output results/predictions/entropy_val.csv --metrics-output results/metrics/entropy_val.json
python scripts/run_baseline.py --baseline rag_compare --memory results/generations/halueval_qa_val_memory.jsonl --context results/generations/halueval_qa_val_context.jsonl --predictions-output results/predictions/rag_compare_val.csv --metrics-output results/metrics/rag_compare_val.json
```

Run the proposed hybrid model (lexical + uncertainty features):

```powershell
python scripts/run_baseline.py --baseline hybrid_proposed --train data/processed/halueval_qa/train.jsonl --eval data/processed/halueval_qa/val.jsonl --train-memory results/scored/memory/train.jsonl --eval-memory results/scored/memory/val.jsonl --eval-context results/scored/context/val.jsonl --predictions-output results/predictions/hybrid_proposed_val.csv --metrics-output results/metrics/hybrid_proposed_val.json
```

Note on convergence warning:
- If you see `ConvergenceWarning` for `hybrid_proposed`, the model still outputs predictions and metrics.
- This means optimizer iterations reached `max_iter` before full convergence.
- Additional run logs now print data sizes, matrix shapes, class balance, and optimizer iterations.
- The metrics JSON now also includes:
  - `optimizer_n_iter`
  - `optimizer_converged` (1 means converged, 0 means not fully converged)
  - `feature_count_total`
  - `train_rows_after_merge`
  - `eval_rows_after_merge`

Run ablations for component contribution analysis:

```powershell
python scripts/run_ablation.py --train-text data/processed/halueval_qa/train.jsonl --eval-text data/processed/halueval_qa/val.jsonl --train-memory results/scored/memory/train.jsonl --eval-memory results/scored/memory/val.jsonl --eval-context results/scored/context/val.jsonl --out-dir results/ablation/val
```

Ablation logging now reports, for each setting:
- train/eval row counts
- feature matrix shapes
- class distribution
- optimizer iterations and convergence status

The same fields are also written into each ablation metrics JSON and `summary.csv`.

Run error analysis on any prediction file:

```powershell
python scripts/run_error_analysis.py --predictions results/predictions/hybrid_proposed_val.csv --out-dir reports/error_analysis/hybrid_proposed_val --top-k 25
```

Error analysis logging now reports:
- loaded sample count
- total error count
- FP/FN/overconfident error counts

Create a report-ready metrics table:

```powershell
python scripts/make_report_assets.py
```

## What Was Added

- `hybrid_proposed` model in `scripts/run_baseline.py` and `src/llm_uncertainty/baselines.py`
- `scripts/run_ablation.py` for ablation settings:
  - `lexical_only`
  - `uncertainty_only`
  - `hybrid_no_context`
  - `hybrid_with_context`
- `scripts/run_error_analysis.py` for:
  - top false positives
  - top false negatives
  - top overconfident errors
  - error rate by answer length bucket

## How To See The Difference

1. Compare metrics JSON side by side:
   - `results/metrics/lexical_svm_val.json`
   - `results/metrics/entropy_val.json`
   - `results/metrics/rag_compare_val.json`
   - `results/metrics/hybrid_proposed_val.json`
2. Check ablation summary at:
   - `results/ablation/val/summary.csv`
   This shows which component (lexical, uncertainty, context features) contributes most.
3. Inspect failure patterns at:
   - `reports/error_analysis/hybrid_proposed_val/summary.csv`
   - `reports/error_analysis/hybrid_proposed_val/false_positives_topk.csv`
   - `reports/error_analysis/hybrid_proposed_val/false_negatives_topk.csv`
   - `reports/error_analysis/hybrid_proposed_val/overconfident_errors_topk.csv`
   - `reports/error_analysis/hybrid_proposed_val/error_by_answer_length.csv`

## Current Status

The repo now contains the first local pipeline implementation. If model or dataset download fails, retry in an environment with Hugging Face network access or pre-download the relevant cache files.
