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

Create a report-ready metrics table:

```powershell
python scripts/make_report_assets.py
```

## Current Status

The repo now contains the first local pipeline implementation. If model or dataset download fails, retry in an environment with Hugging Face network access or pre-download the relevant cache files.
