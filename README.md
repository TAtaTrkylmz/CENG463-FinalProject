# CENG463 Final Project

**Title:** Epistemic Uncertainty in LLM Hallucinations

This project evaluates lexical, uncertainty, semantic, and evidence-grounded signals for binary hallucination detection on HaluEval QA.

- `label=0`: factual answer
- `label=1`: hallucinated answer

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

The repository virtual environment can also be used directly:

```bash
venv/bin/python src/main.py --dry-run
```

Transformer checkpoints are downloaded from Hugging Face on first use. GPU execution is recommended for semantic, NLI, upgraded-entropy, and cross-encoder experiments.

## Dataset

Each HaluEval question provides one factual and one hallucinated candidate answer. Dataset splitting is grouped by `original_sample_id`, so the paired answers for one question always remain in the same split.

Current grouped split:

| Split | QA groups | Rows | Factual | Hallucinated |
|---|---:|---:|---:|---:|
| Train | 7,999 | 15,998 | 7,999 | 7,999 |
| Validation | 1,001 | 2,002 | 1,001 | 1,001 |
| Test | 1,000 | 2,000 | 1,000 | 1,000 |

## Ten Baselines

| # | CLI name | Signal | Method |
|---:|---|---|---|
| 1 | `lexical_svm` | Lexical artifacts | Answer TF-IDF with linear SVM |
| 2 | `lexical_hybrid_svm` | Lexical + uncertainty | TF-IDF and standardized uncertainty features |
| 3 | `entropy_base` | Base LM confidence | Standardized logistic regression over `distilgpt2` features |
| 4 | `entropy_upgraded` | Instruction-tuned LM confidence | Same classifier using a stronger causal LM |
| 5 | `semantic_svm` | Meaning | Question-answer bi-encoder interactions with LinearSVC |
| 6 | `semantic_hybrid_svm` | Meaning + uncertainty | Semantic interactions and standardized uncertainty |
| 7 | `rag_compare_fixed` | Context sensitivity | Memory/context NLL delta with threshold fitted on training data |
| 8 | `nli_evidence` | Knowledge support | Zero-shot entailment score from an MNLI model |
| 9 | `evidence_aware_hybrid` | Combined evidence | Semantic, NLI, and uncertainty features |
| 10 | `cross_encoder` | Joint text reasoning | Fine-tuned question-answer transformer classifier |

Default transformer checkpoints:

- Upgraded uncertainty LM: `Qwen/Qwen2.5-1.5B-Instruct`
- Semantic encoder: `sentence-transformers/all-MiniLM-L6-v2`
- NLI model: `FacebookAI/roberta-large-mnli`
- Cross-encoder initialization: `distilroberta-base`

## Run Everything

`src/main.py` is the canonical experiment entry point. It:

1. Prepares leakage-free grouped splits.
2. Scores train and evaluation rows in memory and context modes.
3. Scores rows with the upgraded instruction-tuned LM.
4. Runs all ten baselines.
5. Saves predictions and metrics.
6. Generates individual and comparison plots.

Preview the complete plan:

```bash
venv/bin/python src/main.py --dry-run
```

Run all experiments on a GPU:

```bash
venv/bin/python src/main.py \
  --device cuda \
  --batch-size 8 \
  --epochs 1
```

Use `--overwrite` to regenerate existing splits, scored features, and experiment outputs:

```bash
venv/bin/python src/main.py \
  --device cuda \
  --batch-size 8 \
  --epochs 1 \
  --overwrite
```

The upgraded LM can be changed:

```bash
venv/bin/python src/main.py \
  --device cuda \
  --upgraded-model-name mistralai/Mistral-7B-Instruct-v0.3 \
  --upgraded-model-dtype bfloat16
```

Large or gated checkpoints may require Hugging Face authentication and additional GPU memory.

## Run a Subset

Fast CPU-oriented baselines:

```bash
venv/bin/python src/main.py \
  --baselines lexical_svm lexical_hybrid_svm entropy_base rag_compare_fixed \
  --device cpu
```

Semantic and evidence baselines:

```bash
venv/bin/python src/main.py \
  --baselines semantic_svm semantic_hybrid_svm nli_evidence evidence_aware_hybrid \
  --device cuda \
  --batch-size 8
```

Cross-encoder only:

```bash
venv/bin/python src/main.py \
  --baselines cross_encoder \
  --device cuda \
  --batch-size 8 \
  --epochs 1
```

## Reuse Existing Features

By default, `main.py` reuses existing scored JSONL files and completed baseline outputs. Use these flags when appropriate:

- `--skip-prepare`: use existing dataset splits
- `--skip-inference`: use existing memory/context/upgraded scores
- `--skip-baselines`: generate reports from existing predictions and metrics
- `--skip-report-assets`: run experiments without creating figures
- `--overwrite`: rerun and replace existing outputs

Generate reports from the current matrix without retraining:

```bash
MPLBACKEND=Agg venv/bin/python src/main.py \
  --skip-prepare \
  --skip-inference \
  --skip-baselines
```

## Outputs

Each baseline writes:

```text
results/matrix/<baseline>/<split>/metrics.json
results/matrix/<baseline>/<split>/predictions.csv
```

Semantic and NLI features are cached under:

```text
results/matrix/feature_cache/
```

Comparison artifacts:

```text
reports/tables/baseline_results_<split>.csv
reports/tables/baseline_comparison_<split>.csv
reports/figures/matrix/baseline_comparison_metrics_<split>.png
reports/figures/matrix/baseline_comparison_roc_<split>.png
reports/figures/matrix/baseline_comparison_precision_recall_<split>.png
```

The same figure directory also contains confusion matrices, ROC curves,
precision-recall curves, and calibration plots for every completed baseline.
Reliability diagrams are only meaningful for outputs already bounded to
`[0, 1]`; unbounded SVM/RAG score panels are explicitly marked unavailable.

Comparative error-analysis artifacts:

```text
reports/error_analysis/<split>/summary_by_baseline.csv
reports/error_analysis/<split>/false_positives_topk.csv
reports/error_analysis/<split>/false_negatives_topk.csv
reports/error_analysis/<split>/overconfident_errors_topk.csv
reports/error_analysis/<split>/hardest_shared_errors.csv
reports/figures/error_analysis/
```

Generate them directly from saved predictions without retraining:

```bash
MPLBACKEND=Agg venv/bin/python scripts/run_error_analysis.py --split val
```

## Tests

Run the offline unit tests:

```bash
venv/bin/python -m unittest discover -s tests -v
```

Compile-check the pipeline:

```bash
venv/bin/python -m py_compile src/main.py src/llm_uncertainty/*.py scripts/*.py tests/*.py
```

## Current Validation Results

All ten baseline runs are available on the grouped validation split.

| Baseline | Accuracy | Macro F1 | AUROC |
|---|---:|---:|---:|
| Lexical SVM | 0.9381 | 0.9380 | 0.9729 |
| Lexical Hybrid SVM | 0.9530 | 0.9530 | 0.9805 |
| Entropy Base | 0.9426 | 0.9425 | 0.9765 |
| Entropy Upgraded | 0.9231 | 0.9231 | 0.9654 |
| Semantic SVM | 0.9695 | 0.9695 | 0.9870 |
| Semantic Hybrid SVM | 0.9675 | 0.9675 | 0.9878 |
| RAG Compare Fixed | 0.7522 | 0.7490 | 0.7723 |
| NLI Evidence | 0.7877 | 0.7866 | 0.7943 |
| Evidence-Aware Hybrid | 0.9770 | 0.9770 | **0.9951** |
| Cross-Encoder | **0.9815** | **0.9815** | 0.9895 |

These are validation results, not final test-set estimates.
