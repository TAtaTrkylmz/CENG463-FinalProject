# CENG463 Final Project

**Title:** Epistemic Uncertainty in LLM Hallucinations

This repository studies binary hallucination detection on the HaluEval QA benchmark using lexical, uncertainty, semantic, and evidence-grounded signals.

- `label=0`: factual answer
- `label=1`: hallucinated answer

## Repository Status

The current codebase is organized around a unified experiment runner in `src/main.py` plus helper scripts in `scripts/`.

What is already committed:

- Processed grouped HaluEval splits in `data/processed/halueval_qa/`
- Checked-in report figures under `reports/figures/`
- A detailed technical summary in `progress.md`
- Final report sources in `final_report/`

What is **not** currently committed:

- The default modern runtime output directory `results/matrix/`
- The upgraded-LM scored cache expected at `results/scored_upgraded/`

There are also older result folders under `results/` such as `entropy/`, `lexical_svm/`, `rag/`, `metrics/`, and `predictions/`. These appear to be legacy outputs from earlier pipeline stages rather than the current source of truth for the 10-baseline matrix.

## Project Goal

Each HaluEval QA item provides one factual and one hallucinated candidate answer. The project predicts whether a candidate answer is hallucinated from combinations of:

- lexical artifacts
- causal-LM uncertainty features
- semantic question-answer interactions
- external evidence support

Dataset splits are grouped by `original_sample_id`, so both candidate answers for the same original question always stay in the same split.

## Dataset

Committed grouped split sizes:

| Split | QA groups | Rows | Factual | Hallucinated |
|---|---:|---:|---:|---:|
| Train | 7,999 | 15,998 | 7,999 | 7,999 |
| Validation | 1,001 | 2,002 | 1,001 | 1,001 |
| Test | 1,000 | 2,000 | 1,000 | 1,000 |

The grouped split replaces an earlier row-level split that could leak paired-question information across train and evaluation sets.

## Baselines

The current experiment matrix contains 10 baselines:

| # | CLI name | Signal | Method |
|---:|---|---|---|
| 1 | `lexical_svm` | Lexical artifacts | TF-IDF over candidate answers with linear SVM |
| 2 | `lexical_hybrid_svm` | Lexical + uncertainty | TF-IDF plus standardized LM uncertainty features |
| 3 | `entropy_base` | Base LM confidence | Logistic regression over `distilgpt2` uncertainty features |
| 4 | `entropy_upgraded` | Instruction-tuned LM confidence | Same classifier with a stronger causal LM |
| 5 | `semantic_svm` | Meaning | Bi-encoder question-answer interaction features with LinearSVC |
| 6 | `semantic_hybrid_svm` | Meaning + uncertainty | Semantic interactions plus uncertainty features |
| 7 | `rag_compare_fixed` | Context sensitivity | Train-derived threshold over memory/context NLL deltas |
| 8 | `nli_evidence` | Knowledge support | Zero-shot entailment score from an MNLI model |
| 9 | `evidence_aware_hybrid` | Combined evidence | Semantic, NLI, and uncertainty features |
| 10 | `cross_encoder` | Joint reasoning | Fine-tuned paired transformer classifier |

Default transformer checkpoints used by the current code:

- Base uncertainty LM: `distilgpt2`
- Upgraded uncertainty LM: `Qwen/Qwen2.5-1.5B-Instruct`
- Semantic encoder: `sentence-transformers/all-MiniLM-L6-v2`
- NLI model: `FacebookAI/roberta-large-mnli`
- Cross-encoder initialization: `distilroberta-base`

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional virtual environment examples:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Transformer checkpoints are downloaded from Hugging Face on first use. GPU execution is strongly recommended for:

- `semantic_svm`
- `semantic_hybrid_svm`
- `nli_evidence`
- `evidence_aware_hybrid`
- `entropy_upgraded`
- `cross_encoder`

Large or gated checkpoints may require Hugging Face authentication and substantial GPU memory.

## Recommended Workflow

`src/main.py` is the canonical experiment entry point. It can:

1. prepare grouped dataset splits
2. score train/eval rows in memory and context modes
3. score upgraded-LM uncertainty features
4. run any subset of the 10 baselines
5. save predictions and metrics
6. generate plots, tables, and comparative error analysis

Preview the run plan:

```bash
python src/main.py --dry-run
```

Run the full matrix:

```bash
python src/main.py --device cuda --batch-size 8 --epochs 1
```

Rerun everything from scratch:

```bash
python src/main.py --device cuda --batch-size 8 --epochs 1 --overwrite
```

Run only a subset:

```bash
python src/main.py ^
  --baselines lexical_svm lexical_hybrid_svm entropy_base rag_compare_fixed ^
  --device cpu
```

On bash shells, use `\` instead of `^` for line continuation.

Useful reuse flags:

- `--skip-prepare`
- `--skip-inference`
- `--skip-baselines`
- `--skip-report-assets`
- `--overwrite`

For out-of-domain evaluation, you can keep HaluEval as training data and point evaluation at a different directory with the same JSONL schema:

```bash
python src/main.py \
  --train-data-dir data/processed/halueval_qa \
  --eval-data-dir data/processed/open_domain_gpt_llama \
  --eval-split val \
  --skip-prepare
```

To prepare an external benchmark JSONL, use:

```bash
python scripts/prepare_open_domain_eval.py \
  --input data/raw/open_domain_generations.jsonl \
  --output-dir data/processed/open_domain_gpt_llama \
  --split val \
  --source-dataset open_domain_generations \
  --generator-model gpt-4o-mini
```

The input JSONL for `prepare_open_domain_eval.py` must contain:

- `question`
- `candidate_answer`
- `label`

Optional fields such as `knowledge`, `reference_answer`, `generator_model`, `notes`, and `topic` are preserved.

A ready-to-copy schema example is provided at `data/open_domain_generations.template.jsonl`.

To prepare a real second benchmark directly from TruthfulQA, use:

```bash
venv/bin/python scripts/prepare_truthfulqa_eval.py \
  --output-dir data/processed/truthfulqa_eval \
  --split val \
  --incorrect-answer-policy first
```

This creates a balanced external set with one factual and one hallucinated answer per TruthfulQA question, using the canonical answer as the factual row and one incorrect answer as the hallucinated row.

To evaluate on actual model-generated answers instead of benchmark-provided incorrect answers, generate TruthfulQA answers from a model and auto-build an evaluation split:

```bash
venv/bin/python scripts/generate_truthfulqa_model_outputs.py \
  --output-dir data/processed/truthfulqa_qwen_generations \
  --provider hf \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --limit 100
```

For OpenAI models, set `OPENAI_API_KEY` and use `--provider openai --model-name gpt-4o-mini`. The script stores both raw generations and a derived evaluation JSONL compatible with `src/main.py`.

Once you have both in-domain and external results, compare them with:

```bash
python scripts/compare_domains.py \
  --source-results results/matrix_halueval \
  --target-results results/matrix_open_domain \
  --output reports/tables/halueval_vs_open_domain.csv \
  --source-label halueval \
  --target-label open_domain
```

This produces a baseline-by-baseline table with the source metric, target metric, and delta for Accuracy, Macro F1, and AUROC.

Example: rebuild report assets from already-generated matrix outputs:

```bash
python src/main.py --skip-prepare --skip-inference --skip-baselines
```

## Helper Scripts

The `scripts/` directory contains smaller entry points around the same pipeline:

- `scripts/prepare_dataset.py`: download and build grouped HaluEval splits
- `scripts/run_inference.py`: score a split with a local causal LM in `memory` or `context` mode
- `scripts/run_baseline.py`: run one baseline and save predictions/metrics
- `scripts/run_experimental_matrix.py`: iterate over the 10-baseline matrix through `run_baseline.py`
- `scripts/run_error_analysis.py`: regenerate comparative error-analysis tables and figures
- `scripts/evaluate.py`: compute metrics from a saved predictions CSV
- `scripts/prepare_truthfulqa_eval.py`: build a TruthfulQA-based external evaluation split
- `scripts/compare_domains.py`: compare in-domain and out-of-domain metrics across baselines

`run_baseline.py` still accepts several legacy aliases such as `entropy`, `rag_compare`, `hybrid_proposed`, and `hybrid_svm` for backward compatibility.

An optional follow-up baseline, `evidence_aware_length_hybrid`, trains separate short-answer and long-answer experts on top of the evidence-aware feature stack. It is not part of the default 10-baseline matrix, but it is available for directly addressing the short-answer failure mode:

```bash
python scripts/run_baseline.py \
  --baseline evidence_aware_length_hybrid \
  --train data/processed/halueval_qa/train.jsonl \
  --eval data/processed/halueval_qa/val.jsonl \
  --train-memory results/scored/memory/train.jsonl \
  --eval-memory results/scored/memory/val.jsonl \
  --predictions-output results/length_aware/predictions.csv \
  --metrics-output results/length_aware/metrics.json
```

## Expected Runtime Outputs

When the current unified pipeline is run, its default outputs are:

```text
results/matrix/<baseline>/<split>/metrics.json
results/matrix/<baseline>/<split>/predictions.csv
results/matrix/feature_cache/
reports/tables/
reports/figures/matrix/
reports/error_analysis/<split>/
reports/figures/error_analysis/
```

Those directories are expected runtime products; they are not fully committed in the repository at the moment.

## Committed Artifacts

The repository already includes:

- grouped processed data in `data/processed/halueval_qa/`
- report figures for the validation matrix in `reports/figures/matrix/`
- comparative error-analysis figures in `reports/figures/error_analysis/`
- validation error-analysis notes in `reports/error_analysis/val/key_findings.md`
- a technical write-up in `progress.md`
- a LaTeX report in `final_report/`

## Validation Snapshot

The repository includes checked-in figures and documentation for a complete 10-baseline validation matrix. The summarized validation results recorded in `progress.md` are:

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

These are validation-set results, not final held-out test estimates.

## Tests

Run the offline unit tests:

```bash
python -m unittest discover -s tests -v
```

Optional compile check:

```bash
python -m py_compile src/main.py src/llm_uncertainty/*.py scripts/*.py tests/*.py
```

## Repository Layout

```text
CENG463-FinalProject/
  data/                 committed grouped HaluEval splits
  docs/                 project brief, papers, images
  final_report/         LaTeX report sources and compiled PDFs
  reports/              committed figures and report-side artifacts
  results/              legacy outputs and scored caches
  scripts/              helper CLIs
  src/                  main experiment pipeline and library code
  tests/                offline unit tests
  progress.md           detailed technical progress summary
```

## Current Gaps

Based on the repository state today:

- the modern `results/matrix/` raw outputs are not checked in
- `results/scored_upgraded/` is not present
- test-set matrix outputs are not committed
- some checked-in `results/` folders reflect older naming and pipeline stages

So the code for the full matrix is present, and the report-side validation artifacts are present, but reproducing the full current matrix from scratch will require regenerating runtime outputs locally.
