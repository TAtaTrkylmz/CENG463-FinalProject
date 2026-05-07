from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.baselines import run_rag_compare
from llm_uncertainty.data import load_records, normalize_halueval_qa, write_splits
from llm_uncertainty.io import ensure_parent, write_jsonl
from llm_uncertainty.local_lm import LocalCausalLMScorer, score_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG compare method only.")
    parser.add_argument("--data-dir", default="data/processed/halueval_qa")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def prepare_dataset(output_dir: Path, limit: int | None, seed: int, overwrite: bool) -> None:
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"
    if not overwrite and train_path.exists() and val_path.exists() and test_path.exists():
        print(f"Using existing splits under {output_dir}")
        return

    records = normalize_halueval_qa(limit=limit)
    write_splits(records, output_dir, seed=seed)
    print(f"Prepared {len(records)} rows from HaluEval QA.")


def score_records(
    input_path: Path,
    output_path: Path,
    mode: str,
    model_name: str,
    overwrite: bool,
    limit: int | None,
) -> None:
    if not overwrite and output_path.exists():
        print(f"Using existing scored file {output_path}")
        return
    records = load_records(input_path, limit=limit)
    scorer = LocalCausalLMScorer(model_name=model_name)
    scored = [score_record(record, scorer, mode) for record in tqdm(records, desc=f"Scoring {mode}")]
    write_jsonl(scored, output_path)
    print(f"Saved {len(scored)} scored records to {output_path}")


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    scored_dir = results_dir / "scored"
    memory_scored_dir = scored_dir / "memory"
    context_scored_dir = scored_dir / "context"

    if not args.skip_prepare:
        prepare_dataset(data_dir, args.limit, args.seed, args.overwrite)

    eval_path = data_dir / f"{args.eval_split}.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError("Missing eval split. Run without --skip-prepare first.")

    memory_eval_path = memory_scored_dir / f"{args.eval_split}.jsonl"
    context_eval_path = context_scored_dir / f"{args.eval_split}.jsonl"

    if not args.skip_inference:
        score_records(eval_path, memory_eval_path, "memory", args.model_name, args.overwrite, args.limit)
        score_records(eval_path, context_eval_path, "context", args.model_name, args.overwrite, args.limit)

    rag_dir = results_dir / "rag" / "compare" / args.eval_split
    rag_pred_path = rag_dir / "predictions.csv"
    rag_metrics_path = rag_dir / "metrics.json"
    predictions, metrics = run_rag_compare(memory_eval_path, context_eval_path)

    ensure_parent(rag_pred_path)
    ensure_parent(rag_metrics_path)
    predictions.to_csv(rag_pred_path, index=False)
    rag_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved predictions to {rag_pred_path}")
    print(f"Saved metrics to {rag_metrics_path}")


if __name__ == "__main__":
    main()
