from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from llm_uncertainty.baselines import run_entropy_classifier, run_lexical_svm, run_rag_compare
from llm_uncertainty.data import load_records, normalize_halueval_qa, write_splits
from llm_uncertainty.io import ensure_parent, write_jsonl
from llm_uncertainty.local_lm import LocalCausalLMScorer, score_record
from llm_uncertainty.reporting import build_plots, build_report_table


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the full hallucination detection pipeline.")
	parser.add_argument("--data-dir", default="data/processed/halueval_qa")
	parser.add_argument("--results-dir", default="results")
	parser.add_argument("--model-name", default="distilgpt2")
	parser.add_argument("--eval-split", choices=["val", "test"], default="val")
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--skip-prepare", action="store_true")
	parser.add_argument("--skip-inference", action="store_true")
	parser.add_argument("--skip-baselines", action="store_true")
	parser.add_argument("--skip-report-assets", action="store_true")
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
	label_counts = {
		"factual": sum(record["label"] == 0 for record in records),
		"hallucinated": sum(record["label"] == 1 for record in records),
	}
	print(f"Prepared {len(records)} rows from HaluEval QA.")
	print(f"Label counts: {label_counts}")


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


def save_baseline_outputs(
	predictions_path: Path,
	metrics_path: Path,
	predictions,
	metrics: dict[str, float],
) -> None:
	ensure_parent(predictions_path)
	ensure_parent(metrics_path)
	predictions.to_csv(predictions_path, index=False)
	metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
	print(f"Saved predictions to {predictions_path}")
	print(f"Saved metrics to {metrics_path}")

def main() -> None:
	args = parse_args()

	data_dir = Path(args.data_dir)
	results_dir = Path(args.results_dir)
	scored_dir = results_dir / "scored"
	memory_scored_dir = scored_dir / "memory"
	context_scored_dir = scored_dir / "context"

	if not args.skip_prepare:
		prepare_dataset(data_dir, args.limit, args.seed, args.overwrite)

	train_path = data_dir / "train.jsonl"
	eval_path = data_dir / f"{args.eval_split}.jsonl"
	if not train_path.exists() or not eval_path.exists():
		raise FileNotFoundError("Missing split files. Run without --skip-prepare first.")

	memory_train_path = memory_scored_dir / "train.jsonl"
	memory_eval_path = memory_scored_dir / f"{args.eval_split}.jsonl"
	context_eval_path = context_scored_dir / f"{args.eval_split}.jsonl"

	if not args.skip_inference:
		score_records(train_path, memory_train_path, "memory", args.model_name, args.overwrite, args.limit)
		score_records(eval_path, memory_eval_path, "memory", args.model_name, args.overwrite, args.limit)
		score_records(eval_path, context_eval_path, "context", args.model_name, args.overwrite, args.limit)

	if not args.skip_baselines:
		lexical_dir = results_dir / "lexical_svm" / args.eval_split
		lexical_pred_path = lexical_dir / "predictions.csv"
		lexical_metrics_path = lexical_dir / "metrics.json"
		predictions, metrics = run_lexical_svm(train_path, eval_path)
		save_baseline_outputs(lexical_pred_path, lexical_metrics_path, predictions, metrics)

		entropy_dir = results_dir / "entropy" / args.eval_split
		entropy_pred_path = entropy_dir / "predictions.csv"
		entropy_metrics_path = entropy_dir / "metrics.json"
		predictions, metrics = run_entropy_classifier(memory_train_path, memory_eval_path)
		save_baseline_outputs(entropy_pred_path, entropy_metrics_path, predictions, metrics)

		rag_dir = results_dir / "rag" / "compare" / args.eval_split
		rag_pred_path = rag_dir / "predictions.csv"
		rag_metrics_path = rag_dir / "metrics.json"
		predictions, metrics = run_rag_compare(memory_eval_path, context_eval_path)
		save_baseline_outputs(rag_pred_path, rag_metrics_path, predictions, metrics)

	if not args.skip_report_assets:
		report_table = Path("reports/tables/initial_results.csv")
		build_report_table(results_dir, report_table)
		build_plots(results_dir, Path("reports/figures"))


if __name__ == "__main__":
	main()
