from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from llm_uncertainty.baselines import (
    run_cross_encoder_classifier,
    run_entropy_classifier,
    run_evidence_aware_hybrid,
    run_evidence_aware_length_hybrid,
    run_lexical_hybrid_svm,
    run_lexical_svm,
    run_nli_evidence_zero_shot,
    run_rag_compare_fixed,
    run_semantic_hybrid_svm,
    run_semantic_svm,
    run_upgraded_entropy_classifier,
)
from llm_uncertainty.data import load_records, normalize_halueval_qa, write_splits
from llm_uncertainty.error_analysis import build_error_analysis
from llm_uncertainty.io import ensure_parent, write_jsonl
from llm_uncertainty.local_lm import LocalCausalLMScorer, score_record
from llm_uncertainty.reporting import (
    build_comparison_assets,
    build_plots,
    build_report_table,
)
from llm_uncertainty.representations import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_NLI_MODEL,
    DEFAULT_SEMANTIC_MODEL,
)


DEFAULT_BASELINES = [
    "lexical_svm",
    "lexical_hybrid_svm",
    "entropy_base",
    "entropy_upgraded",
    "semantic_svm",
    "semantic_hybrid_svm",
    "rag_compare_fixed",
    "nli_evidence",
    "evidence_aware_hybrid",
    "cross_encoder",
]

OPTIONAL_BASELINES = [
    "evidence_aware_length_hybrid",
]

BASELINES = DEFAULT_BASELINES + OPTIONAL_BASELINES


def _dataset_cache_name(path: Path) -> str:
    name = path.resolve().name.strip()
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return safe or "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete 10-baseline hallucination-detection experiment."
    )
    parser.add_argument("--data-dir", default="data/processed/halueval_qa")
    parser.add_argument("--train-data-dir", default=None)
    parser.add_argument("--eval-data-dir", default=None)
    parser.add_argument("--results-dir", default="results/matrix")
    parser.add_argument("--scored-dir", default="results/scored")
    parser.add_argument("--upgraded-scored-dir", default="results/scored_upgraded")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=DEFAULT_BASELINES)

    parser.add_argument("--base-model-name", default="distilgpt2")
    parser.add_argument("--upgraded-model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--device", help="Torch device, for example cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--base-model-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--upgraded-model-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--upgraded-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--semantic-max-length", type=int, default=256)
    parser.add_argument("--nli-max-length", type=int, default=384)
    parser.add_argument("--cross-encoder-max-length", type=int, default=256)
    parser.add_argument("--nli-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--short-answer-threshold", type=int, default=40)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-report-assets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_dataset(output_dir: Path, limit: int | None, seed: int, overwrite: bool) -> None:
    paths = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]
    if not overwrite and all(path.exists() for path in paths):
        print(f"[prepare] Using existing grouped splits under {output_dir}")
        return

    records = normalize_halueval_qa(limit=limit)
    split_paths = write_splits(records, output_dir, seed=seed)
    print(f"[prepare] Prepared {len(records)} balanced rows.")
    print(f"[prepare] Train={split_paths.train} Val={split_paths.val} Test={split_paths.test}")


def score_jobs(
    jobs: list[tuple[Path, Path, str]],
    model_name: str,
    device: str | None,
    model_dtype: str,
    use_chat_template: bool,
    overwrite: bool,
    limit: int | None,
) -> None:
    pending = [
        (input_path, output_path, mode)
        for input_path, output_path, mode in jobs
        if overwrite or not output_path.exists()
    ]
    for _, output_path, _ in jobs:
        if not overwrite and output_path.exists():
            print(f"[inference] Using existing scored file {output_path}")
    if not pending:
        return

    scorer = LocalCausalLMScorer(
        model_name=model_name,
        device=device,
        model_dtype=model_dtype,
        use_chat_template=use_chat_template,
    )
    for input_path, output_path, mode in pending:
        records = load_records(input_path, limit=limit)
        scored = [
            score_record(record, scorer, mode)
            for record in tqdm(records, desc=f"{model_name}:{mode}:{input_path.stem}")
        ]
        write_jsonl(scored, output_path)
        print(f"[inference] Saved {len(scored)} rows to {output_path}")


def save_baseline_outputs(
    baseline: str,
    output_dir: Path,
    run: Callable[[], tuple],
    overwrite: bool,
) -> None:
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    if not overwrite and predictions_path.exists() and metrics_path.exists():
        print(f"[baseline:{baseline}] Using existing outputs under {output_dir}")
        return

    print(f"[baseline:{baseline}] Running...")
    predictions, metrics = run()
    metrics["baseline"] = baseline
    ensure_parent(predictions_path)
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[baseline:{baseline}] Saved predictions and metrics to {output_dir}")


def baseline_runners(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Callable[[], tuple]]:
    feature_cache_dir = args.feature_cache_dir or str(Path(args.results_dir) / "feature_cache")
    return {
        "lexical_svm": lambda: run_lexical_svm(paths["train"], paths["eval"]),
        "lexical_hybrid_svm": lambda: run_lexical_hybrid_svm(
            paths["train"],
            paths["base_train_memory"],
            paths["eval"],
            paths["base_eval_memory"],
        ),
        "entropy_base": lambda: run_entropy_classifier(
            paths["base_train_memory"],
            paths["base_eval_memory"],
        ),
        "entropy_upgraded": lambda: run_upgraded_entropy_classifier(
            paths["upgraded_train_memory"],
            paths["upgraded_eval_memory"],
        ),
        "semantic_svm": lambda: run_semantic_svm(
            paths["train"],
            paths["eval"],
            model_name=args.semantic_model,
            batch_size=args.batch_size,
            max_length=args.semantic_max_length,
            device=args.device,
            feature_cache_dir=feature_cache_dir,
        ),
        "semantic_hybrid_svm": lambda: run_semantic_hybrid_svm(
            paths["train"],
            paths["base_train_memory"],
            paths["eval"],
            paths["base_eval_memory"],
            model_name=args.semantic_model,
            batch_size=args.batch_size,
            max_length=args.semantic_max_length,
            device=args.device,
            feature_cache_dir=feature_cache_dir,
        ),
        "rag_compare_fixed": lambda: run_rag_compare_fixed(
            paths["base_train_memory"],
            paths["base_train_context"],
            paths["base_eval_memory"],
            paths["base_eval_context"],
        ),
        "nli_evidence": lambda: run_nli_evidence_zero_shot(
            paths["eval"],
            model_name=args.nli_model,
            batch_size=args.batch_size,
            max_length=args.nli_max_length,
            device=args.device,
            threshold=args.nli_threshold,
            feature_cache_dir=feature_cache_dir,
        ),
        "evidence_aware_hybrid": lambda: run_evidence_aware_hybrid(
            paths["train"],
            paths["base_train_memory"],
            paths["eval"],
            paths["base_eval_memory"],
            semantic_model_name=args.semantic_model,
            nli_model_name=args.nli_model,
            batch_size=args.batch_size,
            semantic_max_length=args.semantic_max_length,
            nli_max_length=args.nli_max_length,
            device=args.device,
            feature_cache_dir=feature_cache_dir,
        ),
        "cross_encoder": lambda: run_cross_encoder_classifier(
            paths["train"],
            paths["eval"],
            model_name=args.cross_encoder_model,
            batch_size=args.batch_size,
            max_length=args.cross_encoder_max_length,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=args.device,
            model_output_dir=Path(args.results_dir)
            / "cross_encoder"
            / args.eval_split
            / "model",
        ),
        "evidence_aware_length_hybrid": lambda: run_evidence_aware_length_hybrid(
            paths["train"],
            paths["base_train_memory"],
            paths["eval"],
            paths["base_eval_memory"],
            semantic_model_name=args.semantic_model,
            nli_model_name=args.nli_model,
            batch_size=args.batch_size,
            semantic_max_length=args.semantic_max_length,
            nli_max_length=args.nli_max_length,
            device=args.device,
            feature_cache_dir=feature_cache_dir,
            short_token_threshold=args.short_answer_threshold,
        ),
    }


def experiment_paths(args: argparse.Namespace) -> dict[str, Path]:
    train_data_dir = Path(args.train_data_dir or args.data_dir)
    eval_data_dir = Path(args.eval_data_dir or args.data_dir)
    scored_dir = Path(args.scored_dir)
    upgraded_dir = Path(args.upgraded_scored_dir)
    train_name = _dataset_cache_name(train_data_dir)
    eval_name = _dataset_cache_name(eval_data_dir)
    shared_dataset = train_data_dir.resolve() == eval_data_dir.resolve()

    def score_path(root: Path, dataset_name: str, mode: str, split: str) -> Path:
        if shared_dataset:
            return root / mode / f"{split}.jsonl"
        return root / dataset_name / mode / f"{split}.jsonl"

    return {
        "train": train_data_dir / "train.jsonl",
        "eval": eval_data_dir / f"{args.eval_split}.jsonl",
        "base_train_memory": score_path(scored_dir, train_name, "memory", "train"),
        "base_eval_memory": score_path(scored_dir, eval_name, "memory", args.eval_split),
        "base_train_context": score_path(scored_dir, train_name, "context", "train"),
        "base_eval_context": score_path(scored_dir, eval_name, "context", args.eval_split),
        "upgraded_train_memory": score_path(upgraded_dir, train_name, "memory", "train"),
        "upgraded_eval_memory": score_path(upgraded_dir, eval_name, "memory", args.eval_split),
    }


def print_plan(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    print("[dry-run] Complete experiment plan")
    print(f"  data: {args.data_dir}")
    print(f"  train data dir: {args.train_data_dir or args.data_dir}")
    print(f"  eval data dir: {args.eval_data_dir or args.data_dir}")
    print(f"  results: {args.results_dir}")
    print(f"  split: {args.eval_split}")
    print(f"  baselines: {', '.join(args.baselines)}")
    print(f"  base LM: {args.base_model_name}")
    print(f"  upgraded LM: {args.upgraded_model_name}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def main() -> None:
    args = parse_args()
    paths = experiment_paths(args)
    if args.dry_run:
        print_plan(args, paths)
        return

    if not args.skip_prepare:
        prepare_dataset(Path(args.data_dir), args.limit, args.seed, args.overwrite)
    if not paths["train"].exists() or not paths["eval"].exists():
        raise FileNotFoundError("Missing train/evaluation split files.")

    if not args.skip_inference:
        base_jobs = [
            (paths["train"], paths["base_train_memory"], "memory"),
            (paths["eval"], paths["base_eval_memory"], "memory"),
            (paths["train"], paths["base_train_context"], "context"),
            (paths["eval"], paths["base_eval_context"], "context"),
        ]
        score_jobs(
            base_jobs,
            model_name=args.base_model_name,
            device=args.device,
            model_dtype=args.base_model_dtype,
            use_chat_template=False,
            overwrite=args.overwrite,
            limit=args.limit,
        )

        if "entropy_upgraded" in args.baselines:
            upgraded_jobs = [
                (paths["train"], paths["upgraded_train_memory"], "memory"),
                (paths["eval"], paths["upgraded_eval_memory"], "memory"),
            ]
            score_jobs(
                upgraded_jobs,
                model_name=args.upgraded_model_name,
                device=args.device,
                model_dtype=args.upgraded_model_dtype,
                use_chat_template=args.upgraded_chat_template,
                overwrite=args.overwrite,
                limit=args.limit,
            )

    if not args.skip_baselines:
        runners = baseline_runners(args, paths)
        for baseline in args.baselines:
            output_dir = Path(args.results_dir) / baseline / args.eval_split
            save_baseline_outputs(
                baseline,
                output_dir,
                runners[baseline],
                overwrite=args.overwrite,
            )

    if not args.skip_report_assets:
        reports_dir = Path(args.reports_dir)
        figures_dir = reports_dir / "figures" / "matrix"
        tables_dir = reports_dir / "tables"
        results_dir = Path(args.results_dir)
        build_report_table(
            results_dir,
            tables_dir / f"baseline_results_{args.eval_split}.csv",
        )
        build_plots(results_dir, figures_dir)
        build_comparison_assets(
            results_dir,
            figures_dir,
            tables_dir,
            args.eval_split,
        )
        build_error_analysis(
            results_dir=results_dir,
            output_dir=reports_dir / "error_analysis" / args.eval_split,
            figures_dir=reports_dir / "figures" / "error_analysis",
            eval_split=args.eval_split,
            data_path=paths["eval"],
        )


if __name__ == "__main__":
    main()
