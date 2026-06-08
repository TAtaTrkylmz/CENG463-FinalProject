from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


BASELINES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 10-baseline experimental matrix.")
    parser.add_argument("--data-dir", default="data/processed/halueval_qa")
    parser.add_argument("--results-dir", default="results/matrix")
    parser.add_argument("--base-scored-dir", default="results/scored")
    parser.add_argument(
        "--upgraded-scored-dir",
        default="results/scored_upgraded",
        help="Memory-mode train/eval files produced by the upgraded instruction-tuned LM.",
    )
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=BASELINES)
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", default="FacebookAI/roberta-large-mnli")
    parser.add_argument("--cross-encoder-model", default="distilroberta-base")
    parser.add_argument(
        "--feature-cache-dir",
        help="Shared semantic/NLI feature cache (default: <results-dir>/feature_cache).",
    )
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _common_paths(args: argparse.Namespace) -> dict[str, Path]:
    data_dir = Path(args.data_dir)
    scored_dir = Path(args.base_scored_dir)
    upgraded_dir = Path(args.upgraded_scored_dir)
    return {
        "train": data_dir / "train.jsonl",
        "eval": data_dir / f"{args.eval_split}.jsonl",
        "train_memory": scored_dir / "memory" / "train.jsonl",
        "eval_memory": scored_dir / "memory" / f"{args.eval_split}.jsonl",
        "train_context": scored_dir / "context" / "train.jsonl",
        "eval_context": scored_dir / "context" / f"{args.eval_split}.jsonl",
        "upgraded_train": upgraded_dir / "memory" / "train.jsonl",
        "upgraded_eval": upgraded_dir / "memory" / f"{args.eval_split}.jsonl",
    }


def _baseline_inputs(name: str, paths: dict[str, Path]) -> list[str]:
    text = ["--train", str(paths["train"]), "--eval", str(paths["eval"])]
    memory = ["--train-memory", str(paths["train_memory"]), "--eval-memory", str(paths["eval_memory"])]
    if name == "lexical_svm":
        return text
    if name == "lexical_hybrid_svm":
        return text + memory
    if name == "entropy_base":
        return ["--train", str(paths["train_memory"]), "--eval", str(paths["eval_memory"])]
    if name == "entropy_upgraded":
        return ["--train", str(paths["upgraded_train"]), "--eval", str(paths["upgraded_eval"])]
    if name == "semantic_svm":
        return text
    if name == "semantic_hybrid_svm":
        return text + memory
    if name == "rag_compare_fixed":
        return [
            "--train-memory",
            str(paths["train_memory"]),
            "--train-context",
            str(paths["train_context"]),
            "--eval-memory",
            str(paths["eval_memory"]),
            "--eval-context",
            str(paths["eval_context"]),
        ]
    if name == "nli_evidence":
        return ["--eval", str(paths["eval"])]
    if name == "evidence_aware_hybrid":
        return text + memory
    if name == "cross_encoder":
        return text
    raise ValueError(f"Unsupported baseline: {name}")


def _input_paths(command_args: list[str]) -> list[Path]:
    path_flags = {
        "--train",
        "--eval",
        "--train-memory",
        "--eval-memory",
        "--train-context",
        "--eval-context",
    }
    return [Path(command_args[index + 1]) for index, value in enumerate(command_args[:-1]) if value in path_flags]


def main() -> None:
    args = parse_args()
    paths = _common_paths(args)
    runner = Path(__file__).with_name("run_baseline.py")
    feature_cache_dir = args.feature_cache_dir or str(Path(args.results_dir) / "feature_cache")

    for baseline in args.baselines:
        baseline_inputs = _baseline_inputs(baseline, paths)
        missing = [path for path in _input_paths(baseline_inputs) if not path.exists()]
        if missing and not args.dry_run:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing inputs for {baseline}:\n{formatted}")

        output_dir = Path(args.results_dir) / baseline / args.eval_split
        command = [
            sys.executable,
            str(runner),
            "--baseline",
            baseline,
            *baseline_inputs,
            "--semantic-model",
            args.semantic_model,
            "--nli-model",
            args.nli_model,
            "--cross-encoder-model",
            args.cross_encoder_model,
            "--feature-cache-dir",
            feature_cache_dir,
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--predictions-output",
            str(output_dir / "predictions.csv"),
            "--metrics-output",
            str(output_dir / "metrics.json"),
        ]
        if args.device:
            command.extend(["--device", args.device])
        if baseline == "cross_encoder":
            command.extend(["--model-output-dir", str(output_dir / "model")])

        print(f"[matrix] {shlex.join(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
