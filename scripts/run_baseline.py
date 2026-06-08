import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.baselines import (
    run_cross_encoder_classifier,
    run_entropy_classifier,
    run_evidence_aware_hybrid,
    run_hybrid_proposed,
    run_lexical_hybrid_svm,
    run_lexical_svm,
    run_nli_evidence_zero_shot,
    run_rag_compare,
    run_rag_compare_fixed,
    run_semantic_hybrid_svm,
    run_semantic_svm,
    run_upgraded_entropy_classifier,
)
from llm_uncertainty.io import ensure_parent
from llm_uncertainty.representations import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_NLI_MODEL,
    DEFAULT_SEMANTIC_MODEL,
)


CANONICAL_BASELINES = [
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

LEGACY_ALIASES = ["entropy", "rag_compare", "hybrid_proposed", "hybrid_svm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one of the 10 hallucination-detection baselines.")
    parser.add_argument(
        "--baseline",
        choices=CANONICAL_BASELINES + LEGACY_ALIASES,
        required=True,
    )
    parser.add_argument("--train", help="Training text JSONL, or scored JSONL for entropy baselines.")
    parser.add_argument("--eval", help="Evaluation text JSONL, or scored JSONL for entropy baselines.")
    parser.add_argument("--memory", help="Legacy memory-mode scored JSONL for rag_compare.")
    parser.add_argument("--context", help="Legacy context-mode scored JSONL for rag_compare.")
    parser.add_argument("--train-memory", help="Training memory-mode scored JSONL.")
    parser.add_argument("--eval-memory", help="Evaluation memory-mode scored JSONL.")
    parser.add_argument("--train-context", help="Training context-mode scored JSONL.")
    parser.add_argument("--eval-context", help="Evaluation context-mode scored JSONL.")
    parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--device", help="Torch device, for example cpu, cuda, or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--semantic-max-length", type=int, default=256)
    parser.add_argument("--nli-max-length", type=int, default=384)
    parser.add_argument("--cross-encoder-max-length", type=int, default=256)
    parser.add_argument("--nli-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--model-output-dir")
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    return parser.parse_args()


def _require(args: argparse.Namespace, names: list[str]) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"{args.baseline} requires {' '.join(missing)}")


def run_selected(args: argparse.Namespace):
    if args.baseline == "lexical_svm":
        _require(args, ["train", "eval"])
        return run_lexical_svm(args.train, args.eval)

    if args.baseline in {"entropy", "entropy_base"}:
        _require(args, ["train", "eval"])
        return run_entropy_classifier(args.train, args.eval)

    if args.baseline == "entropy_upgraded":
        _require(args, ["train", "eval"])
        return run_upgraded_entropy_classifier(args.train, args.eval)

    if args.baseline in {"lexical_hybrid_svm", "hybrid_svm"}:
        _require(args, ["train", "eval", "train_memory", "eval_memory"])
        return run_lexical_hybrid_svm(
            train_text_path=args.train,
            train_memory_path=args.train_memory,
            eval_text_path=args.eval,
            eval_memory_path=args.eval_memory,
        )

    if args.baseline == "hybrid_proposed":
        _require(args, ["train", "eval", "train_memory", "eval_memory"])
        return run_hybrid_proposed(
            train_text_path=args.train,
            train_memory_path=args.train_memory,
            eval_text_path=args.eval,
            eval_memory_path=args.eval_memory,
            eval_context_path=args.eval_context,
        )

    if args.baseline == "semantic_svm":
        _require(args, ["train", "eval"])
        return run_semantic_svm(
            args.train,
            args.eval,
            model_name=args.semantic_model,
            batch_size=args.batch_size,
            max_length=args.semantic_max_length,
            device=args.device,
            feature_cache_dir=args.feature_cache_dir,
        )

    if args.baseline == "semantic_hybrid_svm":
        _require(args, ["train", "eval", "train_memory", "eval_memory"])
        return run_semantic_hybrid_svm(
            train_text_path=args.train,
            train_memory_path=args.train_memory,
            eval_text_path=args.eval,
            eval_memory_path=args.eval_memory,
            model_name=args.semantic_model,
            batch_size=args.batch_size,
            max_length=args.semantic_max_length,
            device=args.device,
            feature_cache_dir=args.feature_cache_dir,
        )

    if args.baseline == "rag_compare":
        _require(args, ["memory", "context"])
        return run_rag_compare(args.memory, args.context)

    if args.baseline == "rag_compare_fixed":
        _require(args, ["train_memory", "train_context", "eval_memory", "eval_context"])
        return run_rag_compare_fixed(
            train_memory_path=args.train_memory,
            train_context_path=args.train_context,
            eval_memory_path=args.eval_memory,
            eval_context_path=args.eval_context,
        )

    if args.baseline == "nli_evidence":
        _require(args, ["eval"])
        return run_nli_evidence_zero_shot(
            args.eval,
            model_name=args.nli_model,
            batch_size=args.batch_size,
            max_length=args.nli_max_length,
            device=args.device,
            threshold=args.nli_threshold,
            feature_cache_dir=args.feature_cache_dir,
        )

    if args.baseline == "evidence_aware_hybrid":
        _require(args, ["train", "eval", "train_memory", "eval_memory"])
        return run_evidence_aware_hybrid(
            train_text_path=args.train,
            train_memory_path=args.train_memory,
            eval_text_path=args.eval,
            eval_memory_path=args.eval_memory,
            semantic_model_name=args.semantic_model,
            nli_model_name=args.nli_model,
            batch_size=args.batch_size,
            semantic_max_length=args.semantic_max_length,
            nli_max_length=args.nli_max_length,
            device=args.device,
            feature_cache_dir=args.feature_cache_dir,
        )

    if args.baseline == "cross_encoder":
        _require(args, ["train", "eval"])
        return run_cross_encoder_classifier(
            args.train,
            args.eval,
            model_name=args.cross_encoder_model,
            batch_size=args.batch_size,
            max_length=args.cross_encoder_max_length,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=args.device,
            model_output_dir=args.model_output_dir,
        )

    raise ValueError(f"Unsupported baseline: {args.baseline}")


def main() -> None:
    args = parse_args()
    predictions, metrics = run_selected(args)
    metrics["baseline"] = args.baseline

    predictions_path = ensure_parent(args.predictions_output)
    metrics_path = ensure_parent(args.metrics_output)
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
