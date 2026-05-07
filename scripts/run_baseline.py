import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.baselines import run_entropy_classifier, run_lexical_svm, run_rag_compare
from llm_uncertainty.io import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hallucination detection baselines.")
    parser.add_argument("--baseline", choices=["lexical_svm", "entropy", "rag_compare"], required=True)
    parser.add_argument("--train", help="Training JSONL path for supervised baselines.")
    parser.add_argument("--eval", help="Evaluation JSONL path for lexical/entropy baselines.")
    parser.add_argument("--memory", help="Memory-mode scored JSONL for RAG comparison.")
    parser.add_argument("--context", help="Context-mode scored JSONL for RAG comparison.")
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline == "lexical_svm":
        if not args.train or not args.eval:
            raise ValueError("lexical_svm requires --train and --eval")
        predictions, metrics = run_lexical_svm(args.train, args.eval)
    elif args.baseline == "entropy":
        if not args.train or not args.eval:
            raise ValueError("entropy requires --train and --eval")
        predictions, metrics = run_entropy_classifier(args.train, args.eval)
    else:
        if not args.memory or not args.context:
            raise ValueError("rag_compare requires --memory and --context")
        predictions, metrics = run_rag_compare(args.memory, args.context)

    predictions_path = ensure_parent(args.predictions_output)
    metrics_path = ensure_parent(args.metrics_output)
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

