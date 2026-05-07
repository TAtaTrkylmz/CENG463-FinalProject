import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.io import ensure_parent
from llm_uncertainty.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--target-column", default="label")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--score-column", default="hallucination_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    scores = frame[args.score_column].tolist() if args.score_column in frame.columns else None
    metrics = classification_metrics(
        frame[args.target_column].tolist(),
        frame[args.prediction_column].tolist(),
        scores=scores,
    )
    output_path = ensure_parent(args.metrics_output)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

