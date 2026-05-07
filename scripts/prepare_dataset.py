import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.data import normalize_halueval_qa, write_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare HaluEval QA.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of original QA samples to use.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed/halueval_qa")
    return parser.parse_args()


def main() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/external").mkdir(parents=True, exist_ok=True)
    args = parse_args()
    records = normalize_halueval_qa(limit=args.limit)
    paths = write_splits(records, args.output_dir, seed=args.seed)
    label_counts = {
        "factual": sum(record["label"] == 0 for record in records),
        "hallucinated": sum(record["label"] == 1 for record in records),
    }
    print(f"Prepared {len(records)} rows from HaluEval QA.")
    print(f"Label counts: {label_counts}")
    print(f"Train: {paths.train}")
    print(f"Validation: {paths.val}")
    print(f"Test: {paths.test}")


if __name__ == "__main__":
    main()
