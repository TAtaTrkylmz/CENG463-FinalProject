import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.error_analysis import build_error_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparative error-analysis tables and figures."
    )
    parser.add_argument("--results-dir", default="results/matrix")
    parser.add_argument("--out-dir", default="reports/error_analysis")
    parser.add_argument("--figures-dir", default="reports/figures/error_analysis")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--data",
        help="Optional split JSONL used to add question, knowledge, and question type.",
    )
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else Path(
        f"data/processed/halueval_qa/{args.split}.jsonl"
    )
    outputs = build_error_analysis(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.out_dir) / args.split,
        figures_dir=Path(args.figures_dir),
        eval_split=args.split,
        data_path=data_path,
        top_k=args.top_k,
    )
    print(f"[error_analysis] Summary: {outputs['summary']}")
    print(f"[error_analysis] Findings: {outputs['findings']}")
    print(f"[error_analysis] Figures: {outputs['figures']}")


if __name__ == "__main__":
    main()
