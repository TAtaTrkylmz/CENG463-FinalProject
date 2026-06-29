import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.reporting import build_domain_comparison_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline metrics between two result directories, such as HaluEval vs open-domain evaluation."
    )
    parser.add_argument("--source-results", required=True, help="Reference results directory.")
    parser.add_argument("--target-results", required=True, help="Comparison results directory.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--source-label", default="in_domain")
    parser.add_argument("--target-label", default="out_of_domain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_domain_comparison_table(
        Path(args.source_results),
        Path(args.target_results),
        Path(args.output),
        source_label=args.source_label,
        target_label=args.target_label,
    )


if __name__ == "__main__":
    main()
