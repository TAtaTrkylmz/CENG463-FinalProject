import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.data import normalize_open_domain_records
from llm_uncertainty.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize an external open-domain hallucination benchmark into the project's JSONL schema."
    )
    parser.add_argument("--input", required=True, help="Input JSONL with question/candidate_answer/label.")
    parser.add_argument("--output-dir", required=True, help="Directory where the normalized split will be written.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--source-dataset", default="custom_open_domain")
    parser.add_argument("--source-config", default=None)
    parser.add_argument("--generator-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    records = normalize_open_domain_records(
        rows,
        source_dataset=args.source_dataset,
        source_split=args.split,
        source_config=args.source_config,
        generator_model=args.generator_model,
    )
    output_path = Path(args.output_dir) / f"{args.split}.jsonl"
    write_jsonl(records, output_path)
    print(f"Saved {len(records)} normalized records to {output_path}")


if __name__ == "__main__":
    main()
