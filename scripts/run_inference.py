import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.data import load_records
from llm_uncertainty.io import write_jsonl
from llm_uncertainty.local_lm import LocalCausalLMScorer, score_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score HaluEval answers with a local causal LM.")
    parser.add_argument("--input", required=True, help="Processed split JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--mode", choices=["memory", "context"], default="memory")
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input, limit=args.limit)
    scorer = LocalCausalLMScorer(model_name=args.model_name)
    scored = [score_record(record, scorer, args.mode) for record in tqdm(records, desc=f"Scoring {args.mode}")]
    output_path = write_jsonl(scored, args.output)
    print(f"Saved {len(scored)} scored records to {output_path}")


if __name__ == "__main__":
    main()

