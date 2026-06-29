import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a TruthfulQA-based external evaluation set in the project's JSONL schema."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--incorrect-answer-policy",
        choices=["first", "all"],
        default="first",
        help="Use only the first incorrect answer per question, or expand to all incorrect answers.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _reference_knowledge(row: dict) -> str:
    correct = [answer.strip() for answer in row["correct_answers"] if answer and answer.strip()]
    best = row["best_answer"].strip()
    references = [best] + [answer for answer in correct if answer != best]
    return "Reference answers: " + " | ".join(references)


def build_records(split: str, incorrect_answer_policy: str, limit: int | None) -> list[dict]:
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records: list[dict] = []
    for index, row in enumerate(dataset):
        knowledge = _reference_knowledge(row)
        base = {
            "source_dataset": "truthfulqa/truthful_qa",
            "source_config": "generation",
            "source_split": "validation",
            "original_sample_id": f"truthfulqa-{index:05d}",
            "question": row["question"],
            "knowledge": knowledge,
            "reference_answer": row["best_answer"],
            "topic": row["category"],
            "notes": row["type"],
        }
        records.append(
            {
                **base,
                "sample_id": f"truthfulqa-{index:05d}-factual",
                "candidate_answer": row["best_answer"],
                "label": 0,
                "label_name": "factual",
                "generator_model": "truthfulqa_reference",
            }
        )

        incorrect_answers = [
            answer.strip() for answer in row["incorrect_answers"] if answer and answer.strip()
        ]
        if not incorrect_answers:
            continue
        if incorrect_answer_policy == "first":
            incorrect_answers = incorrect_answers[:1]
        for wrong_index, wrong_answer in enumerate(incorrect_answers):
            suffix = f"{wrong_index:02d}" if incorrect_answer_policy == "all" else "00"
            records.append(
                {
                    **base,
                    "sample_id": f"truthfulqa-{index:05d}-hallucinated-{suffix}",
                    "candidate_answer": wrong_answer,
                    "label": 1,
                    "label_name": "hallucinated",
                    "generator_model": "truthfulqa_reference",
                }
            )
    return records


def main() -> None:
    args = parse_args()
    records = build_records(args.split, args.incorrect_answer_policy, args.limit)
    output_path = Path(args.output_dir) / f"{args.split}.jsonl"
    write_jsonl(records, output_path)
    print(f"Saved {len(records)} TruthfulQA-derived records to {output_path}")


if __name__ == "__main__":
    main()
