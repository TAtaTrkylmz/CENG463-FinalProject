import argparse
import csv
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.io import ensure_parent, read_jsonl


PROMPT_LEAK_PATTERNS = {
    "you_are_ai_assistant": "You are an AI assistant",
    "opt_block": "OPT:",
    "correct_answer_phrase": "The correct answer is",
    "this_answer_phrase": "This answer is",
    "justifies_phrase": "This justifies what answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze generated TruthfulQA outputs for prompt leakage and weak auto-labeling."
    )
    parser.add_argument("--input", required=True, help="Evaluation JSONL from generate_truthfulqa_model_outputs.py")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for suspicious factual examples.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.01,
        help="Absolute entailment-margin threshold for weak labels.",
    )
    parser.add_argument(
        "--entailment-threshold",
        type=float,
        default=0.05,
        help="Threshold below which both entailment scores are considered weak.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=12,
        help="Number of suspicious factual examples to print.",
    )
    return parser.parse_args()


def answer_preview(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text[:limit]


def has_prompt_leak(answer: str) -> bool:
    return any(pattern in answer for pattern in PROMPT_LEAK_PATTERNS.values())


def prompt_leak_tags(answer: str) -> str:
    tags = [name for name, pattern in PROMPT_LEAK_PATTERNS.items() if pattern in answer]
    return ",".join(tags)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    total = len(rows)
    label_counts = Counter(row["label_name"] for row in rows)
    topic_counts = Counter(row.get("topic", "unknown") for row in rows)

    print(f"Rows: {total}")
    print("Label counts:")
    for label_name, count in sorted(label_counts.items()):
        print(f"  {label_name}: {count} ({count / total:.1%})")

    print("\nPrompt leakage markers:")
    for name, pattern in PROMPT_LEAK_PATTERNS.items():
        count = sum(pattern in row["candidate_answer"] for row in rows)
        print(f"  {name}: {count} ({count / total:.1%})")
    any_prompt_leak = [row for row in rows if has_prompt_leak(row["candidate_answer"])]
    print(f"  any_prompt_leak: {len(any_prompt_leak)} ({len(any_prompt_leak) / total:.1%})")

    weak_both = [
        row
        for row in rows
        if row["correct_entailment"] < args.entailment_threshold
        and row["incorrect_entailment"] < args.entailment_threshold
    ]
    weak_margin = [
        row
        for row in rows
        if abs(row["correct_entailment"] - row["incorrect_entailment"]) < args.margin_threshold
    ]
    print("\nWeak auto-label indicators:")
    print(f"  both entailments < {args.entailment_threshold:.2f}: {len(weak_both)} ({len(weak_both) / total:.1%})")
    print(f"  absolute margin < {args.margin_threshold:.2f}: {len(weak_margin)} ({len(weak_margin) / total:.1%})")

    suspicious_factuals = []
    for row in rows:
        margin = row["correct_entailment"] - row["incorrect_entailment"]
        both_low = (
            row["correct_entailment"] < args.entailment_threshold
            and row["incorrect_entailment"] < args.entailment_threshold
        )
        tiny_margin = abs(margin) < args.margin_threshold
        prompt_leak = has_prompt_leak(row["candidate_answer"])
        if row["label"] == 0 and (prompt_leak or both_low or tiny_margin):
            suspicious_factuals.append(
                {
                    "sample_id": row["sample_id"],
                    "topic": row.get("topic", "unknown"),
                    "question": row["question"],
                    "label_name": row["label_name"],
                    "correct_entailment": row["correct_entailment"],
                    "incorrect_entailment": row["incorrect_entailment"],
                    "margin": margin,
                    "both_low": both_low,
                    "tiny_margin": tiny_margin,
                    "prompt_leak": prompt_leak,
                    "prompt_leak_tags": prompt_leak_tags(row["candidate_answer"]),
                    "answer_preview": answer_preview(row["candidate_answer"]),
                }
            )

    suspicious_factuals.sort(
        key=lambda row: (
            not row["prompt_leak"],
            not row["both_low"],
            abs(row["margin"]),
        )
    )

    print("\nTop suspicious factual examples:")
    for row in suspicious_factuals[: args.examples]:
        print("---")
        print(
            f"{row['sample_id']} | topic={row['topic']} | margin={row['margin']:.4f} | "
            f"correct={row['correct_entailment']:.4f} | incorrect={row['incorrect_entailment']:.4f}"
        )
        print(row["question"])
        print(row["answer_preview"])

    weak_topics = Counter(row.get("topic", "unknown") for row in weak_both)
    leak_topics = Counter(row.get("topic", "unknown") for row in any_prompt_leak)
    print("\nTop topics among weak auto-label rows:")
    for topic, count in weak_topics.most_common(8):
        print(f"  {topic}: {count}")

    print("\nTop topics among prompt-leak rows:")
    for topic, count in leak_topics.most_common(8):
        print(f"  {topic}: {count}")

    if args.output_csv:
        output_path = ensure_parent(args.output_csv)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "topic",
                    "question",
                    "label_name",
                    "correct_entailment",
                    "incorrect_entailment",
                    "margin",
                    "both_low",
                    "tiny_margin",
                    "prompt_leak",
                    "prompt_leak_tags",
                    "answer_preview",
                ],
            )
            writer.writeheader()
            writer.writerows(suspicious_factuals)
        print(f"\nSaved suspicious factual rows to {output_path}")


if __name__ == "__main__":
    main()
