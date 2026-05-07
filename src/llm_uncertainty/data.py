from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from llm_uncertainty.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class SplitPaths:
    train: Path
    val: Path
    test: Path


def normalize_halueval_qa(limit: int | None = None) -> list[dict[str, Any]]:
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        base = {
            "source_dataset": "pminervini/HaluEval",
            "source_config": "qa",
            "source_split": "data",
            "original_sample_id": index,
            "knowledge": row["knowledge"],
            "question": row["question"],
        }
        records.append(
            {
                **base,
                "sample_id": f"qa-{index:05d}-factual",
                "candidate_answer": row["right_answer"],
                "label": 0,
                "label_name": "factual",
            }
        )
        records.append(
            {
                **base,
                "sample_id": f"qa-{index:05d}-hallucinated",
                "candidate_answer": row["hallucinated_answer"],
                "label": 1,
                "label_name": "hallucinated",
            }
        )
    return records


def write_splits(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    seed: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> SplitPaths:
    if len(records) < 10:
        raise ValueError("At least 10 normalized rows are required for stratified train/val/test splits.")

    output_dir = Path(output_dir)
    labels = [record["label"] for record in records]

    train_val, test = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [record["label"] for record in train_val]
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    paths = SplitPaths(
        train=output_dir / "train.jsonl",
        val=output_dir / "val.jsonl",
        test=output_dir / "test.jsonl",
    )
    write_jsonl(train, paths.train)
    write_jsonl(val, paths.val)
    write_jsonl(test, paths.test)
    return paths


def load_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if limit is not None:
        return records[:limit]
    return records
