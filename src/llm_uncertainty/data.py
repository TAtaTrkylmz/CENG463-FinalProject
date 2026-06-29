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


REQUIRED_RECORD_COLUMNS = {"sample_id", "question", "candidate_answer", "label", "label_name"}


def _normalize_label(value: Any) -> tuple[int, str]:
    if isinstance(value, bool):
        label = int(value)
    elif isinstance(value, int):
        label = value
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "factual", "fact", "supported", "true", "correct"}:
            label = 0
        elif normalized in {"1", "hallucinated", "hallucination", "unsupported", "false", "incorrect"}:
            label = 1
        else:
            raise ValueError(f"Unsupported label value: {value!r}")
    else:
        raise ValueError(f"Unsupported label type: {type(value).__name__}")

    if label not in {0, 1}:
        raise ValueError(f"Label must be 0 or 1, got {label!r}")
    return label, "hallucinated" if label == 1 else "factual"


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


def normalize_open_domain_records(
    rows: list[dict[str, Any]],
    *,
    source_dataset: str,
    source_split: str,
    source_config: str | None = None,
    generator_model: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if "question" not in row or "candidate_answer" not in row or "label" not in row:
            raise ValueError(
                "Each input row must contain question, candidate_answer, and label fields."
            )
        label, label_name = _normalize_label(row["label"])
        sample_id = str(row.get("sample_id") or f"{source_split}-{index:05d}")
        original_sample_id = row.get("original_sample_id", sample_id)
        record = {
            "sample_id": sample_id,
            "original_sample_id": original_sample_id,
            "source_dataset": row.get("source_dataset", source_dataset),
            "source_config": row.get("source_config", source_config),
            "source_split": row.get("source_split", source_split),
            "question": str(row["question"]),
            "knowledge": str(row.get("knowledge", "")),
            "candidate_answer": str(row["candidate_answer"]),
            "label": label,
            "label_name": row.get("label_name", label_name),
        }
        if generator_model is not None and "generator_model" not in row:
            record["generator_model"] = generator_model
        for optional_key in (
            "reference_answer",
            "generator_model",
            "notes",
            "topic",
            "metadata",
        ):
            if optional_key in row:
                record[optional_key] = row[optional_key]
        records.append(record)

    validate_records(records)
    return records


def write_splits(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    seed: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> SplitPaths:
    group_ids = sorted({record["original_sample_id"] for record in records})
    if len(group_ids) < 10:
        raise ValueError("At least 10 original QA groups are required for train/val/test splits.")

    output_dir = Path(output_dir)
    train_val_groups, test_groups = train_test_split(
        group_ids,
        test_size=test_size,
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=relative_val_size,
        random_state=seed,
    )
    train_group_set = set(train_groups)
    val_group_set = set(val_groups)
    test_group_set = set(test_groups)
    train = [record for record in records if record["original_sample_id"] in train_group_set]
    val = [record for record in records if record["original_sample_id"] in val_group_set]
    test = [record for record in records if record["original_sample_id"] in test_group_set]

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
    validate_records(records)
    if limit is not None:
        return records[:limit]
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    sample_ids: set[str] = set()
    for index, record in enumerate(records):
        missing = REQUIRED_RECORD_COLUMNS - set(record)
        if missing:
            raise ValueError(f"Record {index} is missing required columns: {sorted(missing)}")
        sample_id = str(record["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id detected: {sample_id}")
        sample_ids.add(sample_id)
        _, label_name = _normalize_label(record["label"])
        if str(record["label_name"]).strip().lower() != label_name:
            raise ValueError(
                f"Record {sample_id} has label_name={record['label_name']!r}, expected {label_name!r}"
            )
