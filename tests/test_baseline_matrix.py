from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.baselines import _cached_feature_matrix, run_rag_compare_fixed
from llm_uncertainty.data import write_splits
from llm_uncertainty.io import read_jsonl, write_jsonl
from llm_uncertainty.reporting import build_comparison_assets
from llm_uncertainty.representations import (
    resolve_nli_label_indices,
    semantic_interaction_features,
)


class GroupedSplitTests(unittest.TestCase):
    def test_original_question_never_crosses_splits(self) -> None:
        records = []
        for group_id in range(30):
            for label in (0, 1):
                records.append(
                    {
                        "original_sample_id": group_id,
                        "sample_id": f"{group_id}-{label}",
                        "question": f"question {group_id}",
                        "candidate_answer": f"answer {label}",
                        "label": label,
                        "label_name": "hallucinated" if label else "factual",
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            paths = write_splits(records, directory, seed=42)
            split_groups = []
            for path in (paths.train, paths.val, paths.test):
                groups = {row["original_sample_id"] for row in read_jsonl(path)}
                split_groups.append(groups)

        self.assertFalse(split_groups[0] & split_groups[1])
        self.assertFalse(split_groups[0] & split_groups[2])
        self.assertFalse(split_groups[1] & split_groups[2])


class RepresentationTests(unittest.TestCase):
    def test_semantic_interactions_include_true_cosine(self) -> None:
        questions = np.array([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        answers = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        features = semantic_interaction_features(questions, answers)

        self.assertEqual(features.shape, (2, 9))
        np.testing.assert_allclose(features[:, -1], np.array([1.0, 0.0]), atol=1e-6)

    def test_nli_label_mapping_uses_names(self) -> None:
        indices = resolve_nli_label_indices(
            {0: "ENTAILMENT", 1: "CONTRADICTION", 2: "NEUTRAL"}
        )
        self.assertEqual(indices.entailment, 0)
        self.assertEqual(indices.neutral, 2)
        self.assertEqual(indices.contradiction, 1)

    def test_feature_cache_reuses_matrix(self) -> None:
        calls = 0

        def compute() -> np.ndarray:
            nonlocal calls
            calls += 1
            return np.array([[1.0, 2.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            first = _cached_feature_matrix(
                Path(directory) / "cache",
                "semantic",
                source,
                "test-model",
                32,
                compute,
            )
            second = _cached_feature_matrix(
                Path(directory) / "cache",
                "semantic",
                source,
                "test-model",
                32,
                compute,
            )

        self.assertEqual(calls, 1)
        np.testing.assert_array_equal(first, second)


class FixedRagTests(unittest.TestCase):
    @staticmethod
    def _record(index: int, label: int, negative_mean_logprob: float) -> dict:
        return {
            "sample_id": f"sample-{index}",
            "label": label,
            "label_name": "hallucinated" if label else "factual",
            "candidate_answer": f"answer {index}",
            "negative_mean_logprob": negative_mean_logprob,
        }

    def test_threshold_is_learned_from_training_deltas(self) -> None:
        train_memory = []
        train_context = []
        eval_memory = []
        eval_context = []
        for index, label in enumerate([0, 0, 0, 1, 1, 1]):
            train_memory.append(self._record(index, label, 5.0))
            context_nll = 2.0 if label == 0 else 5.5
            train_context.append(self._record(index, label, context_nll))
        for index, label in enumerate([0, 0, 1, 1], start=10):
            eval_memory.append(self._record(index, label, 5.0))
            context_nll = 2.5 if label == 0 else 6.0
            eval_context.append(self._record(index, label, context_nll))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "train_memory": root / "train_memory.jsonl",
                "train_context": root / "train_context.jsonl",
                "eval_memory": root / "eval_memory.jsonl",
                "eval_context": root / "eval_context.jsonl",
            }
            write_jsonl(train_memory, paths["train_memory"])
            write_jsonl(train_context, paths["train_context"])
            write_jsonl(eval_memory, paths["eval_memory"])
            write_jsonl(eval_context, paths["eval_context"])
            _, metrics = run_rag_compare_fixed(**{f"{key}_path": value for key, value in paths.items()})

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["model_type"], "rag_compare_fixed")


class ReportingTests(unittest.TestCase):
    def test_comparison_assets_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "results" / "example_baseline" / "val"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                json.dumps({"accuracy": 0.75, "macro_f1": 0.74, "auroc": 0.8}),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "label": [0, 0, 1, 1],
                    "prediction": [0, 1, 0, 1],
                    "hallucination_score": [0.1, 0.6, 0.4, 0.9],
                }
            ).to_csv(run_dir / "predictions.csv", index=False)

            figures_dir = root / "reports" / "figures"
            tables_dir = root / "reports" / "tables"
            build_comparison_assets(root / "results", figures_dir, tables_dir, "val")

            self.assertTrue((tables_dir / "baseline_comparison_val.csv").exists())
            self.assertTrue((figures_dir / "baseline_comparison_metrics_val.png").exists())
            self.assertTrue((figures_dir / "baseline_comparison_roc_val.png").exists())


if __name__ == "__main__":
    unittest.main()
