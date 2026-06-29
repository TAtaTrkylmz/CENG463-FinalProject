from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import json
import ast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.baselines import _cached_feature_matrix, run_rag_compare_fixed
from llm_uncertainty.data import normalize_open_domain_records, validate_records, write_splits
from llm_uncertainty.error_analysis import build_error_analysis
from llm_uncertainty.io import read_jsonl, write_jsonl
from llm_uncertainty.reporting import build_comparison_assets, build_domain_comparison_table
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

    def test_open_domain_records_are_normalized(self) -> None:
        rows = [
            {
                "question": "Who wrote Hamlet?",
                "candidate_answer": "William Shakespeare",
                "label": "factual",
                "reference_answer": "Shakespeare",
                "generator_model": "gpt-test",
            },
            {
                "question": "Who wrote Hamlet?",
                "candidate_answer": "Charles Dickens",
                "label": "hallucinated",
            },
        ]

        records = normalize_open_domain_records(
            rows,
            source_dataset="custom_eval",
            source_split="val",
        )

        self.assertEqual(records[0]["sample_id"], "val-00000")
        self.assertEqual(records[0]["label"], 0)
        self.assertEqual(records[1]["label_name"], "hallucinated")
        validate_records(records)


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


class MainScriptTests(unittest.TestCase):
    def test_external_eval_uses_dataset_scoped_scored_paths(self) -> None:
        from main import experiment_paths
        import argparse

        args = argparse.Namespace(
            data_dir="data/processed/halueval_qa",
            train_data_dir="data/processed/halueval_qa",
            eval_data_dir="data/processed/truthfulqa_eval",
            scored_dir="results/scored",
            upgraded_scored_dir="results/scored_upgraded",
            eval_split="val",
        )
        paths = experiment_paths(args)
        self.assertIn("halueval_qa", str(paths["base_train_memory"]))
        self.assertIn("truthfulqa_eval", str(paths["base_eval_memory"]))


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
            self.assertTrue(
                (figures_dir / "baseline_comparison_precision_recall_val.png").exists()
            )

    def test_domain_comparison_table_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "results_halueval"
            target_dir = root / "results_open_domain"
            for result_dir, accuracy in ((source_dir, 0.9), (target_dir, 0.7)):
                run_dir = result_dir / "entropy_base" / "val"
                run_dir.mkdir(parents=True)
                (run_dir / "metrics.json").write_text(
                    json.dumps({"accuracy": accuracy, "macro_f1": accuracy, "auroc": accuracy}),
                    encoding="utf-8",
                )

            output_path = root / "reports" / "tables" / "comparison.csv"
            build_domain_comparison_table(
                source_dir,
                target_dir,
                output_path,
                source_label="halueval",
                target_label="open_domain",
            )

            frame = pd.read_csv(output_path)
            self.assertEqual(frame.loc[0, "baseline"], "entropy_base")
            self.assertAlmostEqual(frame.loc[0, "delta_accuracy"], -0.2)

    def test_error_analysis_assets_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for baseline, predictions in {
                "lexical_svm": [0, 1, 0, 1],
                "entropy_base": [0, 0, 1, 1],
            }.items():
                run_dir = root / "results" / baseline / "val"
                run_dir.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "sample_id": [f"sample-{index}" for index in range(4)],
                        "label": [0, 0, 1, 1],
                        "label_name": ["factual", "factual", "hallucinated", "hallucinated"],
                        "candidate_answer": ["a", "short answer", "medium answer", "long answer"],
                        "prediction": predictions,
                        "hallucination_score": [0.1, 0.7, 0.4, 0.9],
                    }
                ).to_csv(run_dir / "predictions.csv", index=False)

            output_dir = root / "reports" / "error_analysis" / "val"
            figures_dir = root / "reports" / "figures" / "error_analysis"
            build_error_analysis(
                root / "results",
                output_dir,
                figures_dir,
                eval_split="val",
                top_k=2,
            )

            self.assertTrue((output_dir / "summary_by_baseline.csv").exists())
            self.assertTrue((output_dir / "hardest_shared_errors.csv").exists())
            self.assertTrue((figures_dir / "confusion_grid_val.png").exists())
            self.assertTrue((figures_dir / "error_overlap_jaccard_val.png").exists())


class ScriptShapeTests(unittest.TestCase):
    def test_prepare_truthfulqa_script_parses(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_truthfulqa_eval.py"
        ast.parse(script_path.read_text(encoding="utf-8"))

    def test_generate_truthfulqa_script_parses(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_truthfulqa_model_outputs.py"
        ast.parse(script_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
