import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.baselines import FEATURE_COLUMNS
from llm_uncertainty.features import add_rag_features
from llm_uncertainty.io import ensure_parent, read_jsonl
from llm_uncertainty.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiments for hybrid hallucination detection.")
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--eval-text", required=True)
    parser.add_argument("--train-memory", required=True)
    parser.add_argument("--eval-memory", required=True)
    parser.add_argument("--eval-context", required=True)
    parser.add_argument("--out-dir", default="results/ablation/val")
    return parser.parse_args()


def _frame(path: str) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def _with_context_features(memory_path: str, context_path: str) -> pd.DataFrame:
    memory = {row["sample_id"]: row for row in read_jsonl(memory_path)}
    context = read_jsonl(context_path)
    rows = []
    for c in context:
        m = memory[c["sample_id"]]
        rows.append({"sample_id": c["sample_id"], **add_rag_features(m, c)})
    return pd.DataFrame(rows)


def _run_setting(
    name: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    use_text: bool,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    print(f"[ablation:{name}] Preparing matrices...")
    x_train_parts = []
    x_eval_parts = []
    if use_text:
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
        x_train_parts.append(vec.fit_transform(train["candidate_answer"]))
        x_eval_parts.append(vec.transform(eval_frame["candidate_answer"]))
    if numeric_cols:
        x_train_parts.append(csr_matrix(train[numeric_cols].astype(float).to_numpy()))
        x_eval_parts.append(csr_matrix(eval_frame[numeric_cols].astype(float).to_numpy()))

    x_train = x_train_parts[0] if len(x_train_parts) == 1 else hstack(x_train_parts, format="csr")
    x_eval = x_eval_parts[0] if len(x_eval_parts) == 1 else hstack(x_eval_parts, format="csr")
    class_counts = train["label"].value_counts().to_dict()
    print(
        f"[ablation:{name}] train_rows={len(train)} eval_rows={len(eval_frame)} "
        f"train_shape={x_train.shape} eval_shape={x_eval.shape} class_dist={class_counts}"
    )

    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    print(f"[ablation:{name}] Training LogisticRegression (solver=lbfgs, max_iter=2000)...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, train["label"])
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    n_iter = int(model.n_iter_[0]) if hasattr(model, "n_iter_") else -1
    converged = n_iter < model.max_iter and len(convergence_warnings) == 0
    print(f"[ablation:{name}] Training finished. n_iter={n_iter}, converged={converged}")
    if convergence_warnings:
        print(
            f"[ablation:{name}] ConvergenceWarning: optimizer hit iteration limit; "
            "outputs are still saved."
        )

    preds = model.predict(x_eval)
    scores = model.predict_proba(x_eval)[:, 1]
    out = eval_frame[["sample_id", "label", "label_name", "candidate_answer"]].copy()
    out["prediction"] = preds
    out["hallucination_score"] = scores
    metrics = classification_metrics(out["label"].tolist(), out["prediction"].tolist(), out["hallucination_score"].tolist())
    metrics["setting"] = name
    metrics["train_rows"] = float(len(train))
    metrics["eval_rows"] = float(len(eval_frame))
    metrics["feature_count_total"] = float(x_train.shape[1])
    metrics["optimizer_n_iter"] = float(n_iter)
    metrics["optimizer_converged"] = 1.0 if converged else 0.0
    return out, metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_parent(out_dir / "dummy.txt")

    print("[ablation] Loading input files...")
    train_text = _frame(args.train_text)
    eval_text = _frame(args.eval_text)
    train_memory = _frame(args.train_memory)
    eval_memory = _frame(args.eval_memory)
    eval_context_feats = _with_context_features(args.eval_memory, args.eval_context)
    print(
        f"[ablation] Loaded rows train_text={len(train_text)} eval_text={len(eval_text)} "
        f"train_memory={len(train_memory)} eval_memory={len(eval_memory)} "
        f"context_features={len(eval_context_feats)}"
    )

    train = train_text.merge(train_memory[["sample_id"] + FEATURE_COLUMNS], on="sample_id", how="inner")
    eval_base = eval_text.merge(eval_memory[["sample_id"] + FEATURE_COLUMNS], on="sample_id", how="inner")
    eval_with_context = eval_base.merge(
        eval_context_feats[["sample_id", "context_improvement", "absolute_context_delta"]],
        on="sample_id",
        how="left",
    )
    eval_with_context[["context_improvement", "absolute_context_delta"]] = eval_with_context[
        ["context_improvement", "absolute_context_delta"]
    ].fillna(0.0)
    train["context_improvement"] = 0.0
    train["absolute_context_delta"] = 0.0
    print(f"[ablation] After merge train={len(train)} eval_base={len(eval_base)} eval_with_context={len(eval_with_context)}")

    settings = [
        ("lexical_only", True, []),
        ("uncertainty_only", False, FEATURE_COLUMNS),
        ("hybrid_no_context", True, FEATURE_COLUMNS),
        ("hybrid_with_context", True, FEATURE_COLUMNS + ["context_improvement", "absolute_context_delta"]),
    ]

    metrics_rows = []
    for name, use_text, cols in settings:
        eval_frame = eval_with_context if "context" in name else eval_base
        preds, metrics = _run_setting(name, train, eval_frame, use_text, cols)
        preds.to_csv(out_dir / f"{name}_predictions.csv", index=False)
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics_rows.append(metrics)
        print(f"Saved ablation outputs for {name}")

    pd.DataFrame(metrics_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved summary to {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
