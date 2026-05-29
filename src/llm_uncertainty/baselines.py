from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning

from llm_uncertainty.features import add_rag_features
from llm_uncertainty.io import read_jsonl
from llm_uncertainty.metrics import classification_metrics


FEATURE_COLUMNS = [
    "negative_mean_logprob",
    "mean_logprob",
    "sum_logprob",
    "min_logprob",
    "max_logprob",
    "perplexity",
    "token_count",
]


def records_to_frame(path: str | Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def run_lexical_svm(train_path: str | Path, eval_path: str | Path) -> tuple[pd.DataFrame, dict[str, float]]:
    train = records_to_frame(train_path)
    eval_frame = records_to_frame(eval_path)
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)),
            ("svm", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(train["candidate_answer"], train["label"])
    predictions = model.predict(eval_frame["candidate_answer"])
    scores = model.predict_proba(eval_frame["candidate_answer"])[:, 1]
    output = eval_frame[["sample_id", "label", "label_name", "candidate_answer"]].copy()
    output["prediction"] = predictions
    output["hallucination_score"] = scores
    return output, classification_metrics(output["label"].tolist(), output["prediction"].tolist(), scores.tolist())


def run_entropy_classifier(train_path: str | Path, eval_path: str | Path) -> tuple[pd.DataFrame, dict[str, float]]:
    train = records_to_frame(train_path)
    eval_frame = records_to_frame(eval_path)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(train[FEATURE_COLUMNS], train["label"])
    predictions = model.predict(eval_frame[FEATURE_COLUMNS])
    scores = model.predict_proba(eval_frame[FEATURE_COLUMNS])[:, 1]
    output = eval_frame[["sample_id", "label", "label_name", "candidate_answer"] + FEATURE_COLUMNS].copy()
    output["prediction"] = predictions
    output["hallucination_score"] = scores
    return output, classification_metrics(output["label"].tolist(), output["prediction"].tolist(), scores.tolist())


def run_rag_compare(memory_path: str | Path, context_path: str | Path) -> tuple[pd.DataFrame, dict[str, float]]:
    memory_records = {record["sample_id"]: record for record in read_jsonl(memory_path)}
    context_records = read_jsonl(context_path)
    rows = []
    for context_record in context_records:
        memory_record = memory_records[context_record["sample_id"]]
        features = add_rag_features(memory_record, context_record)
        rows.append(
            {
                "sample_id": context_record["sample_id"],
                "label": context_record["label"],
                "label_name": context_record["label_name"],
                "candidate_answer": context_record["candidate_answer"],
                **features,
            }
        )

    frame = pd.DataFrame(rows)
    median_score = float(frame["context_improvement"].median())
    frame["prediction"] = (frame["context_improvement"] < median_score).astype(int)
    frame["hallucination_score"] = -frame["context_improvement"]
    metrics = classification_metrics(
        frame["label"].tolist(),
        frame["prediction"].tolist(),
        frame["hallucination_score"].tolist(),
    )
    metrics["decision_threshold"] = median_score
    return frame, metrics


def run_hybrid_proposed(
    train_text_path: str | Path,
    train_memory_path: str | Path,
    eval_text_path: str | Path,
    eval_memory_path: str | Path,
    eval_context_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    print("[hybrid_proposed] Loading data frames...")
    train_text = records_to_frame(train_text_path)
    eval_text = records_to_frame(eval_text_path)
    train_memory = records_to_frame(train_memory_path)
    eval_memory = records_to_frame(eval_memory_path)
    print(
        f"[hybrid_proposed] Rows text(train/eval)={len(train_text)}/{len(eval_text)}, "
        f"memory(train/eval)={len(train_memory)}/{len(eval_memory)}"
    )

    train = train_text.merge(
        train_memory[["sample_id"] + FEATURE_COLUMNS],
        on="sample_id",
        how="inner",
        suffixes=("_text", ""),
    )
    eval_frame = eval_text.merge(
        eval_memory[["sample_id"] + FEATURE_COLUMNS],
        on="sample_id",
        how="inner",
        suffixes=("_text", ""),
    )
    print(
        f"[hybrid_proposed] Rows after merge train={len(train)} eval={len(eval_frame)} "
        "(inner join on sample_id)"
    )

    feature_cols = list(FEATURE_COLUMNS)
    if eval_context_path is not None:
        print("[hybrid_proposed] Context file provided: adding context_improvement feature for eval.")
        eval_context = records_to_frame(eval_context_path)
        context_merged = eval_memory[["sample_id", "negative_mean_logprob"]].merge(
            eval_context[["sample_id", "negative_mean_logprob"]],
            on="sample_id",
            how="inner",
            suffixes=("_memory", "_context"),
        )
        context_merged["context_improvement"] = (
            context_merged["negative_mean_logprob_memory"] - context_merged["negative_mean_logprob_context"]
        )
        eval_frame = eval_frame.merge(context_merged[["sample_id", "context_improvement"]], on="sample_id", how="left")
        eval_frame["context_improvement"] = eval_frame["context_improvement"].fillna(0.0)
        train["context_improvement"] = 0.0
        feature_cols.append("context_improvement")
        print(f"[hybrid_proposed] Context rows={len(eval_context)} merged_context_rows={len(context_merged)}")

    print("[hybrid_proposed] Building lexical TF-IDF and numeric feature matrices...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
    x_train_text = vectorizer.fit_transform(train["candidate_answer"])
    x_eval_text = vectorizer.transform(eval_frame["candidate_answer"])

    x_train_numeric = csr_matrix(train[feature_cols].astype(float).to_numpy())
    x_eval_numeric = csr_matrix(eval_frame[feature_cols].astype(float).to_numpy())

    x_train = hstack([x_train_text, x_train_numeric], format="csr")
    x_eval = hstack([x_eval_text, x_eval_numeric], format="csr")
    print(
        f"[hybrid_proposed] Matrix shapes train={x_train.shape} eval={x_eval.shape}; "
        f"numeric_features={len(feature_cols)}"
    )
    train_class_counts = train["label"].value_counts().to_dict()
    print(f"[hybrid_proposed] Train class distribution: {train_class_counts}")

    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    print("[hybrid_proposed] Training LogisticRegression (solver=lbfgs, max_iter=2000)...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, train["label"])
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    max_iter_used = int(model.n_iter_[0]) if hasattr(model, "n_iter_") else -1
    converged = max_iter_used < model.max_iter and len(convergence_warnings) == 0
    print(
        f"[hybrid_proposed] Training finished. n_iter={max_iter_used}, "
        f"max_iter={model.max_iter}, converged={converged}"
    )
    if convergence_warnings:
        print(
            "[hybrid_proposed] ConvergenceWarning detected: optimizer hit iteration limit. "
            "Predictions are still generated, but coefficients may not be fully optimized."
        )
    predictions = model.predict(x_eval)
    scores = model.predict_proba(x_eval)[:, 1]

    output_columns = ["sample_id", "label", "label_name", "candidate_answer"] + feature_cols
    output = eval_frame[output_columns].copy()
    output["prediction"] = predictions
    output["hallucination_score"] = scores
    metrics = classification_metrics(output["label"].tolist(), output["prediction"].tolist(), scores.tolist())
    metrics["model_type"] = "hybrid_proposed"
    metrics["uses_context_feature"] = 1.0 if eval_context_path is not None else 0.0
    metrics["train_rows_after_merge"] = float(len(train))
    metrics["eval_rows_after_merge"] = float(len(eval_frame))
    metrics["feature_count_total"] = float(x_train.shape[1])
    metrics["optimizer_n_iter"] = float(max_iter_used)
    metrics["optimizer_converged"] = 1.0 if converged else 0.0
    return output, metrics


def run_hybrid_svm(
    train_text_path: str | Path,
    train_memory_path: str | Path,
    eval_text_path: str | Path,
    eval_memory_path: str | Path,
    eval_context_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    print("[hybrid_svm] Loading data frames...")
    train_text = records_to_frame(train_text_path)
    eval_text = records_to_frame(eval_text_path)
    train_memory = records_to_frame(train_memory_path)
    eval_memory = records_to_frame(eval_memory_path)
    print(
        f"[hybrid_svm] Rows text(train/eval)={len(train_text)}/{len(eval_text)}, "
        f"memory(train/eval)={len(train_memory)}/{len(eval_memory)}"
    )

    train = train_text.merge(
        train_memory[["sample_id"] + FEATURE_COLUMNS],
        on="sample_id",
        how="inner",
        suffixes=("_text", ""),
    )
    eval_frame = eval_text.merge(
        eval_memory[["sample_id"] + FEATURE_COLUMNS],
        on="sample_id",
        how="inner",
        suffixes=("_text", ""),
    )
    print(
        f"[hybrid_svm] Rows after merge train={len(train)} eval={len(eval_frame)} "
        "(inner join on sample_id)"
    )

    feature_cols = list(FEATURE_COLUMNS)
    if eval_context_path is not None:
        print("[hybrid_svm] Context file provided: calculating context_improvement for eval analysis.")
        eval_context = records_to_frame(eval_context_path)
        context_merged = eval_memory[["sample_id", "negative_mean_logprob"]].merge(
            eval_context[["sample_id", "negative_mean_logprob"]],
            on="sample_id",
            how="inner",
            suffixes=("_memory", "_context"),
        )
        context_merged["context_improvement"] = (
            context_merged["negative_mean_logprob_memory"] - context_merged["negative_mean_logprob_context"]
        )
        eval_frame = eval_frame.merge(context_merged[["sample_id", "context_improvement"]], on="sample_id", how="left")
        eval_frame["context_improvement"] = eval_frame["context_improvement"].fillna(0.0)
        
        print(f"[hybrid_svm] Context rows={len(eval_context)} merged_context_rows={len(context_merged)}")
        print("[hybrid_svm] Note: 'context_improvement' is kept in output but excluded from SVM training to prevent the zero-variance data leak.")

    print("[hybrid_svm] Building lexical TF-IDF and scaling numeric feature matrices...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
    x_train_text = vectorizer.fit_transform(train["candidate_answer"])
    x_eval_text = vectorizer.transform(eval_frame["candidate_answer"])

    # --- THE FIX: Scale the numeric features ---
    scaler = StandardScaler()
    
    x_train_numeric_raw = train[feature_cols].astype(float).to_numpy()
    x_eval_numeric_raw = eval_frame[feature_cols].astype(float).to_numpy()
    
    x_train_numeric = csr_matrix(scaler.fit_transform(x_train_numeric_raw))
    x_eval_numeric = csr_matrix(scaler.transform(x_eval_numeric_raw))
    # -------------------------------------------

    x_train = hstack([x_train_text, x_train_numeric], format="csr")
    x_eval = hstack([x_eval_text, x_eval_numeric], format="csr")
    print(
        f"[hybrid_svm] Matrix shapes train={x_train.shape} eval={x_eval.shape}; "
        f"numeric_features={len(feature_cols)}"
    )
    train_class_counts = train["label"].value_counts().to_dict()
    print(f"[hybrid_svm] Train class distribution: {train_class_counts}")

    model = SVC(
        kernel="linear",
        class_weight="balanced",
        probability=True,
        max_iter=5000,
        random_state=42,
    )
    print("[hybrid_svm] Training SVC (kernel=linear, max_iter=2000, probability=True)...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, train["label"])
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    max_iter_used = int(model.n_iter_[0]) if hasattr(model, "n_iter_") else -1
    converged = max_iter_used < model.max_iter and len(convergence_warnings) == 0 if max_iter_used >= 0 else False
    print(
        f"[hybrid_svm] Training finished. n_iter={max_iter_used}, "
        f"max_iter={model.max_iter}, converged={converged}"
    )
    if convergence_warnings:
        print(
            "[hybrid_svm] ConvergenceWarning detected: optimizer hit iteration limit. "
            "Predictions are still generated, but coefficients may not be fully optimized."
        )

    predictions = model.predict(x_eval)
    scores = model.predict_proba(x_eval)[:, 1]

    output_columns = ["sample_id", "label", "label_name", "candidate_answer"] + feature_cols
    # Ensure context_improvement gets appended to the output CSV if it exists
    if eval_context_path is not None and "context_improvement" not in output_columns:
        output_columns.append("context_improvement")

    output = eval_frame[output_columns].copy()
    output["prediction"] = predictions
    output["hallucination_score"] = scores
    metrics = classification_metrics(output["label"].tolist(), output["prediction"].tolist(), scores.tolist())
    metrics["model_type"] = "hybrid_svm"
    metrics["uses_context_feature"] = 1.0 if eval_context_path is not None else 0.0
    metrics["train_rows_after_merge"] = float(len(train))
    metrics["eval_rows_after_merge"] = float(len(eval_frame))
    metrics["feature_count_total"] = float(x_train.shape[1])
    metrics["optimizer_n_iter"] = float(max_iter_used)
    metrics["optimizer_converged"] = 1.0 if converged else 0.0
    return output, metrics