from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import torch
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.exceptions import ConvergenceWarning
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from llm_uncertainty.features import add_rag_features
from llm_uncertainty.io import read_jsonl
from llm_uncertainty.metrics import classification_metrics
from llm_uncertainty.representations import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_NLI_MODEL,
    DEFAULT_SEMANTIC_MODEL,
    NLIProbabilityExtractor,
    SequencePairDataset,
    TransformerBiEncoder,
    resolve_device,
)


FEATURE_COLUMNS = [
    "negative_mean_logprob",
    "mean_logprob",
    "sum_logprob",
    "min_logprob",
    "max_logprob",
    "perplexity",
    "token_count",
    "token_variance",
    "ece",
    "brier_score",
]


def records_to_frame(path: str | Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def _require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _base_output(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "label", "label_name", "candidate_answer"]
    return frame[columns].copy()


def _attach_predictions(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    output = _base_output(frame)
    output["prediction"] = predictions
    output["hallucination_score"] = scores
    metrics = classification_metrics(
        output["label"].tolist(),
        output["prediction"].tolist(),
        output["hallucination_score"].tolist(),
    )
    return output, metrics


def _fit_dense_linear_svm(
    x_train: np.ndarray,
    y_train: pd.Series,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, LinearSVC]:
    model = LinearSVC(class_weight="balanced", max_iter=20000, random_state=42)
    model.fit(x_train, y_train)
    return model.predict(x_eval), model.decision_function(x_eval), model


def _merge_text_and_uncertainty(
    text_path: str | Path,
    memory_path: str | Path,
) -> pd.DataFrame:
    text = records_to_frame(text_path)
    memory = records_to_frame(memory_path)
    _require_columns(text, ["sample_id", "question", "candidate_answer", "label"], str(text_path))
    _require_columns(memory, ["sample_id"] + FEATURE_COLUMNS, str(memory_path))
    merged = text.merge(memory[["sample_id"] + FEATURE_COLUMNS], on="sample_id", how="inner")
    if len(merged) != len(text):
        raise ValueError(
            f"Text/scored row mismatch: {text_path} has {len(text)} rows but only "
            f"{len(merged)} matched {memory_path} by sample_id."
        )
    return merged


def _model_names(frame: pd.DataFrame) -> list[str]:
    if "model_name" not in frame.columns:
        return []
    return sorted(str(value) for value in frame["model_name"].dropna().unique())


def _cached_feature_matrix(
    cache_dir: str | Path | None,
    feature_kind: str,
    source_path: str | Path,
    model_name: str,
    max_length: int,
    compute: Callable[[], np.ndarray],
) -> np.ndarray:
    if cache_dir is None:
        return compute()

    source = Path(source_path)
    stat = source.stat()
    identity = "|".join(
        [
            feature_kind,
            str(source.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            model_name,
            str(max_length),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    safe_kind = feature_kind.replace("/", "_")
    cache_path = Path(cache_dir) / f"{safe_kind}_{source.stem}_{digest}.npy"
    if cache_path.exists():
        print(f"[features] Loading cached matrix {cache_path}")
        return np.load(cache_path)

    matrix = compute()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, matrix)
    temporary_path.replace(cache_path)
    print(f"[features] Saved cached matrix {cache_path}")
    return matrix


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
    _require_columns(train, ["label"] + FEATURE_COLUMNS, str(train_path))
    _require_columns(eval_frame, ["label"] + FEATURE_COLUMNS, str(eval_path))
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=5000, random_state=42),
            ),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train["label"])
    predictions = model.predict(eval_frame[FEATURE_COLUMNS])
    scores = model.predict_proba(eval_frame[FEATURE_COLUMNS])[:, 1]
    output = eval_frame[["sample_id", "label", "label_name", "candidate_answer"] + FEATURE_COLUMNS].copy()
    output["prediction"] = predictions
    output["hallucination_score"] = scores
    metrics = classification_metrics(output["label"].tolist(), output["prediction"].tolist(), scores.tolist())
    metrics["model_type"] = "entropy_base"
    classifier = model.named_steps["classifier"]
    metrics["optimizer_n_iter"] = float(classifier.n_iter_[0])
    metrics["optimizer_converged"] = 1.0 if classifier.n_iter_[0] < classifier.max_iter else 0.0
    model_names = _model_names(train)
    if model_names:
        metrics["scoring_model"] = ",".join(model_names)
    return output, metrics


def run_upgraded_entropy_classifier(
    train_path: str | Path,
    eval_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = records_to_frame(train_path)
    eval_frame = records_to_frame(eval_path)
    train_models = _model_names(train)
    eval_models = _model_names(eval_frame)
    if train_models and eval_models and train_models != eval_models:
        raise ValueError(
            "Upgraded entropy train/eval files were scored by different models: "
            f"{train_models} vs {eval_models}."
        )
    if train_models == ["distilgpt2"]:
        warnings.warn(
            "entropy_upgraded received distilgpt2 features. Use scripts/run_inference.py "
            "with a stronger instruction-tuned --model-name for the intended experiment.",
            stacklevel=2,
        )

    output, metrics = run_entropy_classifier(train_path, eval_path)
    metrics["model_type"] = "entropy_upgraded"
    if train_models:
        metrics["scoring_model"] = ",".join(train_models)
    return output, metrics


def run_rag_compare(memory_path: str | Path, context_path: str | Path) -> tuple[pd.DataFrame, dict[str, float]]:
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

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
    labels = frame["label"].values
    
    # Calculate score based on negative improvement, check if inverted
    score_neg_imp = -frame["context_improvement"].values
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, score_neg_imp)
        if auroc < 0.5:
            frame["hallucination_score"] = frame["context_improvement"]
        else:
            frame["hallucination_score"] = -frame["context_improvement"]
    else:
        frame["hallucination_score"] = -frame["context_improvement"]

    scores = frame["hallucination_score"].values
    
    # Threshold optimization for F1
    thresholds = np.unique(scores)
    
    # Sample thresholds if there are too many to avoid slow computation
    if len(thresholds) > 1000:
        thresholds = np.linspace(scores.min(), scores.max(), 1000)
        
    best_f1 = -1
    best_thresh = 0.0
    
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            
    frame["prediction"] = (scores >= best_thresh).astype(int)

    metrics = classification_metrics(
        frame["label"].tolist(),
        frame["prediction"].tolist(),
        frame["hallucination_score"].tolist(),
    )
    metrics["decision_threshold"] = float(best_thresh)
    return frame, metrics


def _rag_frame(memory_path: str | Path, context_path: str | Path) -> pd.DataFrame:
    memory_records = {record["sample_id"]: record for record in read_jsonl(memory_path)}
    rows = []
    for context_record in read_jsonl(context_path):
        sample_id = context_record["sample_id"]
        if sample_id not in memory_records:
            continue
        features = add_rag_features(memory_records[sample_id], context_record)
        rows.append(
            {
                "sample_id": sample_id,
                "label": context_record["label"],
                "label_name": context_record["label_name"],
                "candidate_answer": context_record["candidate_answer"],
                **features,
            }
        )
    if not rows:
        raise ValueError("No matching sample_id values were found between memory and context files.")
    return pd.DataFrame(rows)


def _best_macro_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    unique_scores = np.unique(scores)
    if len(unique_scores) > 2000:
        unique_scores = np.quantile(scores, np.linspace(0.0, 1.0, 2000))

    boundaries = np.concatenate(
        [
            [np.nextafter(unique_scores.min(), -np.inf)],
            unique_scores,
            [np.nextafter(unique_scores.max(), np.inf)],
        ]
    )
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in boundaries:
        predictions = (scores >= threshold).astype(int)
        score = f1_score(labels, predictions, average="macro")
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def run_rag_compare_fixed(
    train_memory_path: str | Path,
    train_context_path: str | Path,
    eval_memory_path: str | Path,
    eval_context_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = _rag_frame(train_memory_path, train_context_path)
    eval_frame = _rag_frame(eval_memory_path, eval_context_path)

    # Positive context improvement indicates that supplied evidence made the
    # answer more probable. Its negative is therefore the hallucination score.
    train_scores = -train["context_improvement"].to_numpy(dtype=float)
    eval_scores = -eval_frame["context_improvement"].to_numpy(dtype=float)
    threshold, train_macro_f1 = _best_macro_f1_threshold(
        train["label"].to_numpy(dtype=int),
        train_scores,
    )
    predictions = (eval_scores >= threshold).astype(int)

    output = eval_frame.copy()
    output["prediction"] = predictions
    output["hallucination_score"] = eval_scores
    metrics = classification_metrics(
        output["label"].tolist(),
        output["prediction"].tolist(),
        output["hallucination_score"].tolist(),
    )
    metrics["model_type"] = "rag_compare_fixed"
    metrics["decision_threshold"] = threshold
    metrics["train_macro_f1_at_threshold"] = train_macro_f1
    metrics["train_rows"] = float(len(train))
    metrics["eval_rows"] = float(len(eval_frame))
    return output, metrics


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

    scaler = StandardScaler()
    
    x_train_numeric_raw = train[feature_cols].astype(float).to_numpy()
    x_eval_numeric_raw = eval_frame[feature_cols].astype(float).to_numpy()
    
    x_train_numeric = csr_matrix(scaler.fit_transform(x_train_numeric_raw))
    x_eval_numeric = csr_matrix(scaler.transform(x_eval_numeric_raw))

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


def run_lexical_hybrid_svm(
    train_text_path: str | Path,
    train_memory_path: str | Path,
    eval_text_path: str | Path,
    eval_memory_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    output, metrics = run_hybrid_svm(
        train_text_path=train_text_path,
        train_memory_path=train_memory_path,
        eval_text_path=eval_text_path,
        eval_memory_path=eval_memory_path,
        eval_context_path=None,
    )
    metrics["model_type"] = "lexical_hybrid_svm"
    return output, metrics


def _semantic_features(
    encoder: TransformerBiEncoder,
    frame: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    return encoder.interaction_features(
        frame["question"].astype(str).tolist(),
        frame["candidate_answer"].astype(str).tolist(),
        batch_size=batch_size,
    )


def run_semantic_svm(
    train_path: str | Path,
    eval_path: str | Path,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    batch_size: int = 32,
    max_length: int = 256,
    device: str | None = None,
    feature_cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = records_to_frame(train_path)
    eval_frame = records_to_frame(eval_path)
    required = ["question", "candidate_answer", "label"]
    _require_columns(train, required, str(train_path))
    _require_columns(eval_frame, required, str(eval_path))

    encoder = TransformerBiEncoder(model_name=model_name, device=device, max_length=max_length)
    x_train = _cached_feature_matrix(
        feature_cache_dir,
        "semantic",
        train_path,
        model_name,
        max_length,
        lambda: _semantic_features(encoder, train, batch_size),
    )
    x_eval = _cached_feature_matrix(
        feature_cache_dir,
        "semantic",
        eval_path,
        model_name,
        max_length,
        lambda: _semantic_features(encoder, eval_frame, batch_size),
    )
    predictions, scores, model = _fit_dense_linear_svm(x_train, train["label"], x_eval)

    output, metrics = _attach_predictions(eval_frame, predictions, scores)
    metrics.update(
        {
            "model_type": "semantic_svm",
            "semantic_model": model_name,
            "feature_count_total": float(x_train.shape[1]),
            "optimizer_n_iter": float(model.n_iter_),
        }
    )
    return output, metrics


def run_semantic_hybrid_svm(
    train_text_path: str | Path,
    train_memory_path: str | Path,
    eval_text_path: str | Path,
    eval_memory_path: str | Path,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    batch_size: int = 32,
    max_length: int = 256,
    device: str | None = None,
    feature_cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = _merge_text_and_uncertainty(train_text_path, train_memory_path)
    eval_frame = _merge_text_and_uncertainty(eval_text_path, eval_memory_path)

    encoder = TransformerBiEncoder(model_name=model_name, device=device, max_length=max_length)
    x_train_semantic = _cached_feature_matrix(
        feature_cache_dir,
        "semantic",
        train_text_path,
        model_name,
        max_length,
        lambda: _semantic_features(encoder, train, batch_size),
    )
    x_eval_semantic = _cached_feature_matrix(
        feature_cache_dir,
        "semantic",
        eval_text_path,
        model_name,
        max_length,
        lambda: _semantic_features(encoder, eval_frame, batch_size),
    )

    scaler = StandardScaler()
    x_train_uncertainty = scaler.fit_transform(train[FEATURE_COLUMNS].astype(float).to_numpy())
    x_eval_uncertainty = scaler.transform(eval_frame[FEATURE_COLUMNS].astype(float).to_numpy())
    x_train = np.hstack([x_train_semantic, x_train_uncertainty]).astype(np.float32)
    x_eval = np.hstack([x_eval_semantic, x_eval_uncertainty]).astype(np.float32)
    predictions, scores, model = _fit_dense_linear_svm(x_train, train["label"], x_eval)

    output, metrics = _attach_predictions(eval_frame, predictions, scores)
    for column in FEATURE_COLUMNS:
        output[column] = eval_frame[column].to_numpy()
    metrics.update(
        {
            "model_type": "semantic_hybrid_svm",
            "semantic_model": model_name,
            "feature_count_total": float(x_train.shape[1]),
            "optimizer_n_iter": float(model.n_iter_),
        }
    )
    return output, metrics


def _nli_features(
    extractor: NLIProbabilityExtractor,
    frame: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    return extractor.predict(
        frame["knowledge"].astype(str).tolist(),
        frame["candidate_answer"].astype(str).tolist(),
        batch_size=batch_size,
    )


def run_nli_evidence_zero_shot(
    eval_path: str | Path,
    model_name: str = DEFAULT_NLI_MODEL,
    batch_size: int = 16,
    max_length: int = 384,
    device: str | None = None,
    threshold: float = 0.5,
    feature_cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    eval_frame = records_to_frame(eval_path)
    _require_columns(
        eval_frame,
        ["knowledge", "candidate_answer", "label"],
        str(eval_path),
    )
    extractor = NLIProbabilityExtractor(
        model_name=model_name,
        device=device,
        max_length=max_length,
    )
    probabilities = _cached_feature_matrix(
        feature_cache_dir,
        "nli",
        eval_path,
        model_name,
        max_length,
        lambda: _nli_features(extractor, eval_frame, batch_size),
    )
    hallucination_scores = 1.0 - probabilities[:, 0]
    predictions = (hallucination_scores >= threshold).astype(int)

    output, metrics = _attach_predictions(eval_frame, predictions, hallucination_scores)
    for index, column in enumerate(extractor.feature_names):
        output[column] = probabilities[:, index]
    metrics.update(
        {
            "model_type": "nli_evidence_zero_shot",
            "nli_model": model_name,
            "decision_threshold": float(threshold),
        }
    )
    return output, metrics


def run_evidence_aware_hybrid(
    train_text_path: str | Path,
    train_memory_path: str | Path,
    eval_text_path: str | Path,
    eval_memory_path: str | Path,
    semantic_model_name: str = DEFAULT_SEMANTIC_MODEL,
    nli_model_name: str = DEFAULT_NLI_MODEL,
    batch_size: int = 16,
    semantic_max_length: int = 256,
    nli_max_length: int = 384,
    device: str | None = None,
    feature_cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = _merge_text_and_uncertainty(train_text_path, train_memory_path)
    eval_frame = _merge_text_and_uncertainty(eval_text_path, eval_memory_path)
    _require_columns(train, ["knowledge"], str(train_text_path))
    _require_columns(eval_frame, ["knowledge"], str(eval_text_path))

    semantic_encoder = TransformerBiEncoder(
        model_name=semantic_model_name,
        device=device,
        max_length=semantic_max_length,
    )
    x_train_semantic = _cached_feature_matrix(
        feature_cache_dir,
        "semantic",
        train_text_path,
        semantic_model_name,
        semantic_max_length,
        lambda: _semantic_features(semantic_encoder, train, batch_size),
    )
    x_eval_semantic = _cached_feature_matrix(
        feature_cache_dir,
        "semantic",
        eval_text_path,
        semantic_model_name,
        semantic_max_length,
        lambda: _semantic_features(semantic_encoder, eval_frame, batch_size),
    )
    del semantic_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nli_extractor = NLIProbabilityExtractor(
        model_name=nli_model_name,
        device=device,
        max_length=nli_max_length,
    )
    x_train_nli = _cached_feature_matrix(
        feature_cache_dir,
        "nli",
        train_text_path,
        nli_model_name,
        nli_max_length,
        lambda: _nli_features(nli_extractor, train, batch_size),
    )
    x_eval_nli = _cached_feature_matrix(
        feature_cache_dir,
        "nli",
        eval_text_path,
        nli_model_name,
        nli_max_length,
        lambda: _nli_features(nli_extractor, eval_frame, batch_size),
    )

    numeric_columns = FEATURE_COLUMNS
    scaler = StandardScaler()
    x_train_numeric = scaler.fit_transform(train[numeric_columns].astype(float).to_numpy())
    x_eval_numeric = scaler.transform(eval_frame[numeric_columns].astype(float).to_numpy())
    x_train = np.hstack([x_train_semantic, x_train_nli, x_train_numeric]).astype(np.float32)
    x_eval = np.hstack([x_eval_semantic, x_eval_nli, x_eval_numeric]).astype(np.float32)
    predictions, scores, model = _fit_dense_linear_svm(x_train, train["label"], x_eval)

    output, metrics = _attach_predictions(eval_frame, predictions, scores)
    for index, column in enumerate(nli_extractor.feature_names):
        output[column] = x_eval_nli[:, index]
    metrics.update(
        {
            "model_type": "evidence_aware_hybrid",
            "semantic_model": semantic_model_name,
            "nli_model": nli_model_name,
            "feature_count_total": float(x_train.shape[1]),
            "optimizer_n_iter": float(model.n_iter_),
        }
    )
    return output, metrics


def run_cross_encoder_classifier(
    train_path: str | Path,
    eval_path: str | Path,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    batch_size: int = 16,
    max_length: int = 256,
    epochs: int = 2,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    device: str | None = None,
    model_output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = records_to_frame(train_path)
    eval_frame = records_to_frame(eval_path)
    required = ["question", "candidate_answer", "label"]
    _require_columns(train, required, str(train_path))
    _require_columns(eval_frame, required, str(eval_path))
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")

    torch.manual_seed(42)
    np.random.seed(42)
    target_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(target_device)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = SequencePairDataset(
        train["question"].astype(str).tolist(),
        train["candidate_answer"].astype(str).tolist(),
        train["label"].astype(int).tolist(),
        tokenizer,
        max_length,
    )
    eval_dataset = SequencePairDataset(
        eval_frame["question"].astype(str).tolist(),
        eval_frame["candidate_answer"].astype(str).tolist(),
        eval_frame["label"].astype(int).tolist(),
        tokenizer,
        max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    optimizer.zero_grad()
    optimizer_steps = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(target_device) for key, value in batch.items()}
            loss = model(**batch).loss / gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * gradient_accumulation_steps
            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1
        print(
            f"[cross_encoder] epoch={epoch + 1}/{epochs} "
            f"mean_loss={running_loss / max(len(train_loader), 1):.6f}"
        )

    if model_output_dir is not None:
        output_dir = Path(model_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    model.eval()
    score_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in eval_loader:
            batch = {key: value.to(target_device) for key, value in batch.items()}
            logits = model(**batch).logits
            score_batches.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    scores = np.concatenate(score_batches) if score_batches else np.empty(0, dtype=np.float32)
    predictions = (scores >= 0.5).astype(int)

    output, metrics = _attach_predictions(eval_frame, predictions, scores)
    metrics.update(
        {
            "model_type": "cross_encoder_classifier",
            "cross_encoder_model": model_name,
            "epochs": float(epochs),
            "optimizer_steps": float(optimizer_steps),
        }
    )
    return output, metrics
