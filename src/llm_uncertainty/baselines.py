from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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
    median_score = float(frame["absolute_context_delta"].median())
    frame["prediction"] = (frame["absolute_context_delta"] >= median_score).astype(int)
    frame["hallucination_score"] = frame["absolute_context_delta"]
    metrics = classification_metrics(
        frame["label"].tolist(),
        frame["prediction"].tolist(),
        frame["hallucination_score"].tolist(),
    )
    metrics["decision_threshold"] = median_score
    return frame, metrics

