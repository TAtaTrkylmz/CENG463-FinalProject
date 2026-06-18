from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


BASELINE_ORDER = [
    "lexical_svm",
    "lexical_hybrid_svm",
    "entropy_base",
    "entropy_upgraded",
    "semantic_svm",
    "semantic_hybrid_svm",
    "rag_compare_fixed",
    "nli_evidence",
    "evidence_aware_hybrid",
    "cross_encoder",
]


def display_name(baseline: str) -> str:
    acronyms = {"nli": "NLI", "rag": "RAG", "svm": "SVM", "lm": "LM"}
    return " ".join(acronyms.get(word, word.title()) for word in baseline.split("_"))


def _ordered_baselines(values: list[str]) -> list[str]:
    known = [baseline for baseline in BASELINE_ORDER if baseline in values]
    return known + sorted(set(values) - set(known))


def _load_metadata(data_path: Path | None) -> pd.DataFrame | None:
    if data_path is None or not data_path.exists():
        return None
    metadata = pd.read_json(data_path, lines=True)
    if "sample_id" not in metadata.columns:
        raise ValueError(f"{data_path} does not contain sample_id.")
    keep = [
        column
        for column in [
            "sample_id",
            "original_sample_id",
            "question",
            "knowledge",
            "candidate_answer",
            "label",
            "label_name",
        ]
        if column in metadata.columns
    ]
    return metadata[keep].drop_duplicates("sample_id")


def _question_type(question: object) -> str:
    text = str(question).strip().lower()
    for prefix in ["who", "what", "when", "where", "why", "how", "which"]:
        if text == prefix or text.startswith(prefix + " "):
            return prefix.title()
    return "Other"


def _infer_threshold(
    scores: np.ndarray,
    predictions: np.ndarray,
    configured_threshold: float | None,
) -> tuple[float, float]:
    candidates: list[float] = []
    if configured_threshold is not None and np.isfinite(configured_threshold):
        candidates.append(float(configured_threshold))
    candidates.extend([0.0, 0.5])

    unique_scores = np.unique(scores)
    if len(unique_scores) > 1:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        if len(midpoints) > 5000:
            indices = np.linspace(0, len(midpoints) - 1, 5000).astype(int)
            midpoints = midpoints[indices]
        candidates.extend(midpoints.tolist())

    best_threshold = candidates[0]
    best_agreement = -1.0
    for threshold in candidates:
        agreement = float(np.mean((scores >= threshold).astype(int) == predictions))
        if agreement > best_agreement:
            best_agreement = agreement
            best_threshold = float(threshold)
    return best_threshold, best_agreement


def _load_run(
    predictions_path: Path,
    metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    baseline = predictions_path.parent.parent.name
    frame = pd.read_csv(predictions_path)
    required = {"sample_id", "label", "prediction", "hallucination_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{predictions_path} is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{predictions_path} contains duplicate sample_id values.")

    metrics_path = predictions_path.with_name("metrics.json")
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )
    configured_threshold = metrics.get("decision_threshold")
    scores = frame["hallucination_score"].to_numpy(dtype=float)
    predictions = frame["prediction"].to_numpy(dtype=int)
    threshold, threshold_agreement = _infer_threshold(
        scores,
        predictions,
        float(configured_threshold) if configured_threshold is not None else None,
    )

    work = frame.copy()
    if metadata is not None:
        extra_columns = [
            column
            for column in metadata.columns
            if column not in work.columns or column in {"question", "knowledge", "original_sample_id"}
        ]
        if extra_columns:
            work = work.merge(
                metadata[["sample_id"] + [c for c in extra_columns if c != "sample_id"]],
                on="sample_id",
                how="left",
                validate="one_to_one",
            )

    work["baseline"] = baseline
    work["is_error"] = work["label"].astype(int) != work["prediction"].astype(int)
    work["error_type"] = np.select(
        [
            (work["label"] == 0) & (work["prediction"] == 1),
            (work["label"] == 1) & (work["prediction"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    work["answer_len_chars"] = work["candidate_answer"].fillna("").astype(str).str.len()
    work["answer_length_bin"] = pd.cut(
        work["answer_len_chars"],
        bins=[-1, 40, 120, np.inf],
        labels=["Short (0–40)", "Medium (41–120)", "Long (121+)"],
    )
    if "question" in work.columns:
        work["question_type"] = work["question"].map(_question_type)

    work["decision_threshold"] = threshold
    work["decision_margin"] = np.abs(work["hallucination_score"] - threshold)
    work["relative_confidence"] = work["decision_margin"].rank(
        method="average", pct=True
    )
    work["confidence_quintile"] = pd.cut(
        work["relative_confidence"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )

    tn, fp, fn, tp = confusion_matrix(
        work["label"], work["prediction"], labels=[0, 1]
    ).ravel()
    summary: dict[str, float | str] = {
        "baseline": baseline,
        "display_name": display_name(baseline),
        "num_samples": float(len(work)),
        "num_errors": float(work["is_error"].sum()),
        "error_rate": float(work["is_error"].mean()),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "inferred_decision_threshold": threshold,
        "threshold_prediction_agreement": threshold_agreement,
        "high_confidence_errors": float(
            ((work["relative_confidence"] > 0.8) & work["is_error"]).sum()
        ),
    }
    return work, summary


def _save_confusion_grid(
    runs: dict[str, pd.DataFrame],
    output_path: Path,
    baselines: list[str],
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4))
    for axis, baseline in zip(axes.flat, baselines):
        frame = runs[baseline]
        matrix = confusion_matrix(frame["label"], frame["prediction"], labels=[0, 1])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        annotations = np.empty_like(matrix, dtype=object)
        for row in range(2):
            for column in range(2):
                annotations[row, column] = (
                    f"{matrix[row, column]:,}\n({normalized[row, column]:.1%})"
                )
        sns.heatmap(
            normalized,
            annot=annotations,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=["Factual", "Hallucinated"],
            yticklabels=["Factual", "Hallucinated"],
            ax=axis,
        )
        axis.set_title(display_name(baseline), fontsize=10)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    for axis in axes.flat[len(baselines) :]:
        axis.axis("off")
    fig.suptitle("Validation Confusion Matrices Across All Baselines", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_fp_fn_plot(
    summary: pd.DataFrame,
    output_path: Path,
    baselines: list[str],
) -> None:
    ordered = summary.set_index("baseline").loc[baselines].reset_index()
    plot = ordered.melt(
        id_vars=["baseline", "display_name"],
        value_vars=["false_positive", "false_negative"],
        var_name="error_type",
        value_name="count",
    )
    plot["error_type"] = plot["error_type"].map(
        {"false_positive": "False positive", "false_negative": "False negative"}
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot, x="display_name", y="count", hue="error_type")
    plt.xlabel("")
    plt.ylabel("Number of validation errors")
    plt.title("False Positives and False Negatives by Baseline")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_group_error_plot(
    table: pd.DataFrame,
    group_column: str,
    output_path: Path,
    title: str,
    baselines: list[str],
) -> None:
    pivot = table.pivot(
        index="baseline", columns=group_column, values="error_rate"
    ).reindex(baselines)
    pivot.index = [display_name(value) for value in pivot.index]
    plt.figure(figsize=(10, 6.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        vmin=0,
        cbar_kws={"label": "Error rate"},
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_overlap_plot(
    runs: dict[str, pd.DataFrame],
    output_path: Path,
    baselines: list[str],
) -> pd.DataFrame:
    error_sets = {
        baseline: set(runs[baseline].loc[runs[baseline]["is_error"], "sample_id"])
        for baseline in baselines
    }
    matrix = np.zeros((len(baselines), len(baselines)), dtype=float)
    for row, left in enumerate(baselines):
        for column, right in enumerate(baselines):
            union = error_sets[left] | error_sets[right]
            matrix[row, column] = (
                len(error_sets[left] & error_sets[right]) / len(union) if union else 1.0
            )
    overlap = pd.DataFrame(matrix, index=baselines, columns=baselines)
    labels = [display_name(value) for value in baselines]
    plt.figure(figsize=(10, 8.5))
    sns.heatmap(
        overlap,
        cmap="mako",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Jaccard overlap"},
    )
    plt.title("Overlap Between Baseline Error Sets")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return overlap


def _save_confidence_plot(
    confidence: pd.DataFrame,
    output_path: Path,
    baselines: list[str],
) -> None:
    order = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
    pivot = confidence.pivot(
        index="baseline", columns="confidence_quintile", values="error_rate"
    ).reindex(index=baselines, columns=order)
    pivot.index = [display_name(value) for value in pivot.index]
    plt.figure(figsize=(10, 6.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="rocket_r",
        vmin=0,
        cbar_kws={"label": "Error rate"},
    )
    plt.xlabel("Relative decision-margin confidence within each baseline")
    plt.ylabel("")
    plt.title("Error Rate by Relative Confidence Quintile")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_consensus_plot(consensus: pd.DataFrame, output_path: Path) -> None:
    counts = (
        consensus["num_baselines_wrong"]
        .value_counts()
        .reindex(range(int(consensus["num_baselines_wrong"].max()) + 1), fill_value=0)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.index, y=counts.values, color="#4c72b0")
    plt.xlabel("Number of baselines wrong on the same sample")
    plt.ylabel("Validation samples")
    plt.title("Cross-Baseline Error Consensus")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _write_findings(
    summary: pd.DataFrame,
    consensus: pd.DataFrame,
    output_path: Path,
) -> None:
    best = summary.sort_values("error_rate").iloc[0]
    worst = summary.sort_values("error_rate", ascending=False).iloc[0]
    fp_biased = summary.assign(
        asymmetry=summary["false_positive"] - summary["false_negative"]
    ).sort_values("asymmetry", ascending=False).iloc[0]
    fn_biased = summary.assign(
        asymmetry=summary["false_negative"] - summary["false_positive"]
    ).sort_values("asymmetry", ascending=False).iloc[0]
    most_shared = consensus.sort_values(
        ["num_baselines_wrong", "sample_id"], ascending=[False, True]
    ).iloc[0]
    all_wrong = int((consensus["num_baselines_wrong"] == len(summary)).sum())
    at_least_eight_wrong = int((consensus["num_baselines_wrong"] >= 8).sum())
    lines = [
        "# Error-analysis highlights",
        "",
        "These results use the grouped validation split (2,002 rows), not the held-out test set.",
        "",
        f"- Lowest error rate: **{best['display_name']}** "
        f"({int(best['num_errors'])} errors, {best['error_rate']:.2%}).",
        f"- Highest error rate: **{worst['display_name']}** "
        f"({int(worst['num_errors'])} errors, {worst['error_rate']:.2%}).",
        f"- Hardest shared sample: `{most_shared['sample_id']}` was missed by "
        f"{int(most_shared['num_baselines_wrong'])} of {len(summary)} baselines.",
        f"- {all_wrong} samples were missed by every baseline; {at_least_eight_wrong} "
        "were missed by at least eight baselines.",
        f"- Strongest false-positive bias: **{fp_biased['display_name']}** "
        f"({int(fp_biased['false_positive'])} FP vs. "
        f"{int(fp_biased['false_negative'])} FN).",
        f"- Strongest false-negative bias: **{fn_biased['display_name']}** "
        f"({int(fn_biased['false_negative'])} FN vs. "
        f"{int(fn_biased['false_positive'])} FP).",
        "- The shared hardest examples are mostly short, plausible entities or close "
        "semantic alternatives copied from the evidence. This suggests a dataset-level "
        "hard-negative/annotation ambiguity that should be discussed in the paper.",
        "- “Relative confidence” is the percentile rank of distance from each model's "
        "decision boundary. It permits cross-model comparison without pretending that "
        "unbounded SVM/RAG scores are calibrated probabilities.",
        "",
        "See `hardest_shared_errors.csv` and the per-baseline top-k tables for qualitative review.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_error_analysis(
    results_dir: Path,
    output_dir: Path,
    figures_dir: Path,
    eval_split: str = "val",
    data_path: Path | None = None,
    top_k: int = 25,
) -> dict[str, Path]:
    prediction_paths = sorted(results_dir.glob(f"*/{eval_split}/predictions.csv"))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No prediction files found under {results_dir} for split={eval_split}."
        )

    metadata = _load_metadata(data_path)
    runs: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, float | str]] = []
    for path in prediction_paths:
        frame, summary = _load_run(path, metadata)
        baseline = str(summary["baseline"])
        runs[baseline] = frame
        summaries.append(summary)

    baselines = _ordered_baselines(list(runs))
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summaries)
    summary["_order"] = summary["baseline"].map(
        {baseline: index for index, baseline in enumerate(baselines)}
    )
    summary = summary.sort_values("_order").drop(columns="_order")
    summary_path = output_dir / "summary_by_baseline.csv"
    summary.to_csv(summary_path, index=False)

    combined = pd.concat([runs[baseline] for baseline in baselines], ignore_index=True)
    length = (
        combined.groupby(["baseline", "answer_length_bin"], observed=True)
        .agg(count=("sample_id", "size"), errors=("is_error", "sum"), error_rate=("is_error", "mean"))
        .reset_index()
    )
    length.to_csv(output_dir / "error_by_answer_length.csv", index=False)

    question_type = pd.DataFrame()
    if "question_type" in combined.columns:
        question_type = (
            combined.groupby(["baseline", "question_type"], observed=True)
            .agg(count=("sample_id", "size"), errors=("is_error", "sum"), error_rate=("is_error", "mean"))
            .reset_index()
        )
        question_type.to_csv(output_dir / "error_by_question_type.csv", index=False)

    confidence = (
        combined.groupby(["baseline", "confidence_quintile"], observed=True)
        .agg(count=("sample_id", "size"), errors=("is_error", "sum"), error_rate=("is_error", "mean"))
        .reset_index()
    )
    confidence.to_csv(output_dir / "error_by_relative_confidence.csv", index=False)

    detail_columns = [
        column
        for column in [
            "baseline",
            "sample_id",
            "original_sample_id",
            "question",
            "knowledge",
            "candidate_answer",
            "label",
            "prediction",
            "hallucination_score",
            "decision_threshold",
            "decision_margin",
            "relative_confidence",
            "error_type",
            "answer_len_chars",
            "answer_length_bin",
            "question_type",
        ]
        if column in combined.columns
    ]
    false_positives = combined[combined["error_type"] == "false_positive"].sort_values(
        ["baseline", "relative_confidence"], ascending=[True, False]
    )
    false_negatives = combined[combined["error_type"] == "false_negative"].sort_values(
        ["baseline", "relative_confidence"], ascending=[True, False]
    )
    overconfident = combined[combined["is_error"]].sort_values(
        ["baseline", "relative_confidence"], ascending=[True, False]
    )
    false_positives.groupby("baseline", sort=False).head(top_k)[detail_columns].to_csv(
        output_dir / "false_positives_topk.csv", index=False
    )
    false_negatives.groupby("baseline", sort=False).head(top_k)[detail_columns].to_csv(
        output_dir / "false_negatives_topk.csv", index=False
    )
    overconfident.groupby("baseline", sort=False).head(top_k)[detail_columns].to_csv(
        output_dir / "overconfident_errors_topk.csv", index=False
    )

    reference = runs[baselines[0]].copy()
    reference_columns = [
        column
        for column in [
            "sample_id",
            "original_sample_id",
            "question",
            "knowledge",
            "candidate_answer",
            "label",
            "label_name",
        ]
        if column in reference.columns
    ]
    consensus = reference[reference_columns].copy()
    error_matrix = pd.DataFrame({"sample_id": consensus["sample_id"]})
    for baseline in baselines:
        errors = runs[baseline].set_index("sample_id")["is_error"]
        error_matrix[baseline] = error_matrix["sample_id"].map(errors).astype(bool)
    consensus["num_baselines_wrong"] = error_matrix[baselines].sum(axis=1)
    consensus["wrong_baselines"] = error_matrix[baselines].apply(
        lambda row: ", ".join([baseline for baseline in baselines if row[baseline]]),
        axis=1,
    )
    consensus = consensus.sort_values(
        ["num_baselines_wrong", "sample_id"], ascending=[False, True]
    )
    consensus.to_csv(output_dir / "sample_error_consensus.csv", index=False)
    consensus[consensus["num_baselines_wrong"] > 0].head(max(top_k, 50)).to_csv(
        output_dir / "hardest_shared_errors.csv", index=False
    )

    _save_confusion_grid(
        runs, figures_dir / f"confusion_grid_{eval_split}.png", baselines
    )
    _save_fp_fn_plot(
        summary, figures_dir / f"false_positive_negative_{eval_split}.png", baselines
    )
    _save_group_error_plot(
        length,
        "answer_length_bin",
        figures_dir / f"error_by_answer_length_{eval_split}.png",
        "Error Rate by Candidate-Answer Length",
        baselines,
    )
    if not question_type.empty:
        _save_group_error_plot(
            question_type,
            "question_type",
            figures_dir / f"error_by_question_type_{eval_split}.png",
            "Error Rate by Question Type",
            baselines,
        )
    overlap = _save_overlap_plot(
        runs, figures_dir / f"error_overlap_jaccard_{eval_split}.png", baselines
    )
    overlap.to_csv(output_dir / "error_overlap_jaccard.csv")
    _save_confidence_plot(
        confidence,
        figures_dir / f"error_by_relative_confidence_{eval_split}.png",
        baselines,
    )
    _save_consensus_plot(
        consensus, figures_dir / f"error_consensus_{eval_split}.png"
    )
    _write_findings(summary, consensus, output_dir / "key_findings.md")

    return {
        "summary": summary_path,
        "findings": output_dir / "key_findings.md",
        "figures": figures_dir,
    }
