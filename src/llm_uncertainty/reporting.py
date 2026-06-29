from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from llm_uncertainty.io import ensure_parent


def _run_name(results_dir: Path, file_path: Path) -> str:
    relative = file_path.relative_to(results_dir).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] in {"metrics", "predictions"}:
        parts = parts[:-1]
    return "/".join(parts)


def build_report_table(results_dir: Path, output_path: Path) -> None:
    rows = []
    metrics_paths = sorted(results_dir.rglob("metrics.json"))
    for path in metrics_paths:
        rows.append({"run": _run_name(results_dir, path), **json.loads(path.read_text(encoding="utf-8"))})

    if not rows:
        print(f"No metrics JSON files found under {results_dir}.")
        return

    ensure_parent(output_path)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved report table to {output_path}")


def build_domain_comparison_table(
    source_results_dir: Path,
    target_results_dir: Path,
    output_path: Path,
    *,
    source_label: str = "in_domain",
    target_label: str = "out_of_domain",
) -> None:
    def load_metrics(results_dir: Path) -> pd.DataFrame:
        rows = []
        for path in sorted(results_dir.rglob("metrics.json")):
            run_name = _run_name(results_dir, path)
            parts = run_name.split("/")
            if len(parts) < 2:
                continue
            baseline = parts[0]
            eval_split = parts[1]
            rows.append(
                {
                    "baseline": baseline,
                    "eval_split": eval_split,
                    **json.loads(path.read_text(encoding="utf-8")),
                }
            )
        if not rows:
            raise ValueError(f"No metrics JSON files found under {results_dir}.")
        return pd.DataFrame(rows)

    source = load_metrics(source_results_dir).rename(
        columns={
            "accuracy": f"{source_label}_accuracy",
            "macro_f1": f"{source_label}_macro_f1",
            "auroc": f"{source_label}_auroc",
        }
    )
    target = load_metrics(target_results_dir).rename(
        columns={
            "accuracy": f"{target_label}_accuracy",
            "macro_f1": f"{target_label}_macro_f1",
            "auroc": f"{target_label}_auroc",
        }
    )

    merged = source.merge(target, on=["baseline", "eval_split"], how="inner")
    if merged.empty:
        raise ValueError(
            "No overlapping baseline/eval_split pairs were found between the two result directories."
        )

    for metric in ("accuracy", "macro_f1", "auroc"):
        left = f"{source_label}_{metric}"
        right = f"{target_label}_{metric}"
        if left in merged.columns and right in merged.columns:
            merged[f"delta_{metric}"] = merged[right] - merged[left]

    preferred_columns = [
        "baseline",
        "eval_split",
        f"{source_label}_accuracy",
        f"{target_label}_accuracy",
        "delta_accuracy",
        f"{source_label}_macro_f1",
        f"{target_label}_macro_f1",
        "delta_macro_f1",
        f"{source_label}_auroc",
        f"{target_label}_auroc",
        "delta_auroc",
    ]
    ordered = [column for column in preferred_columns if column in merged.columns]
    ordered += [column for column in merged.columns if column not in ordered]
    merged = merged.sort_values(["baseline", "eval_split"])[ordered]

    ensure_parent(output_path)
    merged.to_csv(output_path, index=False)
    print(f"Saved domain comparison table to {output_path}")


def _plot_confusion(y_true: list[int], y_pred: list[int], output_path: Path, title: str) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(4.5, 4.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_roc(y_true: list[int], scores: list[float], output_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(fpr, tpr, color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_precision_recall(
    y_true: list[int],
    scores: list[float],
    output_path: Path,
    title: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    average_precision = average_precision_score(y_true, scores)
    prevalence = float(sum(y_true) / len(y_true))
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(recall, precision, color="darkgreen", lw=2, label=f"AP={average_precision:.3f}")
    plt.axhline(prevalence, color="gray", lw=1, linestyle="--", label="Prevalence")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_calibration(y_true: list[int], scores: list[float], output_path: Path, title: str) -> None:
    min_score = min(scores)
    max_score = max(scores)
    if min_score < 0.0 or max_score > 1.0:
        plt.figure(figsize=(4.5, 4.0))
        plt.text(
            0.5,
            0.55,
            "Calibration unavailable",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
        )
        plt.text(
            0.5,
            0.42,
            "Model emits an unbounded decision score,\nnot a predicted probability.",
            ha="center",
            va="center",
            fontsize=10,
        )
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Marked calibration unavailable for {title}: scores are outside [0, 1].")
        return

    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10, strategy="uniform")
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_plots(results_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions = sorted(results_dir.rglob("predictions.csv"))
    if not predictions:
        print(f"No prediction CSV files found under {results_dir}.")
        return

    for path in predictions:
        frame = pd.read_csv(path)
        if "label" not in frame.columns:
            print(f"Skipping {path.name}: missing 'label' column.")
            continue

        run_name = _run_name(results_dir, path)
        safe_name = run_name.replace("/", "__")
        y_true = frame["label"].tolist()
        if "prediction" in frame.columns:
            y_pred = frame["prediction"].tolist()
            output_path = figures_dir / f"{safe_name}_confusion.png"
            _plot_confusion(y_true, y_pred, output_path, f"Confusion: {run_name}")
        else:
            print(f"Skipping confusion plot for {run_name}: missing 'prediction' column.")

        if "hallucination_score" not in frame.columns:
            print(f"Skipping ROC/calibration for {run_name}: missing 'hallucination_score' column.")
            continue

        scores = frame["hallucination_score"].tolist()
        if len(set(y_true)) != 2:
            print(f"Skipping ROC/calibration for {run_name}: non-binary labels.")
            continue

        roc_path = figures_dir / f"{safe_name}_roc.png"
        _plot_roc(y_true, scores, roc_path, f"ROC: {run_name}")

        pr_path = figures_dir / f"{safe_name}_precision_recall.png"
        _plot_precision_recall(
            y_true,
            scores,
            pr_path,
            f"Precision–Recall: {run_name}",
        )

        calibration_path = figures_dir / f"{safe_name}_calibration.png"
        _plot_calibration(y_true, scores, calibration_path, f"Calibration: {run_name}")


def _display_name(run_name: str) -> str:
    baseline = run_name.split("/", maxsplit=1)[0]
    words = baseline.split("_")
    acronyms = {"nli": "NLI", "rag": "RAG", "svm": "SVM", "lr": "LR"}
    return " ".join(acronyms.get(word, word.title()) for word in words)


def build_comparison_assets(
    results_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    eval_split: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for path in sorted(results_dir.glob(f"*/{eval_split}/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        baseline = path.parent.parent.name
        metric_rows.append({"baseline": baseline, **metrics})

    if not metric_rows:
        print(f"No baseline metrics found under {results_dir} for split={eval_split}.")
        return

    metrics_frame = pd.DataFrame(metric_rows).sort_values("baseline")
    table_path = tables_dir / f"baseline_comparison_{eval_split}.csv"
    metrics_frame.to_csv(table_path, index=False)
    print(f"Saved baseline comparison table to {table_path}")

    metric_columns = [
        column
        for column in ["accuracy", "macro_f1", "auroc"]
        if column in metrics_frame.columns
    ]
    if metric_columns:
        plot_frame = metrics_frame[["baseline"] + metric_columns].melt(
            id_vars="baseline",
            var_name="metric",
            value_name="score",
        )
        plot_frame["baseline"] = plot_frame["baseline"].map(_display_name)
        plot_frame["metric"] = plot_frame["metric"].map(
            {"accuracy": "Accuracy", "macro_f1": "Macro F1", "auroc": "AUROC"}
        )
        plt.figure(figsize=(max(10.0, len(metrics_frame) * 1.25), 6.0))
        sns.barplot(data=plot_frame, x="baseline", y="score", hue="metric")
        minimum_score = float(plot_frame["score"].min())
        plt.ylim(max(0.0, minimum_score - 0.05), 1.0)
        plt.xlabel("Baseline")
        plt.ylabel("Score")
        plt.title(f"Baseline Metric Comparison ({eval_split})")
        plt.xticks(rotation=35, ha="right")
        plt.legend(title="Metric", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout()
        metric_plot_path = figures_dir / f"baseline_comparison_metrics_{eval_split}.png"
        plt.savefig(metric_plot_path, dpi=200)
        plt.close()
        print(f"Saved baseline metric comparison to {metric_plot_path}")

    plt.figure(figsize=(8.0, 6.5))
    plotted = 0
    for path in sorted(results_dir.glob(f"*/{eval_split}/predictions.csv")):
        frame = pd.read_csv(path)
        if not {"label", "hallucination_score"}.issubset(frame.columns):
            continue
        y_true = frame["label"].tolist()
        if len(set(y_true)) != 2:
            continue
        scores = frame["hallucination_score"].tolist()
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        baseline = path.parent.parent.name
        plt.plot(fpr, tpr, lw=1.8, label=f"{_display_name(baseline)} ({roc_auc:.3f})")
        plotted += 1

    if plotted:
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Baseline ROC Comparison ({eval_split})")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        roc_plot_path = figures_dir / f"baseline_comparison_roc_{eval_split}.png"
        plt.savefig(roc_plot_path, dpi=200)
        plt.close()
        print(f"Saved baseline ROC comparison to {roc_plot_path}")
    else:
        plt.close()

    plt.figure(figsize=(8.0, 6.5))
    plotted = 0
    prevalence = None
    for path in sorted(results_dir.glob(f"*/{eval_split}/predictions.csv")):
        frame = pd.read_csv(path)
        if not {"label", "hallucination_score"}.issubset(frame.columns):
            continue
        y_true = frame["label"].tolist()
        if len(set(y_true)) != 2:
            continue
        scores = frame["hallucination_score"].tolist()
        precision, recall, _ = precision_recall_curve(y_true, scores)
        average_precision = average_precision_score(y_true, scores)
        baseline = path.parent.parent.name
        plt.plot(
            recall,
            precision,
            lw=1.8,
            label=f"{_display_name(baseline)} ({average_precision:.3f})",
        )
        prevalence = float(sum(y_true) / len(y_true))
        plotted += 1

    if plotted:
        plt.axhline(
            prevalence,
            color="gray",
            lw=1,
            linestyle="--",
            label=f"Prevalence ({prevalence:.2f})",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Baseline Precision–Recall Comparison ({eval_split})")
        plt.legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        pr_plot_path = figures_dir / f"baseline_comparison_precision_recall_{eval_split}.png"
        plt.savefig(pr_plot_path, dpi=200)
        plt.close()
        print(f"Saved baseline precision-recall comparison to {pr_plot_path}")
    else:
        plt.close()
