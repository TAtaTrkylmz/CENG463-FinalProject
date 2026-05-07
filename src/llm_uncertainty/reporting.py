from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

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


def _plot_calibration(y_true: list[int], scores: list[float], output_path: Path, title: str) -> None:
    min_score = min(scores)
    max_score = max(scores)
    if min_score < 0.0 or max_score > 1.0:
        if max_score == min_score:
            print(f"Skipping calibration for {title}: constant scores outside [0, 1].")
            return
        scores = [(value - min_score) / (max_score - min_score) for value in scores]

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

        calibration_path = figures_dir / f"{safe_name}_calibration.png"
        _plot_calibration(y_true, scores, calibration_path, f"Calibration: {run_name}")
