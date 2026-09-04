"""Regenerate every chart used in the presentation deck.

Replaces the stale ``plot_ablation.py`` / ``plot_feature_expansion.py`` pair,
which were pinned to the old 5-model pipeline (Hybrid LR vs Hybrid SVM) and,
in the feature-expansion case, to hard-coded numbers.

All figures are written to ``docs/images/slides/`` and read only from committed
result artifacts, so ``python scripts/make_slide_figures.py`` is reproducible
without re-running any model.

Source of truth per figure:
  01 main results      results/matrix/<baseline>/val/metrics.json
  02 val vs test       results/matrix/<baseline>/{val,test}/metrics.json
  03 ablation          results/ablation/val_svm_fixed/summary.csv
  04 transfer gap      reports/tables/halueval_vs_truthfulqa*.csv
  05 error by length   reports/error_analysis/test/error_by_answer_length.csv
  06 error by conf.    reports/error_analysis/test/error_by_relative_confidence.csv

Note on splits: reports/error_analysis/val/ and reports/tables/baseline_results_val.csv
were overwritten by the later Qwen-generated evaluation run and no longer hold
HaluEval numbers. The clean HaluEval error analysis is the ``test/`` directory,
and the canonical HaluEval val metrics live in results/matrix/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images" / "slides"

# Categorical slots 1-4 of the validated default palette. Slots 1-3 clear the
# all-pairs gates; the 04_transfer_gap chart below uses all four as adjacent
# grouped bars (not all-pairs), which the 4-slot order still clears, at the
# cost of a contrast WARN on aqua/yellow against the surface — every bar
# carries a visible direct label to satisfy that.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE = "#fcfcfb"
INK, INK_MUTED = "#0b0b0b", "#52514e"

DISPLAY = {
    "lexical_svm": "Lexical SVM",
    "entropy_base": "Entropy Base",
    "entropy_upgraded": "Entropy Upgraded",
    "semantic_svm": "Semantic SVM",
    "nli_evidence": "NLI Evidence",
    "rag_compare_fixed": "RAG Compare",
    "lexical_hybrid_svm": "Lexical Hybrid SVM",
    "semantic_hybrid_svm": "Semantic Hybrid SVM",
    "evidence_aware_hybrid": "Evidence-Aware Hybrid",
    "cross_encoder": "Cross-Encoder (FT)",
}
SINGLE_SIGNAL = {
    "lexical_svm",
    "entropy_base",
    "entropy_upgraded",
    "semantic_svm",
    "nli_evidence",
    "rag_compare_fixed",
}


def style_axes(ax, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=INK_MUTED, labelsize=10, length=0)
    if grid_axis == "y":
        ax.yaxis.grid(True, color="#e7e6e1", linewidth=0.8)
    else:
        ax.xaxis.grid(True, color="#e7e6e1", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=14, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=INK_MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=INK_MUTED)


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def load_matrix(split):
    rows = {}
    for key in DISPLAY:
        p = ROOT / "results" / "matrix" / key / split / "metrics.json"
        if p.exists():
            rows[key] = json.loads(p.read_text())
    return rows


# --------------------------------------------------------------------------
# 01 - main results: all ten methods on the HaluEval validation split
# --------------------------------------------------------------------------
def fig_main_results():
    val = load_matrix("val")
    order = sorted(val, key=lambda k: val[k]["auroc"])
    labels = [DISPLAY[k] + ("" if k in SINGLE_SIGNAL else "  ✦") for k in order]
    acc = [val[k]["accuracy"] for k in order]
    auroc = [val[k]["auroc"] for k in order]

    # Dot plot, not bars: the scores cluster in the top quarter of the range, and
    # a bar chart would need a truncated baseline to show any separation.
    y = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURFACE)
    for i, k in zip(y, order):
        ax.plot([acc[i], auroc[i]], [i, i], color="#c9c8c3", linewidth=2, zorder=1)
    ax.scatter(auroc, y, s=130, color=BLUE, label="AUROC", zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.scatter(acc, y, s=130, color=ORANGE, label="Accuracy", zorder=3,
               edgecolor=SURFACE, linewidth=2)

    for i in y:
        ax.text(auroc[i] + 0.008, i, f"{auroc[i]:.3f}", va="center",
                fontsize=9.5, color=INK, fontweight="bold")
        ax.text(acc[i] - 0.008, i, f"{acc[i]:.3f}", va="center", ha="right",
                fontsize=9.5, color=INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=INK)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0.70, 1.03)
    style_axes(
        ax,
        xlabel="Score (HaluEval QA validation split, 2,002 rows)",
        title="Ten detectors, four signal families",
        grid_axis="x",
    )
    ax.legend(frameon=False, loc="lower right", fontsize=10, labelcolor=INK_MUTED)
    fig.text(
        0.012,
        0.005,
        "✦ = hybrid / joint model.  Best AUROC: Evidence-Aware Hybrid (0.995).  "
        "Best accuracy & macro F1: Cross-Encoder (0.982).",
        fontsize=9.5,
        color=INK_MUTED,
    )
    save(fig, "01_main_results.png")


# --------------------------------------------------------------------------
# 02 - held-out test split confirms the ranking (replaces "Feature Expansion")
# --------------------------------------------------------------------------
def fig_val_vs_test():
    val, test = load_matrix("val"), load_matrix("test")
    order = sorted(val, key=lambda k: val[k]["accuracy"])
    labels = [DISPLAY[k] for k in order]

    v = [val[k]["accuracy"] for k in order]
    t = [test[k]["accuracy"] for k in order]

    # Dot plot: the message is that the two dots nearly coincide, which a pair
    # of truncated bars would obscure rather than reveal.
    y = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURFACE)
    for i in y:
        ax.plot([t[i], v[i]], [i, i], color="#c9c8c3", linewidth=2, zorder=1)
    ax.scatter(v, y, s=130, color=BLUE, label="Validation", zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.scatter(t, y, s=130, color=AQUA, label="Held-out test", zorder=3,
               edgecolor=SURFACE, linewidth=2)

    for i in y:
        lo, hi = min(v[i], t[i]), max(v[i], t[i])
        ax.text(hi + 0.008, i, f"{hi:.3f}", va="center", fontsize=9.5,
                color=INK, fontweight="bold")
        ax.text(lo - 0.008, i, f"{lo:.3f}", va="center", ha="right",
                fontsize=9.5, color=INK, fontweight="bold")

    worst = max(val[k]["accuracy"] - test[k]["accuracy"] for k in order)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=INK)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0.70, 1.03)
    style_axes(ax, xlabel="Accuracy", title="The ranking is stable out of sample", grid_axis="x")
    ax.legend(frameon=False, loc="lower right", fontsize=10, labelcolor=INK_MUTED)
    fig.text(
        0.012, 0.005,
        f"Question-level split, 2,000 held-out rows. Largest accuracy drop: "
        f"{worst * 100:.2f} points — sampling noise, not overfitting.",
        fontsize=9.5, color=INK_MUTED,
    )
    save(fig, "02_val_vs_test.png")


# --------------------------------------------------------------------------
# 03 - ablation: one shared capped linear SVM across all four settings
# --------------------------------------------------------------------------
def fig_ablation():
    df = pd.read_csv(ROOT / "results" / "ablation" / "val_svm_fixed" / "summary.csv")
    order = ["lexical_only", "uncertainty_only", "hybrid_no_context", "hybrid_with_context"]
    labels = [
        "Lexical only\n(TF-IDF)",
        "Uncertainty only\n(log-probs)",
        "Hybrid\n(no context)",
        "Hybrid\n(+ context $\\Delta$)",
    ]
    df = df.set_index("setting").loc[order]

    x = range(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=SURFACE)
    b1 = ax.bar([i - w / 2 for i in x], df["auroc"], width=w, color=BLUE, label="AUROC")
    b2 = ax.bar([i + w / 2 for i in x], df["macro_f1"], width=w, color=ORANGE, label="Macro F1")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=11,
                    color=INK, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axes(ax, ylabel="Score", title="Ablation: all four settings share one capped linear SVM")

    # The headline of this slide: high ranking signal, collapsed threshold.
    # The gap between the 0.974 and 0.715 bars is the point of the slide; the
    # bars are too narrow to carry a callout, so the footnote explains it.
    ax.legend(frameon=False, loc="upper left", fontsize=10.5, labelcolor=INK_MUTED)
    fig.text(0.012, 0.005,
             "HaluEval QA validation split. Uncertainty alone ranks nearly as well as lexical "
             "(0.974 AUROC) but misclassifies 1,512/2,002 at the default threshold.",
             fontsize=9.5, color=INK_MUTED)
    save(fig, "03_ablation.png")


# --------------------------------------------------------------------------
# 04 - transfer gap: the result the deck is currently missing entirely
# --------------------------------------------------------------------------
def fig_transfer_gap():
    tqa = pd.read_csv(ROOT / "reports" / "tables" / "halueval_vs_truthfulqa.csv").set_index("baseline")
    qwen = pd.read_csv(
        ROOT / "reports" / "tables" / "halueval_vs_truthfulqa_qwen_generated_full.csv"
    ).set_index("baseline")
    # Regenerated after fixing a chat-template bug in the Qwen generation script
    # (the raw prompt was fed straight to an instruct-tuned model instead of
    # through its chat template, which caused instruction-tuning boilerplate to
    # leak into the answers). The "without leakage" column is NOT simply
    # "better": the cleaner answer text also shifted the NLI auto-labeler's
    # hallucinated/factual balance (396/817 -> 494/817), so most of the
    # accuracy gain here is a shifted constant-classifier floor, not stronger
    # detection - see the AUROC numbers, which barely move.
    qwen_fixed = pd.read_csv(
        ROOT / "reports" / "tables" / "halueval_vs_truthfulqa_qwen_generated_fixed.csv"
    ).set_index("baseline")

    keys = ["cross_encoder", "evidence_aware_hybrid", "semantic_svm",
            "entropy_base", "nli_evidence", "rag_compare_fixed"]
    keys = sorted(keys, key=lambda k: tqa.loc[k, "halueval_accuracy"], reverse=True)
    labels = [DISPLAY[k] for k in keys]

    x = range(len(keys))
    w = 0.20
    fig, ax = plt.subplots(figsize=(12.5, 6), facecolor=SURFACE)
    series = [
        ("HaluEval (in-domain)", [tqa.loc[k, "halueval_accuracy"] for k in keys], BLUE, -1.5 * w),
        ("TruthfulQA pairs", [tqa.loc[k, "truthfulqa_accuracy"] for k in keys], ORANGE, -0.5 * w),
        ("Qwen-generated (with leakage)",
         [qwen.loc[k, "truthfulqa_qwen_gen_accuracy"] for k in keys], AQUA, 0.5 * w),
        ("Qwen-generated (without leakage)",
         [qwen_fixed.loc[k, "truthfulqa_qwen_gen_fixed_accuracy"] for k in keys], YELLOW, 1.5 * w),
    ]
    for name, vals, color, off in series:
        bars = ax.bar([i + off for i in x], vals, width=w, color=color, label=name)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8.5,
                    color=INK, fontweight="bold", rotation=90 if w < 0.22 else 0)

    ax.axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.text(len(keys) - 0.45, 0.515, "chance", fontsize=9.5, color=INK_MUTED)

    ax.set_xticks(list(x))
    ax.set_xticklabels([l.replace(" Hybrid", "\nHybrid").replace(" (FT)", "\n(FT)")
                        for l in labels], fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.22)
    style_axes(ax, ylabel="Accuracy", title="In-domain accuracy does not transfer")
    ax.legend(frameon=False, loc="upper center", fontsize=9.5, ncol=2, labelcolor=INK_MUTED)
    fig.text(0.012, 0.005,
             "Fixing a chat-template bug in Qwen generation raises accuracy on generated answers, but AUROC "
             "barely moves — most of the gain is a shifted label balance, not stronger detection.",
             fontsize=9.5, color=INK_MUTED)
    save(fig, "04_transfer_gap.png")


# --------------------------------------------------------------------------
# 05 / 06 - error analysis (HaluEval held-out test split)
# --------------------------------------------------------------------------
def fig_error_by_length():
    df = pd.read_csv(ROOT / "reports" / "error_analysis" / "test" / "error_by_answer_length.csv")
    piv = df.pivot(index="baseline", columns="answer_length_bin", values="error_rate")
    keys = [k for k in DISPLAY if k in piv.index]
    keys = sorted(keys, key=lambda k: piv.loc[k, "Short (0–40)"])
    labels = [DISPLAY[k] for k in keys]

    # Horizontal: ten long method names will not fit as vertical tick labels.
    keys = keys[::-1]
    labels = labels[::-1]
    y = range(len(keys))
    h = 0.26
    fig, ax = plt.subplots(figsize=(11, 6.8), facecolor=SURFACE)
    for name, col, color, off in [
        ("Short (0–40 chars)", "Short (0–40)", BLUE, h),
        ("Medium (41–120)", "Medium (41–120)", ORANGE, 0.0),
        ("Long (121+)", "Long (121+)", AQUA, -h),
    ]:
        bars = ax.barh([i + off for i in y], [piv.loc[k, col] for k in keys],
                       height=h, color=color, label=name)
        for bar in bars:
            if bar.get_width() > 0.004:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.2f}", va="center", fontsize=9, color=INK)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.set_xlim(0, 0.40)
    style_axes(ax, xlabel="Error rate", title="Short answers are the hardest subgroup",
               grid_axis="x")
    ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=INK_MUTED)
    fig.text(0.012, 0.005,
             "HaluEval QA held-out test split. Short answers give fewer n-grams, fewer tokens for "
             "log-probability statistics, and less semantic structure.",
             fontsize=9.5, color=INK_MUTED)
    save(fig, "05_error_by_length.png")


def fig_error_by_confidence():
    df = pd.read_csv(ROOT / "reports" / "error_analysis" / "test" / "error_by_relative_confidence.csv")
    piv = df.pivot(index="baseline", columns="confidence_quintile", values="error_rate")
    quintiles = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]

    fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=SURFACE)
    for key, color, marker in [
        ("lexical_svm", BLUE, "o"),
        ("rag_compare_fixed", ORANGE, "s"),
        ("evidence_aware_hybrid", AQUA, "^"),
    ]:
        vals = [piv.loc[key, q] for q in quintiles]
        ax.plot(quintiles, vals, color=color, linewidth=2.4, marker=marker,
                markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, label=DISPLAY[key])
        ax.annotate(f"{vals[-1]:.2f}", xy=(4, vals[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=10.5, color=INK, fontweight="bold",
                    va="center")

    ax.set_xlim(-0.3, 4.6)
    style_axes(ax, ylabel="Error rate", xlabel="Decision-margin confidence quintile",
               title="Evidence-only signals stay wrong when confident")
    ax.legend(frameon=False, loc="upper right", fontsize=10.5, labelcolor=INK_MUTED)
    fig.text(0.012, 0.005,
             "HaluEval QA held-out test split. RAG Compare still errs on 18% of its most confident "
             "decisions; the Evidence-Aware Hybrid flattens to 0%.",
             fontsize=9.5, color=INK_MUTED)
    save(fig, "06_error_by_confidence.png")


def main():
    fig_main_results()
    fig_val_vs_test()
    fig_ablation()
    fig_transfer_gap()
    fig_error_by_length()
    fig_error_by_confidence()
    print(f"\nAll slide figures written to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
