import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate error analysis tables from predictions CSV.")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV with label, prediction, hallucination_score.")
    parser.add_argument("--out-dir", default="reports/error_analysis")
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "label", "prediction", "hallucination_score", "candidate_answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = frame.copy()
    work["is_error"] = (work["label"] != work["prediction"]).astype(int)
    work["answer_len_chars"] = work["candidate_answer"].astype(str).str.len()
    work["abs_score_distance"] = (work["hallucination_score"] - 0.5).abs()
    return work


def _summary(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(work)
    errors = int(work["is_error"].sum())
    rows.append({"metric": "num_samples", "value": total})
    rows.append({"metric": "num_errors", "value": errors})
    rows.append({"metric": "error_rate", "value": errors / total if total else 0.0})

    for label_value in sorted(work["label"].unique().tolist()):
        subset = work[work["label"] == label_value]
        rate = subset["is_error"].mean() if len(subset) else 0.0
        rows.append({"metric": f"error_rate_label_{label_value}", "value": float(rate)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.predictions)
    print(f"[error_analysis] Loaded {len(frame)} rows from {args.predictions}")
    work = _prepare(frame)
    print(f"[error_analysis] Total errors: {int(work['is_error'].sum())} / {len(work)}")

    summary = _summary(work)
    summary.to_csv(out_dir / "summary.csv", index=False)

    false_pos = work[(work["label"] == 0) & (work["prediction"] == 1)].sort_values("hallucination_score", ascending=False)
    false_neg = work[(work["label"] == 1) & (work["prediction"] == 0)].sort_values("hallucination_score", ascending=True)
    overconfident_errors = work[work["is_error"] == 1].sort_values("abs_score_distance", ascending=False)

    false_pos.head(args.top_k).to_csv(out_dir / "false_positives_topk.csv", index=False)
    false_neg.head(args.top_k).to_csv(out_dir / "false_negatives_topk.csv", index=False)
    overconfident_errors.head(args.top_k).to_csv(out_dir / "overconfident_errors_topk.csv", index=False)

    work["len_bin"] = pd.cut(
        work["answer_len_chars"],
        bins=[-1, 40, 120, 100000],
        labels=["short_0_40", "medium_41_120", "long_121_plus"],
    )
    by_length = work.groupby("len_bin", observed=True).agg(
        count=("sample_id", "count"),
        error_rate=("is_error", "mean"),
    )
    by_length.to_csv(out_dir / "error_by_answer_length.csv")

    print(
        f"[error_analysis] FP={len(false_pos)} FN={len(false_neg)} "
        f"OverconfidentErrors={len(overconfident_errors)}"
    )
    print(f"Saved error analysis to {out_dir}")


if __name__ == "__main__":
    main()
