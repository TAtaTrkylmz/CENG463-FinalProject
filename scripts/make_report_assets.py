import json
from pathlib import Path

import pandas as pd


def main() -> None:
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)

    metrics_dir = Path("results/metrics")
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        rows.append({"run": path.stem, **json.loads(path.read_text(encoding="utf-8"))})

    if not rows:
        print("No metrics JSON files found under results/metrics.")
        return

    output = Path("reports/tables/initial_results.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved report table to {output}")


if __name__ == "__main__":
    main()
