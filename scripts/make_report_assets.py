import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.reporting import build_plots, build_report_table


def main() -> None:
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results")
    build_report_table(results_dir, tables_dir / "initial_results.csv")
    build_plots(results_dir, figures_dir)


if __name__ == "__main__":
    main()
