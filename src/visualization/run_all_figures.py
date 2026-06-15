"""Master runner: generate all paper figures and LaTeX tables."""

from pathlib import Path
import sys


def check_inputs():
    required = {
        "results/divergence.csv":            "run src/analysis/run_analysis.py first",
        "results/correlations.csv":          "run src/analysis/run_analysis.py first",
        "results/clip_level_divergence.csv": "run src/analysis/run_analysis.py first",
    }
    all_ok = True
    for path, hint in required.items():
        exists = Path(path).exists()
        status = "\u2713" if exists else "\u2717"
        print(f"  {status} {path}")
        if not exists:
            print(f"    -> {hint}")
            all_ok = False
    return all_ok


def main():
    print("Checking required input files...")
    if not check_inputs():
        print("\nNot ready. Run analysis first.")
        sys.exit(1)

    print("\nGenerating figures...")

    from src.visualization.plot_fig1_odr import plot_odr
    plot_odr("results/divergence.csv")
    print("  \u2713 fig1_odr.pdf + .png")

    from src.visualization.plot_fig2_wer_gap import plot_wer_odr_gap
    plot_wer_odr_gap("results/divergence.csv")
    print("  \u2713 fig2_wer_gap.pdf + .png")

    from src.visualization.plot_fig3_correlations import plot_correlations
    plot_correlations("results/correlations.csv", "results/metrics_core.csv")
    print("  \u2713 fig3_correlations.pdf + .png")

    print("\nGenerating LaTeX tables...")

    from src.visualization.generate_tables import (
        generate_table_conditions,
        generate_table_results,
        generate_table_correlations,
    )
    generate_table_conditions()
    print("  \u2713 table1_conditions.tex")

    generate_table_results("results/divergence.csv")
    print("  \u2713 table2_results.tex")

    generate_table_correlations("results/correlations.csv")
    print("  \u2713 table3_correlations.tex")

    print("\nAll outputs ready.")
    print("Figures : paper/figures/")
    print("Tables  : paper/tables/")
    print("Next    : latexmk -pdf -cd paper/emnlp2026_industry.tex")


if __name__ == "__main__":
    main()
