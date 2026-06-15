"""Master analysis runner: divergence then correlations."""

from src.analysis.compute_divergence_minimal import compute_odr_minimal
from src.analysis.compute_correlations_minimal import compute_correlations_minimal


def main():
    print("=" * 60)
    print("STEP 1: Output Divergence Rate (ODR)")
    print("=" * 60)
    div = compute_odr_minimal()
    print()

    print("=" * 60)
    print("STEP 2: Metric-ODR Correlations")
    print("=" * 60)
    corr = compute_correlations_minimal()
    print()

    print("=" * 60)
    print("DONE — results saved:")
    print("  results/divergence.csv")
    print("  results/clip_level_divergence.csv")
    print("  results/correlations.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
