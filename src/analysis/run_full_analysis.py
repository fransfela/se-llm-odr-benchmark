"""Master analysis runner: Dataset A (SLURP/ODR) + Dataset B (DNS/SDR_sent)."""

from pathlib import Path

from src.analysis.compute_divergence import compute_odr
from src.analysis.compute_correlations import compute_metric_odr_correlations
from src.analysis.statistical_tests import run_all_tests
from src.analysis.compute_divergence_dns import compute_sdr_sent, compute_mos_sdr_correlation
from src.visualization.plot_mos_correlation import plot_mos_correlation

ROOT = Path(__file__).parent.parent.parent


def main() -> None:
    # ── Dataset A: SLURP — intent classification (ODR) ───────────────────────
    print("Step 1/6: Computing ODR (Dataset A)...")
    divergence_df = compute_odr(ROOT / "results" / "intents_gpt.csv")

    print("Step 2/6: Computing metric–ODR correlations (Dataset A)...")
    correlations_df = compute_metric_odr_correlations(
        ROOT / "results" / "metrics.csv",
        ROOT / "results" / "clip_level_divergence.csv",
    )

    print("Step 3/6: Running statistical tests (Dataset A)...")
    tests_df = run_all_tests(
        ROOT / "results" / "divergence.csv",
        ROOT / "results" / "clip_level_divergence.csv",
    )

    # ── Dataset B: DNS blind test — sentiment classification (SDR_sent) ───────
    print("Step 4/6: Computing DNS sentiment divergence SDR_sent (Dataset B)...")
    dns_div_df = compute_sdr_sent(ROOT / "results")

    print("Step 5/6: Computing MOS vs SDR_sent correlations (Dataset B)...")
    dns_corr_df = compute_mos_sdr_correlation(ROOT / "results" / "metrics_dns.csv")

    print("Step 6/6: Generating Figure 5 (MOS correlation plot)...")
    plot_mos_correlation(
        ROOT / "results" / "mos_sdr_correlation.csv",
        ROOT / "results" / "squim_vs_p808.csv",
    )

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Dataset A — conditions analysed:   {len(divergence_df)}")
    print(f"Dataset A — metrics evaluated:     {len(correlations_df)}")
    print(f"Dataset A — statistical tests run: {len(tests_df)}")
    print(f"Dataset B — DNS conditions:        {len(dns_div_df)}")
    print(f"Dataset B — MOS/metric predictors: {len(dns_corr_df)}")
    print(f"Results saved to: results/")
    print(f"Figures saved to: paper/figures/")


if __name__ == "__main__":
    main()
