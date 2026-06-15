"""Figure 5: Subjective MOS vs. SDR_sent divergence and SQUIM vs. P.808 agreement."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error

from src.visualization.plot_utils import apply_acl_style, save_figure, CONDITION_COLOURS

ROOT = Path(__file__).parent.parent.parent

# Extend colours for DNS conditions (add any system colour not already in dict)
_DNS_COLOURS = {
    "dns_noisy":  "#B3B3B3",
    "dns_SEGAN":  "#1F77B4",
    "dns_NSNet":  "#FF7F0E",
    "dns_DTLN":   "#2CA02C",
    "dns_NSNet2": "#D62728",
}


def _logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k * (x - x0)))


def plot_mos_correlation(
    mos_sdr_csv:  str | Path,
    squim_p808_csv: str | Path,
) -> None:
    """Two-panel figure:
    Left  — ovrl_mos vs. SDR_sent scatter + logistic curve
    Right — ovrl_mos (P.808) vs. squim_mos scatter with y=x diagonal
    """
    mos_sdr_csv   = Path(mos_sdr_csv)
    squim_p808_csv = Path(squim_p808_csv)

    apply_acl_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.0), constrained_layout=True)

    # ── Panel A: ovrl_mos vs. SDR_sent ────────────────────────────────────────
    ax = axes[0]
    metrics_csv = ROOT / "results" / "metrics_dns.csv"
    div_csv     = ROOT / "results" / "divergence_dns.csv"

    if metrics_csv.exists() and div_csv.exists():
        met_df = pd.read_csv(metrics_csv)
        div_df = pd.read_csv(div_csv)
        plot_df = met_df.merge(div_df[["condition_name", "sdr_sent"]],
                               on="condition_name", how="inner")
        plot_df = plot_df[["ovrl_mos", "sdr_sent", "condition_name"]].dropna()

        for _, row_data in plot_df.iterrows():
            cname = row_data["condition_name"]
            colour = _DNS_COLOURS.get(cname, CONDITION_COLOURS.get(cname, "#888888"))
            ax.scatter(row_data["ovrl_mos"], row_data["sdr_sent"],
                       color=colour, s=25, alpha=0.7, zorder=3,
                       label=cname)

        if len(plot_df) >= 4:
            x_arr = plot_df["ovrl_mos"].values
            y_arr = plot_df["sdr_sent"].values
            try:
                x_fit = np.linspace(x_arr.min() - 0.1, x_arr.max() + 0.1, 200)
                p0 = [y_arr.max(), -1.0, x_arr.mean()]
                popt, _ = curve_fit(_logistic, x_arr, y_arr, p0=p0, maxfev=5000)
                ax.plot(x_fit, _logistic(x_fit, *popt), color="#333333",
                        linewidth=1.5, linestyle="--", label="Logistic fit")
            except Exception:
                pass
            rho, _ = spearmanr(x_arr, y_arr)
            ax.text(0.04, 0.96, f"Spearman ρ = {rho:.2f}",
                    transform=ax.transAxes, va="top", fontsize=8)

    ax.set_xlabel("P.808 ovrl MOS")
    ax.set_ylabel("SDR_sent")
    ax.set_title("(a) MOS vs. Sentiment Divergence Rate")

    # ── Panel B: squim_mos vs. ovrl_mos ───────────────────────────────────────
    ax2 = axes[1]
    if metrics_csv.exists():
        met_df = pd.read_csv(metrics_csv)
        sub = met_df[["squim_mos", "ovrl_mos", "condition_name"]].dropna()

        for _, row_data in sub.iterrows():
            cname  = row_data["condition_name"]
            colour = _DNS_COLOURS.get(cname, CONDITION_COLOURS.get(cname, "#888888"))
            ax2.scatter(row_data["ovrl_mos"], row_data["squim_mos"],
                        color=colour, s=25, alpha=0.7, zorder=3)

        if len(sub) >= 2:
            all_vals = pd.concat([sub["ovrl_mos"], sub["squim_mos"]])
            lo, hi   = all_vals.min() - 0.1, all_vals.max() + 0.1
            ax2.plot([lo, hi], [lo, hi], color="#888888", linewidth=1.0,
                     linestyle=":", label="y = x")

            if squim_p808_csv.exists():
                sq_df = pd.read_csv(squim_p808_csv)
                row = sq_df[sq_df["squim_metric"] == "squim_mos"]
                if not row.empty:
                    r    = row.iloc[0]["pearson_r"]
                    mae  = row.iloc[0]["mae"]
                    ax2.text(0.04, 0.96,
                             f"Pearson r = {r:.2f}\nMAE = {mae:.2f}",
                             transform=ax2.transAxes, va="top", fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax2.set_xlabel("P.808 ovrl MOS")
    ax2.set_ylabel("SQUIM MOS")
    ax2.set_title("(b) SQUIM MOS vs. P.808 MOS")

    # deduplicate legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=min(len(by_label), 5),
               bbox_to_anchor=(0.5, -0.05), fontsize=7, frameon=False)

    save_figure(fig, "fig5_mos_correlation")
    plt.close(fig)
    print("Saved fig5_mos_correlation.pdf + .png → paper/figures/")


if __name__ == "__main__":
    plot_mos_correlation(
        ROOT / "results" / "mos_sdr_correlation.csv",
        ROOT / "results" / "squim_vs_p808.csv",
    )
