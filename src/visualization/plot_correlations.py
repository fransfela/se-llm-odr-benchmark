"""Figure 2 — Spearman rho dot plot for metric-ODR correlations."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.plot_utils import (
    CONDITION_COLOURS, METRIC_LABELS, apply_acl_style, save_figure
)

_SIG_COLOUR = CONDITION_COLOURS["ns_diffusion"]   # #FF7F0E
_NS_COLOUR  = "#AAAAAA"


def plot_correlations(correlations_csv: str | Path) -> None:
    """Dot-plot (left) + text summary panel (right) for metric–ODR correlations."""
    apply_acl_style()
    df = pd.read_csv(Path(correlations_csv)).sort_values("rho", ascending=False)

    fig, (ax_dot, ax_txt) = plt.subplots(1, 2, figsize=(6.5, 3.5),
                                          gridspec_kw={"width_ratios": [3, 2]})

    # ── Left: dot plot ─────────────────────────────────────────────────────
    ax_dot.axvline(0, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)

    for i, row in enumerate(df.itertuples()):
        sig   = bool(row.p_bonferroni < 0.05)
        colour = _SIG_COLOUR if sig else _NS_COLOUR
        marker = "o"
        kw = {"color": colour, "markersize": 7, "zorder": 3}
        if sig:
            ax_dot.plot(row.rho, i, marker=marker, **kw)
        else:
            ax_dot.plot(row.rho, i, marker=marker, markerfacecolor="none",
                        markeredgecolor=colour, markersize=7, zorder=3)
        ax_dot.errorbar(row.rho, i,
                        xerr=[[row.rho - row.ci_low], [row.ci_high - row.rho]],
                        fmt="none", ecolor=colour, elinewidth=1, capsize=3)

    labels = [METRIC_LABELS.get(m, m) for m in df["metric"]]
    ax_dot.set_yticks(range(len(df)))
    ax_dot.set_yticklabels(labels, fontsize=8)
    ax_dot.set_xlabel("Spearman ρ (metric score vs. ODR)")
    ax_dot.invert_yaxis()

    # ── Right: text summary ────────────────────────────────────────────────
    ax_txt.axis("off")
    best  = df.iloc[0]
    worst = df.iloc[-1]
    best_label  = METRIC_LABELS.get(best["metric"],  best["metric"])
    worst_label = METRIC_LABELS.get(worst["metric"], worst["metric"])
    best_sig  = "significant" if best["p_bonferroni"]  < 0.05 else "not significant"
    worst_sig = "significant" if worst["p_bonferroni"] < 0.05 else "not significant"
    summary = (
        f"Best predictor:\n  {best_label}\n  ρ = {best['rho']:.2f} ({best_sig})\n\n"
        f"Weakest predictor:\n  {worst_label}\n  ρ = {worst['rho']:.2f} ({worst_sig})\n\n"
        f"Significant metrics (Bonferroni):\n  "
        f"n = {int(df['significant'].sum())} / {len(df)}"
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#CCCCCC"))

    fig.tight_layout()
    save_figure(fig, "fig2_correlations")
    plt.close()
