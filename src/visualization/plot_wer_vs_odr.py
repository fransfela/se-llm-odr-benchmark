"""Figure 3 — WER vs ODR scatter (single column, 3.25 × 3.25 in)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.plot_utils import (
    CONDITION_COLOURS, CONDITION_LABELS, apply_acl_style, save_figure
)

NON_CLEAN = ["noisy", "ns_metricgan", "ns_diffusion", "dereverb", "aec_sim", "ns_aec_combined"]


def plot_wer_odr_gap(divergence_csv: str | Path) -> None:
    """Scatter: mean_wer (x) vs ODR (y), with y=x reference line and shaded gap region."""
    apply_acl_style()
    df = pd.read_csv(Path(divergence_csv))
    df = df[df["condition"].isin(NON_CLEAN)].dropna(subset=["mean_wer", "odr"])

    fig, ax = plt.subplots(figsize=(3.25, 3.25))

    # Diagonal reference y = x
    lim = max(df["mean_wer"].max(), df["odr"].max()) * 1.15
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=0.9,
            label="WER = ODR", zorder=1)

    # Shade region above diagonal (y > x) — where SE causes more semantic damage than WER shows
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="red", alpha=0.07, zorder=0)
    ax.text(lim * 0.08, lim * 0.82,
            "WER underestimates\nLLM semantic damage",
            fontsize=7, style="italic", color="#990000")

    # Scatter points
    offsets = {"noisy": (0.005, -0.012), "ns_metricgan": (0.005, 0.005),
               "ns_diffusion": (-0.04, 0.007), "dereverb": (0.005, 0.005),
               "aec_sim": (0.005, -0.012), "ns_aec_combined": (0.005, 0.005)}
    for _, row in df.iterrows():
        cond = row["condition"]
        ax.scatter(row["mean_wer"], row["odr"],
                   color=CONDITION_COLOURS.get(cond, "#888888"),
                   s=60, zorder=3)
        dx, dy = offsets.get(cond, (0.005, 0.005))
        ax.text(row["mean_wer"] + dx, row["odr"] + dy,
                CONDITION_LABELS.get(cond, cond), fontsize=7)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean WER (normalised)")
    ax.set_ylabel("Output Divergence Rate (ODR)")
    ax.set_aspect("equal")

    fig.tight_layout()
    save_figure(fig, "fig3_wer_odr_gap")
    plt.close()
