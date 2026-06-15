"""Figure 1 — ODR heatmap + bar chart with CIs and significance markers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.plot_utils import (
    CONDITION_COLOURS, CONDITION_LABELS, apply_acl_style, save_figure, sig_marker
)

ROOT = Path(__file__).parent.parent.parent
ORDER = ["noisy", "ns_metricgan", "ns_diffusion", "dereverb", "aec_sim", "ns_aec_combined"]


def plot_odr_results(divergence_csv: str | Path) -> None:
    """Heatmap (left) + horizontal bar chart (right) of ODR per condition."""
    apply_acl_style()
    df = pd.read_csv(Path(divergence_csv))
    df = df[df["condition"].isin(ORDER)].set_index("condition").reindex(ORDER).reset_index()

    # Try to attach significance markers from statistical_tests.csv if available
    sig_map: dict[str, str] = {}
    tests_path = ROOT / "results" / "statistical_tests.csv"
    if tests_path.exists():
        tests = pd.read_csv(tests_path)
        for _, r in tests.iterrows():
            if str(r["test_name"]).startswith("odr>0:"):
                cond = r["test_name"].replace("odr>0:", "")
                sig_map[cond] = sig_marker(float(r["p_value"]))

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(6.5, 3.5),
                                          gridspec_kw={"width_ratios": [1, 3]})

    # ── Left: heatmap (single column) ─────────────────────────────────────
    odr_vals = df["odr"].values.reshape(-1, 1)
    im = ax_heat.imshow(odr_vals, cmap="YlOrRd", aspect="auto", vmin=0, vmax=odr_vals.max())
    ax_heat.set_xticks([])
    ax_heat.set_yticks(range(len(ORDER)))
    ax_heat.set_yticklabels([CONDITION_LABELS.get(c, c) for c in ORDER], fontsize=8)
    ax_heat.set_title("ODR", fontsize=9)
    for i, v in enumerate(df["odr"]):
        ax_heat.text(0, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                     color="white" if v > 0.15 else "black")
    plt.colorbar(im, ax=ax_heat, fraction=0.08, pad=0.04)

    # ── Right: horizontal bar chart ────────────────────────────────────────
    y = np.arange(len(ORDER))
    for i, row in df.iterrows():
        cond   = row["condition"]
        colour = CONDITION_COLOURS.get(cond, "#888888")
        err_lo = float(row["odr"]) - float(row["odr_ci_low"])
        err_hi = float(row["odr_ci_high"]) - float(row["odr"])
        ax_bar.barh(i, row["odr"], color=colour, xerr=[[err_lo], [err_hi]],
                    error_kw={"elinewidth": 1, "capsize": 3}, height=0.6)
        marker = sig_map.get(cond, "")
        if marker:
            ax_bar.text(float(row["odr_ci_high"]) + 0.002, i, marker,
                        va="center", fontsize=8)

    ax_bar.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([CONDITION_LABELS.get(c, c) for c in ORDER], fontsize=8)
    ax_bar.set_xlabel("Output Divergence Rate (ODR)")
    ax_bar.set_xlim(left=-0.01)

    fig.tight_layout()
    save_figure(fig, "fig1_odr_main")
    plt.close()
