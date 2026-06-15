"""Figure 4 — 100% stacked bar chart of error taxonomy per condition."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.plot_utils import CONDITION_LABELS, apply_acl_style, save_figure

ERROR_TYPES = ["domain_shift", "action_confusion", "complete_failure", "other"]
ERROR_LABELS = {
    "domain_shift":      "Domain shift",
    "action_confusion":  "Action confusion",
    "complete_failure":  "Complete failure",
    "other":             "Other",
}
ORDER = ["noisy", "ns_metricgan", "ns_diffusion", "dereverb", "aec_sim", "ns_aec_combined"]


def plot_taxonomy(taxonomy_summary_csv: str | Path) -> None:
    """100% stacked horizontal bars — proportion of each error type per condition."""
    apply_acl_style()
    df = pd.read_csv(Path(taxonomy_summary_csv))

    # Sort by ODR proxy (total_divergent descending) then align to ORDER
    df = df[df["condition"].isin(ORDER)]
    df = df.set_index("condition").reindex(ORDER).dropna(how="all").reset_index()

    palette = sns.color_palette("Set2", n_colors=len(ERROR_TYPES))
    colours = dict(zip(ERROR_TYPES, palette))

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    y = np.arange(len(df))

    lefts = np.zeros(len(df))
    for et in ERROR_TYPES:
        col = f"{et}_n"
        totals = df["total_divergent"].replace(0, np.nan).values
        proportions = df[col].values / totals
        proportions = np.nan_to_num(proportions)
        ax.barh(y, proportions, left=lefts, color=colours[et],
                label=ERROR_LABELS[et], height=0.6)
        lefts += proportions

    # Annotate n= at right edge
    for i, row in df.iterrows():
        ax.text(1.01, i, f"n={int(row['total_divergent'])}",
                va="center", fontsize=7, transform=ax.get_yaxis_transform())

    ax.set_yticks(y)
    ax.set_yticklabels([CONDITION_LABELS.get(c, c) for c in df["condition"]], fontsize=8)
    ax.set_xlabel("Proportion of divergent clips")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig4_taxonomy")
    plt.close()
