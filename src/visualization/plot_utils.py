"""Shared plot constants and utilities for all EMNLP 2026 figures."""

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent

CONDITION_COLOURS = {
    "clean":           "#4D4D4D",
    "noisy":           "#B3B3B3",
    "ns_metricgan":    "#1F77B4",
    "ns_diffusion":    "#FF7F0E",
    "dereverb":        "#2CA02C",
    "aec_sim":         "#D62728",
    "echo_sim":        "#D62728",
    "aec_full":        "#8C1515",
    "ns_aec_combined": "#9467BD",
    # Dataset B — DNS blind test
    "dns_noisy":       "#B3B3B3",
    "dns_SEGAN":       "#17BECF",
    "dns_NSNet":       "#BCBD22",
    "dns_NSNet2":      "#E377C2",
    "dns_DTLN":        "#7F7F7F",
}

CONDITION_LABELS = {
    "clean":           "Clean",
    "noisy":           "Noisy",
    "ns_metricgan":    "MetricGAN+",
    "ns_diffusion":    "SGMSE+",
    "dereverb":        "Dereverb",
    "aec_sim":         "Echo (sim)",
    "echo_sim":        "Echo (sim)",
    "aec_full":        "Echo + DTLN-AEC",
    "ns_aec_combined": "NS + AEC",
    "dns_noisy":       "DNS Noisy",
    "dns_SEGAN":       "SEGAN",
    "dns_NSNet":       "NS-Net",
    "dns_NSNet2":      "NS-Net 2",
    "dns_DTLN":        "DTLN",
}

METRIC_LABELS = {
    "pesq":       "PESQ",
    "stoi":       "STOI",
    "dnsmos_ovrl":"DNSMOS",
    "nisqa":      "NISQA",
    "scoreq_nr":  "SCOREQ",
    "warpq":      "WARP-Q",
    "srmr":       "SRMR",
    "snr":        "SNR",
    "si_sdr":     "SI-SDR",
    "squim_mos":  "SQUIM-MOS",
}


def apply_acl_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "font.family":       "serif",
        "font.serif":        ["Times New Roman"],
        "font.size":         9,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as both PDF (for LaTeX) and PNG (for preview) to paper/figures/."""
    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)


def sig_marker(p_corrected: float) -> str:
    """Return significance star string based on corrected p-value thresholds."""
    if p_corrected < 0.001:
        return "***"
    if p_corrected < 0.01:
        return "**"
    if p_corrected < 0.05:
        return "*"
    return "n.s."
