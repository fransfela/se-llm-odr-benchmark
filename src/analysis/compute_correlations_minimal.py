"""Compute Spearman correlations between audio quality metrics and clip-level ODR."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent

np.random.seed(42)
random.seed(42)

METRICS = ["pesq", "stoi", "si_sdr", "snr", "srmr", "squim_mos"]


def _bootstrap_rho(x, y, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n):
        idx = rng.choice(len(x), size=len(x), replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(r)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def compute_correlations_minimal():
    """Compute Spearman rho between each metric and clip-level divergence."""
    metrics_csv = ROOT / "results" / "metrics_core.csv"
    div_csv = ROOT / "results" / "clip_level_divergence.csv"

    mdf = pd.read_csv(metrics_csv)
    ddf = pd.read_csv(div_csv)
    mdf["clip_id"] = mdf["clip_id"].astype(str)
    ddf["clip_id"] = ddf["clip_id"].astype(str)

    merged = mdf.merge(ddf[["clip_id", "condition", "diverged"]],
                       on=["clip_id", "condition"], how="inner")

    rows = []
    for metric in METRICS:
        if metric not in merged.columns:
            continue
        subset = merged.dropna(subset=[metric])
        x = subset[metric].values
        y = subset["diverged"].values
        rho, p = stats.spearmanr(x, y)
        p_corr = min(p * len(METRICS), 1.0)
        ci_low, ci_high = _bootstrap_rho(x, y)
        sig = p_corr < 0.05
        rows.append({"metric": metric, "rho": round(rho, 4), "p_raw": p,
                      "p_corrected": round(p_corr, 6), "ci_low": round(ci_low, 4),
                      "ci_high": round(ci_high, 4), "significant": sig})

    out = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
    out.to_csv(ROOT / "results" / "correlations.csv", index=False)

    best = out.iloc[0]
    print(f"Best predictor: {best['metric']} rho={best['rho']:.2f}")
    dnsmos = out[out["metric"] == "dnsmos_ovrl"]
    if len(dnsmos):
        d = dnsmos.iloc[0]
        print(f"DNSMOS rho={d['rho']:.2f} ({'sig' if d['significant'] else 'n.s.'})")
    print(f"\nSaved correlations.csv ({len(rows)} metrics)")
    return out


if __name__ == "__main__":
    compute_correlations_minimal()
