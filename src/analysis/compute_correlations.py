"""Spearman correlation between per-clip audio quality metrics and ODR."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
METRICS = [
    "snr", "si_snr", "si_sdr", "sdr",
    "pesq", "stoi", "visqol", "warpq",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "nisqa", "noresqa", "noresqa_mos",
    "scoreq_nr", "nomad", "squim_mos", "squim_stoi", "squim_pesq", "squim_si_sdr",
    "srmr", "emotion_sim",
]
N_METRICS = len(METRICS)  # Bonferroni n = 22

np.random.seed(42)
random.seed(42)


def _bootstrap_rho(x: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    rhos = [stats.spearmanr(x[rng.choice(len(x), len(x), replace=True)],
                             y[rng.choice(len(x), len(x), replace=True)]).statistic
            for _ in range(n)]
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def compute_metric_odr_correlations(
    metrics_csv: str | Path,
    clip_divergence_csv: str | Path,
) -> pd.DataFrame:
    """Correlate each audio metric with clip-level divergence; return one row per metric."""
    merged = pd.read_csv(Path(metrics_csv)).merge(
        pd.read_csv(Path(clip_divergence_csv))[["clip_id", "condition", "diverged"]],
        on=["clip_id", "condition"], how="inner",
    )

    rows = []
    for metric in METRICS:
        sub = merged.dropna(subset=[metric])
        x = sub[metric].values.astype(float)
        y = sub["diverged"].values.astype(float)

        rho, p = stats.spearmanr(x, y)
        p_bon = min(float(p) * N_METRICS, 1.0)
        ci_lo, ci_hi = _bootstrap_rho(x, y)

        direction = "Higher" if rho > 0 else "Lower"
        effect    = "less" if rho > 0 else "more"
        interp = f"{direction} {metric} → {effect} LLM divergence (rho={rho:.2f})"
        print(interp)

        rows.append({"metric": metric, "rho": round(float(rho), 4),
                     "p_raw": round(float(p), 6), "p_bonferroni": round(p_bon, 6),
                     "ci_low": round(ci_lo, 4), "ci_high": round(ci_hi, 4),
                     "significant": bool(p_bon < 0.05), "interpretation": interp})

    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results" / "correlations.csv", index=False)
    return result


def compute_squim_vs_intrusive(metrics_csv: str | Path) -> pd.DataFrame:
    """Compare SQUIM non-intrusive predictions against intrusive ground truth."""
    df = pd.read_csv(Path(metrics_csv))
    PAIRS = [("squim_stoi", "stoi"), ("squim_pesq", "pesq"), ("squim_si_sdr", "si_sdr")]
    rows = []
    for pred_col, true_col in PAIRS:
        sub = df[[pred_col, true_col]].dropna()
        if sub.empty:
            continue
        x = sub[pred_col].values.astype(float)
        y = sub[true_col].values.astype(float)
        pearson_r, _ = stats.pearsonr(x, y)
        spearman_rho, _ = stats.spearmanr(x, y)
        mae  = float(np.mean(np.abs(x - y)))
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
        rows.append({"pair": f"{pred_col}_vs_{true_col}",
                     "pearson_r": round(float(pearson_r), 4),
                     "spearman_rho": round(float(spearman_rho), 4),
                     "mae": round(mae, 4), "rmse": round(rmse, 4)})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results" / "squim_vs_intrusive.csv", index=False)
    return result


def compute_emotion_odr_correlation(
    metrics_csv: str | Path,
    divergence_csv: str | Path,
) -> dict:
    """Correlate emotion_sim with clip-level divergence. Answers: do emotionally distorted clips diverge more?"""
    merged = pd.read_csv(Path(metrics_csv)).merge(
        pd.read_csv(Path(divergence_csv))[["clip_id", "condition", "diverged"]],
        on=["clip_id", "condition"], how="inner",
    )
    sub = merged.dropna(subset=["emotion_sim"])
    x = sub["emotion_sim"].values.astype(float)
    y = sub["diverged"].values.astype(float)
    rho, p = stats.spearmanr(x, y)
    diverged_emo    = float(sub[sub["diverged"] == 1]["emotion_sim"].mean())
    undiverted_emo  = float(sub[sub["diverged"] == 0]["emotion_sim"].mean())
    t_stat, t_p = stats.ttest_ind(
        sub[sub["diverged"] == 1]["emotion_sim"].dropna(),
        sub[sub["diverged"] == 0]["emotion_sim"].dropna(),
    )
    result = {
        "spearman_rho": round(float(rho), 4), "p_value": round(float(p), 6),
        "mean_emo_diverged": round(diverged_emo, 4),
        "mean_emo_undiverted": round(undiverted_emo, 4),
        "t_stat": round(float(t_stat), 4), "t_p": round(float(t_p), 6),
    }
    pd.DataFrame([result]).to_csv(ROOT / "results" / "emotion_correlation.csv", index=False)
    return result


if __name__ == "__main__":
    df = compute_metric_odr_correlations(
        ROOT / "results" / "metrics.csv",
        ROOT / "results" / "clip_level_divergence.csv",
    )
    best  = df.loc[df["rho"].abs().idxmax()]
    worst = df.loc[df["rho"].abs().idxmin()]
    sig   = "significant" if best["p_bonferroni"] < 0.05 else "not significant"
    print(f"Best predictor:  {best['metric']} (rho={best['rho']:.2f}, {sig})")
    print(f"Worst predictor: {worst['metric']} (rho={worst['rho']:.2f})")
