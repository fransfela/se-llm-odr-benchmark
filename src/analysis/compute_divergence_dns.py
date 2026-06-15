"""Compute SDR_sent / wSDR_sent for DNS blind test and correlate with MOS."""

import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

np.random.seed(42)
random.seed(42)

# Distance matrix for sentiment (ordinal weighting)
_SENT_DIST = {
    ("positive", "negative"):  1.0,
    ("negative", "positive"):  1.0,
    ("positive", "neutral"):   0.5,
    ("neutral",  "positive"):  0.5,
    ("negative", "neutral"):   0.5,
    ("neutral",  "negative"):  0.5,
    ("positive", "positive"):  0.0,
    ("neutral",  "neutral"):   0.0,
    ("negative", "negative"):  0.0,
}

VALID_SENTIMENTS = {"positive", "neutral", "negative"}
REFERENCE_CONDITION = "dns_noisy"  # baseline for SDR_sent


def _sent_dist(a: str, b: str) -> float:
    return _SENT_DIST.get((a, b), float("nan"))


def _bootstrap_ci(arr: np.ndarray, n: int = 10000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def compute_sdr_sent(sentiment_csvs_dir: str | Path) -> pd.DataFrame:
    """Compute SDR_sent and wSDR_sent from all sentiment CSVs under sentiment_csvs_dir.

    Reference baseline = dns_noisy condition.
    Returns DataFrame saved to results/divergence_dns.csv.
    """
    sentiment_csvs_dir = Path(sentiment_csvs_dir)

    all_frames = []
    for csv_path in sorted(sentiment_csvs_dir.glob("sentiment_dns_*.csv")):
        try:
            df = pd.read_csv(csv_path)
            all_frames.append(df)
        except Exception:
            continue

    if not all_frames:
        raise FileNotFoundError(f"No sentiment CSVs found in {sentiment_csvs_dir}")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined[combined["predicted_sentiment"].isin(VALID_SENTIMENTS)]

    ref_df = combined[combined["condition_name"] == REFERENCE_CONDITION][
        ["clip_id", "predicted_sentiment"]
    ].rename(columns={"predicted_sentiment": "ref_sent"})

    if len(ref_df) == 0:
        raise ValueError(f"No rows for reference condition '{REFERENCE_CONDITION}'")

    merged = combined.merge(ref_df, on="clip_id", how="inner")
    merged = merged[merged["condition_name"] != REFERENCE_CONDITION]

    results = []
    for cname, grp in merged.groupby("condition_name"):
        diverged = (grp["predicted_sentiment"] != grp["ref_sent"]).astype(float).values
        sdr      = float(diverged.mean())
        ci_lo, ci_hi = _bootstrap_ci(diverged)

        distances = grp.apply(
            lambda r: _sent_dist(r["predicted_sentiment"], r["ref_sent"]), axis=1
        ).values.astype(float)
        wsdr = float(np.nanmean(distances))

        # wSDR_sent CI (bootstrap on distances, not just binary)
        finite_d   = distances[~np.isnan(distances)]
        wd_lo, wd_hi = _bootstrap_ci(finite_d) if len(finite_d) > 0 else (float("nan"), float("nan"))

        results.append({
            "condition_name":     cname,
            "n_clips":            len(grp),
            "sdr_sent":           sdr,
            "sdr_ci_lo":          ci_lo,
            "sdr_ci_hi":          ci_hi,
            "wsdr_sent":          wsdr,
            "wsdr_ci_lo":         wd_lo,
            "wsdr_ci_hi":         wd_hi,
        })

    out_df = pd.DataFrame(results).sort_values("condition_name").reset_index(drop=True)
    out_csv = ROOT / "results" / "divergence_dns.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"SDR_sent saved → {out_csv}  ({len(out_df)} conditions)")
    return out_df


def compute_mos_sdr_correlation(metrics_dns_csv: str | Path) -> pd.DataFrame:
    """Correlate subjective MOS (P.808 sig/bak/ovrl) and all 22 objective metrics with SDR_sent.

    Also compares SQUIM predictions vs. P.808 MOS.
    Saves results/mos_sdr_correlation.csv and results/squim_vs_p808.csv.
    """
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import mean_absolute_error
    import math

    metrics_dns_csv = Path(metrics_dns_csv)
    df = pd.read_csv(metrics_dns_csv)
    divergence_csv  = ROOT / "results" / "divergence_dns.csv"
    if not divergence_csv.exists():
        raise FileNotFoundError("Run compute_sdr_sent first")
    div_df = pd.read_csv(divergence_csv)

    merged = df.merge(div_df[["condition_name", "sdr_sent"]], on="condition_name", how="inner")
    if len(merged) == 0:
        raise ValueError("No matching condition_name rows between metrics_dns.csv and divergence_dns.csv")

    OBJECTIVE_METRICS = [
        "snr", "si_snr", "si_sdr", "sdr",
        "pesq", "stoi", "visqol", "warpq",
        "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
        "nisqa", "noresqa", "noresqa_mos", "scoreq_nr", "nomad",
        "squim_mos", "squim_stoi", "squim_pesq", "squim_si_sdr",
        "srmr", "emotion_sim",
    ]
    MOS_DIMS = ["sig_mos", "bak_mos", "ovrl_mos"]
    ALL_PREDICTORS = MOS_DIMS + OBJECTIVE_METRICS
    n_comparisons   = len(ALL_PREDICTORS)

    records = []
    for predictor in ALL_PREDICTORS:
        if predictor not in merged.columns:
            continue
        sub = merged[[predictor, "sdr_sent"]].dropna()
        if len(sub) < 5:
            continue
        rho, pval = spearmanr(sub[predictor], sub["sdr_sent"])
        bonf_p    = min(pval * n_comparisons, 1.0)
        records.append({
            "predictor":   predictor,
            "spearman_rho": round(rho, 4),
            "p_value":      round(pval, 6),
            "bonferroni_p": round(bonf_p, 6),
            "n_clips":      len(sub),
        })

    corr_df = pd.DataFrame(records).sort_values("spearman_rho", ascending=False)
    out_corr = ROOT / "results" / "mos_sdr_correlation.csv"
    corr_df.to_csv(out_corr, index=False)
    print(f"MOS–SDR_sent correlations saved → {out_corr}")

    # SQUIM vs. P.808 agreement
    squim_pairs = [
        ("squim_mos",  "ovrl_mos"),
        ("squim_stoi", "ovrl_mos"),   # STOI doesn't directly map to P.808 dim, use ovrl as proxy
        ("squim_pesq", "sig_mos"),    # PESQ signal quality ↔ sig MOS
    ]
    squim_records = []
    for pred_col, ref_col in squim_pairs:
        if pred_col not in merged.columns or ref_col not in merged.columns:
            continue
        sub = merged[[pred_col, ref_col]].dropna()
        if len(sub) < 5:
            continue
        pear_r, pear_p = pearsonr(sub[pred_col], sub[ref_col])
        spear_r, spear_p = spearmanr(sub[pred_col], sub[ref_col])
        mae  = mean_absolute_error(sub[ref_col], sub[pred_col])
        rmse = math.sqrt(((sub[pred_col] - sub[ref_col]) ** 2).mean())
        squim_records.append({
            "squim_metric":  pred_col,
            "p808_dim":      ref_col,
            "pearson_r":     round(pear_r,  4),
            "pearson_p":     round(pear_p,  6),
            "spearman_rho":  round(spear_r, 4),
            "spearman_p":    round(spear_p, 6),
            "mae":           round(mae,     4),
            "rmse":          round(rmse,    4),
            "n_clips":       len(sub),
        })

    squim_df = pd.DataFrame(squim_records)
    out_squim = ROOT / "results" / "squim_vs_p808.csv"
    squim_df.to_csv(out_squim, index=False)
    print(f"SQUIM vs. P.808 saved → {out_squim}")

    return corr_df


if __name__ == "__main__":
    compute_sdr_sent(ROOT / "results")
    compute_mos_sdr_correlation(ROOT / "results" / "metrics_dns.csv")
