"""Compute condition-specific correlations with domain-appropriate metrics."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

np.random.seed(42)
random.seed(42)


def _bootstrap_ci_spearman(x, y, n=10000, seed=42):
    rng = np.random.RandomState(seed)
    rhos = []
    for _ in range(n):
        idx = rng.choice(len(x), len(x), replace=True)
        r, _ = stats.spearmanr(x.iloc[idx], y.iloc[idx])
        rhos.append(r)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def compute_extended_correlations():
    div = pd.read_csv(ROOT / "results" / "clip_level_divergence.csv")
    div["clip_id"] = div["clip_id"].astype(str)
    met = pd.read_csv(ROOT / "results" / "metrics_core.csv")
    met["clip_id"] = met["clip_id"].astype(str)

    results = []

    # --- AECMOS on aec_sim ---
    aecmos_path = ROOT / "results" / "metrics_aecmos.csv"
    if aecmos_path.exists():
        aecmos = pd.read_csv(aecmos_path)
        aecmos["clip_id"] = aecmos["clip_id"].astype(str)
        aec_div = div[div.condition == "aec_sim"]
        aec_m = aec_div.merge(aecmos, on="clip_id", how="inner")

        for col in ["echo_mos", "deg_mos", "erle"]:
            sub = aec_m[["diverged", col]].dropna()
            if len(sub) < 30:
                continue
            rho, p = stats.spearmanr(sub[col], sub["diverged"])
            ci_low, ci_high = _bootstrap_ci_spearman(sub[col], sub["diverged"])
            results.append({
                "metric": col, "condition_subset": "aec_sim",
                "n": len(sub), "rho": round(rho, 4),
                "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
                "p_corrected": round(min(p * 3, 1.0), 6),
            })
            print(f"  {col:15} (aec_sim): rho={rho:+.3f} "
                  f"[{ci_low:+.3f},{ci_high:+.3f}] p={min(p*3,1):.4f}")

        # Compare: PESQ on aec_sim
        aec_met = met[met.condition == "aec_sim"]
        aec_pesq = aec_div.merge(aec_met[["clip_id", "pesq"]], on="clip_id")
        sub = aec_pesq[["diverged", "pesq"]].dropna()
        if len(sub) >= 30:
            rho, p = stats.spearmanr(sub["pesq"], sub["diverged"])
            ci_low, ci_high = _bootstrap_ci_spearman(sub["pesq"], sub["diverged"])
            results.append({
                "metric": "pesq", "condition_subset": "aec_sim",
                "n": len(sub), "rho": round(rho, 4),
                "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
                "p_corrected": round(min(p * 3, 1.0), 6),
            })
            print(f"  {'pesq':15} (aec_sim): rho={rho:+.3f} "
                  f"[{ci_low:+.3f},{ci_high:+.3f}]")

    # --- CD on dereverb ---
    cd_path = ROOT / "results" / "metrics_cd.csv"
    if cd_path.exists():
        cd = pd.read_csv(cd_path)
        cd["clip_id"] = cd["clip_id"].astype(str)
        der_div = div[div.condition == "dereverb"]
        der_m = der_div.merge(cd, on="clip_id", how="inner")

        sub = der_m[["diverged", "cd"]].dropna()
        if len(sub) >= 30:
            rho, p = stats.spearmanr(sub["cd"], sub["diverged"])
            ci_low, ci_high = _bootstrap_ci_spearman(sub["cd"], sub["diverged"])
            results.append({
                "metric": "cd", "condition_subset": "dereverb",
                "n": len(sub), "rho": round(rho, 4),
                "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
                "p_corrected": round(min(p * 2, 1.0), 6),
            })
            print(f"  {'cd':15} (dereverb): rho={rho:+.3f} "
                  f"[{ci_low:+.3f},{ci_high:+.3f}]")

        # Compare: SRMR on dereverb
        der_met = met[met.condition == "dereverb"]
        der_srmr = der_div.merge(der_met[["clip_id", "srmr"]], on="clip_id")
        sub = der_srmr[["diverged", "srmr"]].dropna()
        if len(sub) >= 30:
            rho, p = stats.spearmanr(sub["srmr"], sub["diverged"])
            ci_low, ci_high = _bootstrap_ci_spearman(sub["srmr"], sub["diverged"])
            results.append({
                "metric": "srmr", "condition_subset": "dereverb",
                "n": len(sub), "rho": round(rho, 4),
                "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
                "p_corrected": round(min(p * 2, 1.0), 6),
            })
            print(f"  {'srmr':15} (dereverb): rho={rho:+.3f} "
                  f"[{ci_low:+.3f},{ci_high:+.3f}]")

    # --- General metrics on noise conditions ---
    noise_div = div[div.condition.isin(["noisy", "ns_metricgan"])]
    noise_met = met[met.condition.isin(["noisy", "ns_metricgan"])]
    noise_m = noise_div.merge(noise_met, on=["clip_id", "condition"], how="inner")
    for col in ["pesq", "stoi"]:
        sub = noise_m[["diverged", col]].dropna()
        if len(sub) < 30:
            continue
        rho, p = stats.spearmanr(sub[col], sub["diverged"])
        ci_low, ci_high = _bootstrap_ci_spearman(sub[col], sub["diverged"])
        results.append({
            "metric": col, "condition_subset": "noise",
            "n": len(sub), "rho": round(rho, 4),
            "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
            "p_corrected": round(min(p * 2, 1.0), 6),
        })
        print(f"  {col:15} (noise):   rho={rho:+.3f} "
              f"[{ci_low:+.3f},{ci_high:+.3f}]")

    df = pd.DataFrame(results)
    df.to_csv(ROOT / "results" / "extended_correlations.csv", index=False)
    print(f"\nSaved results/extended_correlations.csv ({len(df)} rows)")
    return df


if __name__ == "__main__":
    print("=== Condition-specific correlations ===\n")
    compute_extended_correlations()
