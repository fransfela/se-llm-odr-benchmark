"""Compute ODR, WER-ODR gap, and INVALID rate per condition."""

import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

np.random.seed(42)
random.seed(42)

CONDITIONS = ["noisy", "ns_metricgan", "aec_sim", "dereverb"]
INTENT_SLUG = "google_gemini-2_5-flash-lite"


def _bootstrap_ci(arr, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_odr_minimal():
    """Compute ODR and WER-ODR gap for each condition vs clean."""
    # Load clean intents
    clean_csv = ROOT / "results" / f"intents_{INTENT_SLUG}_clean.csv"
    clean = pd.read_csv(clean_csv)
    clean = clean[clean["is_valid"].astype(str).str.lower() == "true"]
    clean_intents = dict(zip(clean["clip_id"].astype(str), clean["predicted_intent"]))

    rows = []
    clip_rows = []

    for cond in CONDITIONS:
        intent_csv = ROOT / "results" / f"intents_{INTENT_SLUG}_{cond}.csv"
        if not intent_csv.exists():
            print(f"SKIP {cond}: {intent_csv.name} not found")
            continue

        df = pd.read_csv(intent_csv)
        n_total = len(df)
        n_invalid = (df["is_valid"].astype(str).str.lower() == "false").sum()
        df_valid = df[df["is_valid"].astype(str).str.lower() == "true"].copy()
        df_valid["clip_id"] = df_valid["clip_id"].astype(str)

        # Only clips that are valid in both clean and condition
        shared = [c for c in df_valid["clip_id"] if c in clean_intents]
        diverged = np.array([
            int(df_valid.loc[df_valid["clip_id"] == c, "predicted_intent"].values[0] != clean_intents[c])
            for c in shared
        ])

        odr = diverged.mean()
        ci_low, ci_high = _bootstrap_ci(diverged)

        # WER
        wer_csv = ROOT / "results" / f"transcripts_{cond}.csv"
        wer_mean_raw = float("nan")
        wer_mean = float("nan")
        if wer_csv.exists():
            wdf = pd.read_csv(wer_csv)
            wer_mean_raw = wdf["wer"].astype(float).mean()
            wer_mean = wdf["wer"].astype(float).clip(upper=1.0).mean()
        gap = odr - min(wer_mean, 1.0)

        if cond == "aec_sim" and wer_mean_raw > 1.0:
            print(f"  NOTE: aec_sim raw WER={wer_mean_raw:.3f} — "
                  f"Whisper transcribing echo signal (expected). "
                  f"Capped at 1.0 for gap computation.")

        # Clip-level divergence
        for c, d in zip(shared, diverged):
            clip_rows.append({
                "clip_id": c, "condition": cond,
                "intent_clean": clean_intents[c],
                "intent_condition": df_valid.loc[df_valid["clip_id"] == c, "predicted_intent"].values[0],
                "diverged": int(d),
            })

        rows.append({
            "condition": cond, "odr": round(odr, 4),
            "odr_ci_low": round(ci_low, 4), "odr_ci_high": round(ci_high, 4),
            "wer_mean_raw": round(wer_mean_raw, 4), "wer_mean": round(wer_mean, 4),
            "wer_odr_gap": round(gap, 4),
            "n_clips": len(shared), "n_divergent": int(diverged.sum()),
            "n_invalid": int(n_invalid), "invalid_rate": round(n_invalid / n_total, 4),
        })
        print(f"{cond}: ODR={odr:.3f} [{ci_low:.3f},{ci_high:.3f}], "
              f"WER={wer_mean:.3f}, gap={gap:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "divergence.csv", index=False)

    clip_df = pd.DataFrame(clip_rows)
    clip_df.to_csv(ROOT / "results" / "clip_level_divergence.csv", index=False)

    print(f"\nSaved divergence.csv ({len(rows)} conditions)")
    print(f"Saved clip_level_divergence.csv ({len(clip_rows)} rows)")
    return out


if __name__ == "__main__":
    compute_odr_minimal()
