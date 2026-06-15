"""Compute ODR for wav2vec2 pipeline and compare with Whisper."""

import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

np.random.seed(42)
random.seed(42)

CONDITIONS = ["noisy", "ns_metricgan", "aec_sim", "dereverb"]


def _bootstrap_ci(arr, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_odr_and_compare():
    # --- wav2vec2 ODR ---
    clean_csv = ROOT / "results" / "intents_wav2vec2_clean.csv"
    clean = pd.read_csv(clean_csv)
    clean = clean[clean["is_valid"].astype(str).str.lower() == "true"]
    clean_intents = dict(zip(clean["clip_id"].astype(str), clean["predicted_intent"]))

    rows = []
    for cond in CONDITIONS:
        intent_csv = ROOT / "results" / f"intents_wav2vec2_{cond}.csv"
        if not intent_csv.exists():
            print(f"SKIP {cond}: not found")
            continue

        df = pd.read_csv(intent_csv)
        n_total = len(df)
        n_invalid = (df["is_valid"].astype(str).str.lower() == "false").sum()
        df_valid = df[df["is_valid"].astype(str).str.lower() == "true"].copy()
        df_valid["clip_id"] = df_valid["clip_id"].astype(str)

        shared = [c for c in df_valid["clip_id"] if c in clean_intents]
        diverged = np.array([
            int(df_valid.loc[df_valid["clip_id"] == c, "predicted_intent"].values[0]
                != clean_intents[c])
            for c in shared
        ])

        odr = diverged.mean()
        ci_low, ci_high = _bootstrap_ci(diverged)

        wer_csv = ROOT / "results" / f"transcripts_wav2vec2_{cond}.csv"
        wer_mean = float("nan")
        if wer_csv.exists():
            wdf = pd.read_csv(wer_csv)
            wer_mean = wdf["wer"].astype(float).clip(upper=1.0).mean()
        gap = odr - min(wer_mean, 1.0)

        rows.append({
            "condition": cond, "odr": round(odr, 4),
            "odr_ci_low": round(ci_low, 4), "odr_ci_high": round(ci_high, 4),
            "wer_mean": round(wer_mean, 4), "wer_odr_gap": round(gap, 4),
            "n_clips": len(shared), "n_divergent": int(diverged.sum()),
            "n_invalid": int(n_invalid),
            "invalid_rate": round(n_invalid / n_total, 4),
        })
        print(f"wav2vec2 {cond}: ODR={odr:.3f} [{ci_low:.3f},{ci_high:.3f}], "
              f"WER={wer_mean:.3f}, gap={gap:+.3f}")

    w2v_df = pd.DataFrame(rows)
    w2v_df.to_csv(ROOT / "results" / "divergence_wav2vec2.csv", index=False)
    print(f"\nSaved divergence_wav2vec2.csv ({len(rows)} conditions)")

    # --- Cross-model comparison ---
    whisper_df = pd.read_csv(ROOT / "results" / "divergence.csv")
    comp = whisper_df[["condition", "odr"]].rename(columns={"odr": "whisper_odr"})
    comp = comp.merge(
        w2v_df[["condition", "odr"]].rename(columns={"odr": "wav2vec2_odr"}),
        on="condition", how="inner")
    comp["abs_diff"] = (comp["whisper_odr"] - comp["wav2vec2_odr"]).abs().round(4)

    print("\n=== Cross-model comparison ===")
    for _, r in comp.iterrows():
        print(f"  {r.condition:15}: Whisper={r.whisper_odr:.3f}  "
              f"wav2vec2={r.wav2vec2_odr:.3f}  delta={r.abs_diff:.3f}")

    whisper_rank = comp.sort_values("whisper_odr")["condition"].tolist()
    w2v_rank = comp.sort_values("wav2vec2_odr")["condition"].tolist()
    agree = whisper_rank == w2v_rank

    delta = comp["abs_diff"].mean()
    finding = "Architecture-agnostic" if delta < 0.05 else "Architecture-dependent"

    print(f"\nRanking agreement: {'YES' if agree else 'NO'}")
    print(f"Mean ODR delta: {delta:.3f}")
    print(f"Finding: {finding}")

    comp["ranking_agreement"] = agree
    comp["mean_delta"] = round(delta, 4)
    comp["finding"] = finding
    comp.to_csv(ROOT / "results" / "asr_model_comparison.csv", index=False)
    print("Saved asr_model_comparison.csv")


if __name__ == "__main__":
    compute_odr_and_compare()
