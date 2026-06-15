"""Compute wav2vec2 aec_full ODR and update divergence_wav2vec2.csv, then regenerate all figures."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

def compute_wav2vec2_aec_full_odr():
    intent_csv = ROOT / "results" / "intents_wav2vec2_aec_full.csv"
    ref_csv = ROOT / "results" / "intents_wav2vec2_clean.csv"
    transcript_csv = ROOT / "results" / "transcripts_wav2vec2_aec_full.csv"

    df = pd.read_csv(intent_csv)
    ref = pd.read_csv(ref_csv)
    trans = pd.read_csv(transcript_csv)

    # Merge with ref intents
    merged = df.merge(
        ref[['clip_id', 'predicted_intent']].rename(columns={'predicted_intent': 'ref_intent'}),
        on='clip_id'
    )
    # Filter invalid
    merged = merged[(merged.predicted_intent != 'INVALID') & (merged.ref_intent != 'INVALID')]

    diverged = (merged.predicted_intent != merged.ref_intent)
    n = len(merged)
    n_div = diverged.sum()
    odr = diverged.mean()

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boots = np.array([diverged.values[rng.randint(0, n, n)].mean() for _ in range(10000)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    # WER (capped)
    wer_capped = trans['wer'].clip(upper=1.0).mean()
    gap = odr - wer_capped

    print(f"wav2vec2 aec_full ODR: {odr:.4f}")
    print(f"  CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  WER (capped): {wer_capped:.4f}")
    print(f"  Gap: {gap:.4f}")
    print(f"  n={n}, n_divergent={n_div}")

    # Update divergence_wav2vec2.csv
    div_csv = ROOT / "results" / "divergence_wav2vec2.csv"
    div_df = pd.read_csv(div_csv)
    new_row = pd.DataFrame([{
        'condition': 'aec_full',
        'odr': round(odr, 4),
        'odr_ci_low': round(ci_lo, 4),
        'odr_ci_high': round(ci_hi, 4),
        'wer_mean': round(wer_capped, 4),
        'wer_odr_gap': round(gap, 4),
        'n_clips': n,
        'n_divergent': int(n_div),
    }])
    if 'aec_full' in div_df.condition.values:
        div_df = div_df[div_df.condition != 'aec_full']
    div_df = pd.concat([div_df, new_row], ignore_index=True)
    div_df.to_csv(div_csv, index=False)
    print(f"\nUpdated {div_csv}")
    print(div_df[['condition','odr','wer_mean','wer_odr_gap']].to_string(index=False))

    return odr, ci_lo, ci_hi, wer_capped, gap


def regenerate_figures():
    from src.visualization.plot_fig1_odr import plot_odr
    from src.visualization.plot_fig2_wer_gap import plot_wer_odr_gap
    plot_odr()
    print("Figure 1 regenerated")
    plot_wer_odr_gap()
    print("Figure 2 regenerated")


if __name__ == "__main__":
    odr, ci_lo, ci_hi, wer, gap = compute_wav2vec2_aec_full_odr()
    regenerate_figures()
    print("\nDone!")
