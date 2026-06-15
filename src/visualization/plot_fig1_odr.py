"""Figure 1: Grouped horizontal bar chart of ODR — Whisper vs wav2vec2."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.visualization.plot_utils import (
    apply_acl_style, save_figure, CONDITION_LABELS,
)


def _bootstrap_ci(intent_csv, ref_csv, n_boot=2000, seed=42):
    """Bootstrap 95% CI for ODR from intent CSVs."""
    rng = np.random.RandomState(seed)
    df = pd.read_csv(intent_csv)
    ref = pd.read_csv(ref_csv)
    merged = df.merge(ref[['clip_id', 'predicted_intent']].rename(
        columns={'predicted_intent': 'ref_intent'}), on='clip_id')
    merged = merged[
        (merged.predicted_intent != 'INVALID') & (merged.ref_intent != 'INVALID')]
    diverged = (merged.predicted_intent != merged.ref_intent).values
    n = len(diverged)
    boots = np.array([diverged[rng.randint(0, n, n)].mean()
                      for _ in range(n_boot)])
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def plot_odr(
        whisper_csv='results/divergence.csv',
        wav2vec2_csv='results/divergence_wav2vec2.csv') -> None:
    apply_acl_style()
    wh = pd.read_csv(whisper_csv)
    wh = wh[wh.condition != 'clean'].sort_values('odr', ascending=False)
    w2v = pd.read_csv(wav2vec2_csv)

    merged = wh[['condition', 'odr', 'odr_ci_low', 'odr_ci_high']].merge(
        w2v[['condition', 'odr']].rename(columns={'odr': 'odr_w2v'}),
        on='condition', how='left')
    # Conditions without wav2vec2 data get NaN for odr_w2v
    merged['odr_w2v'] = merged['odr_w2v'].fillna(float('nan'))

    # Bootstrap wav2vec2 CIs
    w2v_ci_lo, w2v_ci_hi = [], []
    for cond in merged.condition:
        intent_p = Path(f'results/intents_wav2vec2_{cond}.csv')
        ref_p = Path('results/intents_wav2vec2_clean.csv')
        if intent_p.exists() and ref_p.exists():
            lo, hi = _bootstrap_ci(str(intent_p), str(ref_p))
        else:
            odr_val = merged.loc[merged.condition == cond, 'odr_w2v'].values[0]
            lo, hi = odr_val, odr_val
        w2v_ci_lo.append(lo)
        w2v_ci_hi.append(hi)
    merged['w2v_ci_low'] = w2v_ci_lo
    merged['w2v_ci_high'] = w2v_ci_hi

    conditions = merged.condition.tolist()
    labels = [CONDITION_LABELS[c] for c in conditions]
    n = len(conditions)
    y = np.arange(n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(3.25, 2.8))

    bars_wh = ax.barh(y + height / 2, merged.odr,
                      height=height, label='Whisper large-v3',
                      color='#1F77B4', alpha=0.88, zorder=3,
                      edgecolor='#0D3F6E', linewidth=0.5)
    # Only draw wav2vec2 bars for conditions that have data
    w2v_vals = merged.odr_w2v.values.copy()
    w2v_plot = np.where(np.isnan(w2v_vals), 0, w2v_vals)
    bars_w2v = ax.barh(y - height / 2, w2v_plot,
                       height=height, label='wav2vec2-large',
                       color='#AEC7E8', alpha=0.88, zorder=3,
                       hatch='///', edgecolor='#5A9EC9', linewidth=0.5)
    # Hide bars for conditions without wav2vec2 data
    for i, val in enumerate(w2v_vals):
        if np.isnan(val):
            bars_w2v[i].set_visible(False)

    xerr_lo = merged.odr.values - merged.odr_ci_low.values
    xerr_hi = merged.odr_ci_high.values - merged.odr.values
    ax.errorbar(merged.odr, y + height / 2,
                xerr=[xerr_lo, xerr_hi],
                fmt='none', color='#333333',
                capsize=2.5, linewidth=0.9, zorder=4)

    w2v_xerr_lo = merged.odr_w2v.values - merged.w2v_ci_low.values
    w2v_xerr_hi = merged.w2v_ci_high.values - merged.odr_w2v.values
    # Only draw error bars for conditions with wav2vec2 data
    w2v_mask = ~np.isnan(merged.odr_w2v.values)
    if w2v_mask.any():
        ax.errorbar(merged.odr_w2v.values[w2v_mask],
                    y[w2v_mask] - height / 2,
                    xerr=[w2v_xerr_lo[w2v_mask], w2v_xerr_hi[w2v_mask]],
                    fmt='none', color='#5A9EC9',
                    capsize=2, linewidth=0.9, zorder=4)

    for bar, val in zip(bars_wh, merged.odr):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left',
                fontsize=6.5, color='#1F77B4')
    for bar, val in zip(bars_w2v, merged.odr_w2v):
        if pd.notna(val):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', ha='left',
                    fontsize=6.5, color='#5A9EC9')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Output Divergence Rate (ODR)', fontsize=9)
    ax.set_xlim(0, max(merged.odr_ci_high.max(),
                       merged.w2v_ci_high.max(),
                       merged.odr_w2v.max()) * 1.25)
    ax.axvline(x=0, color='#cccccc', linewidth=0.8, zorder=0)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=0.3, linestyle='--', zorder=0)
    ax.invert_yaxis()

    fig.tight_layout(pad=0.4)
    save_figure(fig, 'fig1_odr')
    plt.close()


if __name__ == '__main__':
    plot_odr()
