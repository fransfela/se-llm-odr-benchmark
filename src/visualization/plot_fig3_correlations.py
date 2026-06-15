"""Figure 3: Dual-panel metric correlations — Pearson r and Spearman rho."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

from src.visualization.plot_utils import apply_acl_style, save_figure

METRIC_DISPLAY = {
    'pesq': 'PESQ', 'stoi': 'STOI', 'snr': 'SNR',
    'srmr': 'SRMR', 'si_sdr': 'SI-SDR', 'squim_mos': 'SQUIM-MOS',
}

METRICS_6 = ['pesq', 'stoi', 'snr', 'srmr', 'si_sdr', 'squim_mos']


def _compute_correlations(metrics_csv='results/metrics_core.csv'):
    """Compute Spearman rho and Pearson r for each metric vs WER and vs ODR."""
    met = pd.read_csv(metrics_csv)
    clip = pd.read_csv('results/clip_level_divergence.csv')

    trans_all = []
    for c in ['noisy', 'ns_metricgan', 'aec_sim', 'aec_full', 'dereverb']:
        p = Path(f'results/transcripts_{c}.csv')
        if p.exists():
            try:
                t = pd.read_csv(p)[['clip_id', 'condition', 'wer']]
            except UnicodeDecodeError:
                t = pd.read_csv(p, encoding='latin-1')[['clip_id', 'condition', 'wer']]
            trans_all.append(t)
    wer_df = pd.concat(trans_all)
    wer_df['wer_capped'] = wer_df['wer'].clip(upper=1.0)

    met_filt = met[met.condition.isin(
        ['noisy', 'ns_metricgan', 'aec_sim', 'aec_full', 'dereverb'])]
    merged_wer = met_filt.merge(
        wer_df[['clip_id', 'condition', 'wer_capped']],
        on=['clip_id', 'condition'])
    merged_odr = met_filt.merge(
        clip[['clip_id', 'condition', 'diverged']],
        on=['clip_id', 'condition'])

    results = {}
    for m in METRICS_6:
        sub_w = merged_wer[[m, 'wer_capped']].dropna()
        sub_o = merged_odr[[m, 'diverged']].dropna()
        rho_w, _ = stats.spearmanr(sub_w[m], sub_w['wer_capped'])
        rho_o, _ = stats.spearmanr(sub_o[m], sub_o['diverged'])
        r_w, _ = stats.pearsonr(sub_w[m], sub_w['wer_capped'])
        r_o, _ = stats.pearsonr(sub_o[m], sub_o['diverged'])
        results[m] = {
            'spearman_wer': rho_w, 'spearman_odr': rho_o,
            'pearson_wer': r_w, 'pearson_odr': r_o,
        }
    return results


def _make_panel(ax, labels, wer_vals, odr_vals, xlabel, xlim):
    """Draw a grouped horizontal bar chart with WER and ODR bars.

    BW-safe: WER bars use diagonal hatching (///), ODR bars are solid.
    Colors are kept for on-screen reading but hatching ensures B/W
    differentiation.
    """
    y = np.arange(len(labels))
    h = 0.32

    bars_wer = ax.barh(y + h / 2, wer_vals, height=h, color='#5B9BD5',
            alpha=0.85, label='vs. WER', zorder=3,
            hatch='///', edgecolor='#3A6EA5', linewidth=0.5)
    bars_odr = ax.barh(y - h / 2, odr_vals, height=h, color='#FF7F0E',
            alpha=0.85, label='vs. ODR', zorder=3,
            edgecolor='#CC6600', linewidth=0.5)

    for i, (vw, vo) in enumerate(zip(wer_vals, odr_vals)):
        nudge = -0.02 if vw < 0 else 0.01
        ax.text(vw + nudge, i + h / 2, f'{vw:.2f}',
                va='center', ha='right' if vw < 0 else 'left',
                fontsize=5.5, color='#3A6EA5')
        nudge = -0.02 if vo < 0 else 0.01
        ax.text(vo + nudge, i - h / 2, f'{vo:.2f}',
                va='center', ha='right' if vo < 0 else 'left',
                fontsize=5.5, color='#CC6600')

    ax.axvline(0, color='#999999', linewidth=0.7, linestyle='-', zorder=1)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_xlim(xlim)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, loc='lower left', framealpha=0.9,
              edgecolor='#CCCCCC', borderpad=0.3)


def plot_correlations(
        correlations_csv='results/correlations.csv',
        metrics_csv='results/metrics_core.csv') -> None:
    apply_acl_style()

    data = _compute_correlations(metrics_csv)

    # Sort by Spearman ODR (strongest negative first)
    order = sorted(METRICS_6, key=lambda m: data[m]['spearman_odr'])
    labels = [METRIC_DISPLAY[m] for m in order]

    # --- Panel A: Pearson r ---
    fig_a, ax_a = plt.subplots(figsize=(3.25, 2.4))
    pearson_wer = [data[m]['pearson_wer'] for m in order]
    pearson_odr = [data[m]['pearson_odr'] for m in order]
    _make_panel(ax_a, labels, pearson_wer, pearson_odr,
                'Pearson $r$', (-0.65, 0.15))
    fig_a.tight_layout(pad=0.4)
    save_figure(fig_a, 'fig3a_pearson')
    plt.close(fig_a)

    # --- Panel B: Spearman rho ---
    fig_b, ax_b = plt.subplots(figsize=(3.25, 2.4))
    spearman_wer = [data[m]['spearman_wer'] for m in order]
    spearman_odr = [data[m]['spearman_odr'] for m in order]
    _make_panel(ax_b, labels, spearman_wer, spearman_odr,
                'Spearman $\\rho$', (-0.65, 0.15))
    fig_b.tight_layout(pad=0.4)
    save_figure(fig_b, 'fig3b_spearman')
    plt.close(fig_b)


if __name__ == '__main__':
    plot_correlations()
