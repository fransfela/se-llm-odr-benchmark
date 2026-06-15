"""Figure 2: WER vs ODR scatter — Whisper (filled) and wav2vec2 (open).

BW-safe: each condition gets a unique marker shape so the figure is
fully readable when printed in grayscale.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.visualization.plot_utils import (
    apply_acl_style, save_figure,
    CONDITION_COLOURS, CONDITION_LABELS,
)

# Distinct markers per condition — readable in B/W
CONDITION_MARKERS = {
    'noisy':        'o',   # circle
    'ns_metricgan': 's',   # square
    'aec_sim':      'D',   # diamond
    'aec_full':     '^',   # triangle up
    'dereverb':     'v',   # triangle down
}


def plot_wer_odr_gap(
        whisper_csv='results/divergence.csv',
        wav2vec2_csv='results/divergence_wav2vec2.csv') -> None:
    apply_acl_style()
    wh = pd.read_csv(whisper_csv)
    wh = wh[wh.condition != 'clean']
    w2v = pd.read_csv(wav2vec2_csv)

    fig, ax = plt.subplots(figsize=(3.25, 3.0))

    lim = 1.05
    ax.plot([0, lim], [0, lim], color='#BBBBBB', linewidth=0.8,
            linestyle='--', zorder=1, label='_nolegend_')
    ax.fill_between([0, lim], [0, lim], 0, color='#F0F7F0',
                    alpha=0.4, zorder=0)

    # Whisper: filled markers
    for _, row in wh.iterrows():
        cond = row.condition
        ax.scatter(row.wer_mean, row.odr,
                   color=CONDITION_COLOURS[cond],
                   marker=CONDITION_MARKERS.get(cond, 'o'),
                   s=55, zorder=5,
                   edgecolors='black', linewidths=0.6)

    # wav2vec2: open markers (same shape, hollow)
    for _, row in w2v.iterrows():
        cond = row.condition
        ax.scatter(row.wer_mean, row.odr,
                   facecolors='none',
                   edgecolors=CONDITION_COLOURS[cond],
                   marker=CONDITION_MARKERS.get(cond, 'o'),
                   s=55, zorder=5, linewidths=1.2)

    # Connecting lines between Whisper and wav2vec2 for same condition
    for _, wh_row in wh.iterrows():
        w2v_row = w2v[w2v.condition == wh_row.condition]
        if len(w2v_row):
            w2v_row = w2v_row.iloc[0]
            ax.plot([wh_row.wer_mean, w2v_row.wer_mean],
                    [wh_row.odr, w2v_row.odr],
                    color=CONDITION_COLOURS[wh_row.condition],
                    linewidth=0.7, alpha=0.4, zorder=3)

    # Manually tuned offsets: each label placed to avoid all overlaps
    label_specs = {
        'dereverb':     (0.507, 0.108, -72,  -8),
        'noisy':        (0.520, 0.135,  +8,  +3),
        'ns_metricgan': (0.614, 0.318,  +8,  -8),
        'aec_sim':      (0.928, 0.836, -82,  -8),
        'aec_full':     (0.649, 0.404,  +8,  +5),
    }
    for cond, (x, y, dx, dy) in label_specs.items():
        if cond not in wh.condition.values:
            continue
        ax.annotate(CONDITION_LABELS[cond],
                    xy=(x, y), xytext=(dx, dy),
                    textcoords='offset points',
                    fontsize=5.5, color='#333333',
                    arrowprops=dict(arrowstyle='-', color='#AAAAAA',
                                   linewidth=0.4)
                    if abs(dx) > 30 else None)

    # Legend: show filled vs open distinction + condition shapes
    legend_elements = [
        Line2D([0], [0], color='#BBBBBB', linestyle='--',
               linewidth=0.8, label='WER = ODR'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#555555', markeredgecolor='black',
               markersize=5, linewidth=0,
               label='Whisper (filled)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='none', markeredgecolor='#555555',
               markersize=5, linewidth=0, markeredgewidth=1.0,
               label='wav2vec2 (open)'),
    ]
    ax.legend(handles=legend_elements, fontsize=5.5,
              loc='upper left', framealpha=0.9,
              edgecolor='#CCCCCC', borderpad=0.4,
              handletextpad=0.4)

    ax.text(0.62, 0.03,
            'LLM partially robust',
            transform=ax.transAxes,
            fontsize=5, fontstyle='italic',
            color='#999999', va='bottom')

    ax.set_xlabel('Mean WER', fontsize=9)
    ax.set_ylabel('Output Divergence Rate (ODR)', fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')

    fig.tight_layout(pad=0.4)
    save_figure(fig, 'fig2_wer_gap')
    plt.close()


if __name__ == '__main__':
    plot_wer_odr_gap()
