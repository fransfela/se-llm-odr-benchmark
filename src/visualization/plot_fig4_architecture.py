"""Figure 4: Architecture comparison — Whisper vs wav2vec2 ODR."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visualization.plot_utils import (
    apply_acl_style, save_figure, CONDITION_LABELS,
)


def plot_architecture_comparison(
        whisper_csv='results/divergence.csv',
        wav2vec2_csv='results/divergence_wav2vec2.csv') -> None:
    apply_acl_style()

    whisper = pd.read_csv(whisper_csv)
    whisper = whisper[whisper.condition != 'clean']
    w2v = pd.read_csv(wav2vec2_csv)

    merged = whisper[['condition', 'odr']].merge(
        w2v[['condition', 'odr']], on='condition',
        suffixes=('_whisper', '_wav2vec2'))
    merged = merged.sort_values('odr_whisper', ascending=False)

    conditions = merged.condition.tolist()
    labels = [CONDITION_LABELS[c] for c in conditions]
    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.25, 2.6))

    bars1 = ax.bar(x - width / 2, merged.odr_whisper,
                   width, label='Whisper large-v3',
                   color='#1F77B4', alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width / 2, merged.odr_wav2vec2,
                   width, label='wav2vec2-large',
                   color='#AEC7E8', alpha=0.85, zorder=3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom',
                fontsize=6.5, color='#333333')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom',
                fontsize=6.5, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=12, ha='right')
    ax.set_ylabel('Output Divergence Rate (ODR)', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

    fig.tight_layout(pad=0.4)
    save_figure(fig, 'fig4_architecture')
    plt.close()


if __name__ == '__main__':
    plot_architecture_comparison()
