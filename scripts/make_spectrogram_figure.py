"""
Generate two spectrogram figures:

  fig0_spectrograms.pdf/.png      — 1 row × 6 conditions (main body)
  fig_appendix_spectrograms.pdf/.png — 5 rows × 6 conditions, with
                                        reference transcripts as row labels (appendix)

All conditions: clean, noisy, ns_metricgan, aec_sim, aec_full, dereverb
"""

import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent

DIV        = ROOT / "results" / "clip_level_divergence.csv"
AUDIO_ROOT = ROOT / "data" / "enhanced"
LISTEN_DIR = ROOT / "tmp" / "listen"
TRANS_CSV  = ROOT / "results" / "transcripts_clean.csv"
LISTEN_DIR.mkdir(parents=True, exist_ok=True)

ALL_CONDITIONS = ["clean", "noisy", "ns_metricgan", "aec_sim", "aec_full", "dereverb"]

LABELS = {
    "clean":        "Clean",
    "noisy":        "Noisy",
    "ns_metricgan": "MetricGAN+",
    "aec_sim":      "Echo (sim)",
    "aec_full":     "Echo + DTLN-AEC",
    "dereverb":     "Dereverb",
}
COLOURS = {
    "clean":        "#4D4D4D",
    "noisy":        "#B3B3B3",
    "ns_metricgan": "#1F77B4",
    "aec_sim":      "#D62728",
    "aec_full":     "#8C1515",
    "dereverb":     "#2CA02C",
}

SR    = 16_000
N_FFT = 512
HOP   = 128
N_MELS = 80


def load_audio(condition: str, clip_id: int) -> np.ndarray:
    path = AUDIO_ROOT / condition / f"{clip_id}.wav"
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SR
    assert audio.ndim == 1
    return audio


def log_mel(audio: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import as_strided
    win     = np.hanning(N_FFT).astype(np.float32)
    n_frames = 1 + (len(audio) - N_FFT) // HOP
    frames  = as_strided(audio,
                         shape=(n_frames, N_FFT),
                         strides=(audio.strides[0] * HOP, audio.strides[0]))
    spec    = np.abs(np.fft.rfft(frames * win, n=N_FFT)) ** 2
    f_min, f_max = 0.0, SR / 2
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_pts = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((N_FFT + 1) * hz_pts / SR).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        l, c, r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return np.log(fb @ spec.T + 1e-6)


def all_audio_present(clip_id: int) -> bool:
    return all((AUDIO_ROOT / c / f"{clip_id}.wav").exists() for c in ALL_CONDITIONS)


def spectral_dist(clip_id: int, cond_a: str, cond_b: str) -> float:
    try:
        sa = log_mel(load_audio(cond_a, clip_id))
        sb = log_mel(load_audio(cond_b, clip_id))
        t  = min(sa.shape[1], sb.shape[1])
        return float(np.mean(np.abs(sa[:, :t] - sb[:, :t])))
    except Exception:
        return 0.0


def get_transcript(clip_id: int) -> str:
    if not TRANS_CSV.exists():
        return f"clip {clip_id}"
    df = pd.read_csv(TRANS_CSV)
    row = df[df.clip_id == clip_id]
    if row.empty:
        return f"clip {clip_id}"
    text = str(row.iloc[0].get("ref_text", row.iloc[0].get("whisper_text", f"clip {clip_id}")))
    # Truncate long transcripts for label readability
    return (text[:55] + "…") if len(text) > 55 else text


def pick_clips() -> list[int]:
    """Return 5 clips, each highlighting a different failure story."""
    df = pd.read_csv(DIV)
    sets = {c: set(df.loc[(df.condition == c) & (df.diverged == 1), "clip_id"])
            for c in ["noisy", "ns_metricgan", "aec_sim", "aec_full", "dereverb"]}
    all_ids = set(df.clip_id.unique())

    def candidates_diverging(*conds):
        s = set.intersection(*[sets[c] for c in conds])
        return sorted(c for c in s if all_audio_present(c))

    def clean_hf_energy(clip_id: int) -> float:
        """Mean log-mel energy above mel bin 55 (≈5 kHz) in the clean recording.
        Low value = clean-looking spectrogram."""
        try:
            s = log_mel(load_audio("clean", clip_id))
            return float(np.mean(s[55:, :]))
        except Exception:
            return 999.0

    def best_by_dist(pool, cond_a, cond_b, limit=150):
        # Pre-filter: keep only clips whose clean recording looks actually clean
        hf_threshold = np.percentile(
            [clean_hf_energy(c) for c in pool[:min(limit, len(pool))]],
            40  # bottom 40% high-frequency energy = acceptably clean
        )
        clean_pool = [c for c in pool[:limit] if clean_hf_energy(c) <= hf_threshold]
        if not clean_pool:
            clean_pool = pool[:limit]
        scored = [(spectral_dist(c, cond_a, cond_b), c) for c in clean_pool]
        return max(scored)[1]

    # Row 1: Echo (sim) — far-end speaker dominates
    r1_pool = candidates_diverging("aec_sim")
    r1 = best_by_dist(r1_pool, "clean", "aec_sim")

    # Row 2: MetricGAN+ only (not echo) — spectral smearing
    r2_pool = [c for c in candidates_diverging("ns_metricgan")
               if c not in sets["aec_sim"] and all_audio_present(c)]
    r2 = best_by_dist(r2_pool, "clean", "ns_metricgan")

    # Row 3: AEC recovery (diverged under aec_sim, less so under aec_full)
    r3_pool = [c for c in candidates_diverging("aec_sim")
               if c not in sets["aec_full"] and c != r1 and all_audio_present(c)]
    r3 = best_by_dist(r3_pool, "clean", "aec_sim", limit=200) if r3_pool else r1_pool[1]

    # Row 4: Dereverb — reverberation tail
    r4_pool = [c for c in candidates_diverging("dereverb")
               if c not in sets["aec_sim"] and c not in {r1, r2, r3} and all_audio_present(c)]
    r4 = best_by_dist(r4_pool, "clean", "dereverb")

    # Row 5: Robust clip — diverges under nothing (LLM absorbs all distortions)
    robust_pool = sorted(
        c for c in all_ids
        if all_audio_present(c) and not any(c in sets[s] for s in sets)
        and c not in {r1, r2, r3, r4}
    )
    r5 = robust_pool[42] if len(robust_pool) > 42 else robust_pool[0]

    clips = [r1, r2, r3, r4, r5]
    print(f"Selected clips: {clips}")
    return clips


def apply_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "font.family":       "serif",
        "font.serif":        ["Times New Roman"],
        "font.size":         8,
        "axes.labelsize":    8,
        "axes.titlesize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
    })


def render_row(axes, clip_id: int, vmin: float, vmax: float,
               show_xlabel: bool, row_label: str = "") -> None:
    audio = {c: load_audio(c, clip_id) for c in ALL_CONDITIONS}
    min_len = min(len(a) for a in audio.values())
    audio = {c: a[:min_len] for c, a in audio.items()}
    specs = {c: log_mel(a) for c, a in audio.items()}
    dur = min_len / SR
    for ax, cond in zip(axes, ALL_CONDITIONS):
        ax.imshow(specs[cond], origin="lower", aspect="auto",
                  extent=[0, dur, 0, SR / 2 / 1000],
                  vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest")
        ax.tick_params(length=2, pad=1)
        if show_xlabel:
            ax.set_xlabel("Time (s)", labelpad=1)
        else:
            ax.set_xticklabels([])
    axes[0].set_ylabel(row_label, fontsize=7, labelpad=3, wrap=True)
    axes[0].yaxis.label.set_rotation(0)
    axes[0].yaxis.label.set_ha("right")
    axes[0].yaxis.label.set_va("center")


def global_range(clips: list[int]) -> tuple[float, float]:
    vals = []
    for clip_id in clips:
        for cond in ALL_CONDITIONS:
            try:
                vals.append(log_mel(load_audio(cond, clip_id)).ravel())
            except Exception:
                pass
    flat = np.concatenate(vals)
    return float(np.percentile(flat, 3)), float(np.percentile(flat, 99))


def main() -> None:
    apply_style()
    clips = pick_clips()
    vmin, vmax = global_range(clips)
    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # ── 30 individual panels: 5 clips × 6 conditions ─────────────────────────
    for row_idx, clip_id in enumerate(clips):
        clip_id = int(clip_id)
        transcript = get_transcript(clip_id)

        audio = {c: load_audio(c, clip_id) for c in ALL_CONDITIONS}
        min_len = min(len(a) for a in audio.values())
        audio = {c: a[:min_len] for c, a in audio.items()}
        specs = {c: log_mel(a) for c, a in audio.items()}
        dur   = min_len / SR

        for col_idx, cond in enumerate(ALL_CONDITIONS):
            # Each panel is self-contained with its own colorbar
            fig, ax = plt.subplots(figsize=(1.30, 1.60))

            im = ax.imshow(specs[cond], origin="lower", aspect="auto",
                           extent=[0, dur, 0, SR / 2 / 1000],
                           vmin=vmin, vmax=vmax, cmap="magma",
                           interpolation="nearest")
            ax.tick_params(length=2, labelsize=8)
            ax.set_xlabel("Time (s)", fontsize=9, labelpad=2)

            # Y-axis label only on the Clean (first) column
            if col_idx == 0:
                ax.set_ylabel("Freq (kHz)", fontsize=9)
                ax.tick_params(axis="y", labelsize=8)
            else:
                ax.set_yticklabels([])

            # Colorbar only on the last (Dereverb) column
            if cond == "dereverb":
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("Amplitude", fontsize=8)
                cb.ax.tick_params(labelsize=7)

            fname = out / f"spec_r{row_idx:02d}_{cond}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"  Row {row_idx}: clip {clip_id}  \"{transcript}\"")

    print(f"\nSaved 30 panels: spec_r{{00-04}}_{{condition}}.png")

    # Save WAV files for all clips
    for clip_id in clips:
        clip_id = int(clip_id)
        for cond in ALL_CONDITIONS:
            try:
                a = load_audio(cond, clip_id)
                fname = LISTEN_DIR / f"{clip_id}_{LABELS[cond].replace(' ','_').replace('+','plus')}.wav"
                sf.write(fname, (a * 32767).astype(np.int16), SR)
            except Exception:
                pass
    print(f"WAV files in: {LISTEN_DIR}")

if __name__ == "__main__":
    main()
