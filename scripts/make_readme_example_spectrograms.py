"""Generate clean-vs-condition mel-spectrogram thumbnails for the specific
clips used in the README "Qualitative examples" section. Outputs to
assets/examples/{condition}_{clip_id}.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
AUDIO_ROOT = ROOT / "data" / "enhanced"
OUT_DIR = ROOT / "assets" / "examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR, N_FFT, HOP, N_MELS = 16_000, 512, 128, 80

LABELS = {
    "noisy": "Noisy",
    "ns_metricgan": "MetricGAN+",
    "aec_sim": "Echo (sim)",
    "aec_full": "Echo + DTLN-AEC",
    "dereverb": "Dereverb",
}

# (condition, clip_id) pairs matching the README examples
CLIPS = [
    ("noisy", 10335),
    ("ns_metricgan", 10970),
    ("aec_sim", 14668),
    ("aec_full", 11916),
    ("dereverb", 6477),
]


def load_audio(condition: str, clip_id: int) -> np.ndarray:
    audio, sr = sf.read(AUDIO_ROOT / condition / f"{clip_id}.wav", dtype="float32")
    assert sr == SR
    return audio


def log_mel(audio: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import as_strided
    win = np.hanning(N_FFT).astype(np.float32)
    n_frames = 1 + (len(audio) - N_FFT) // HOP
    frames = as_strided(audio, shape=(n_frames, N_FFT), strides=(audio.strides[0] * HOP, audio.strides[0]))
    spec = np.abs(np.fft.rfft(frames * win, n=N_FFT)) ** 2
    mel_min, mel_max = 0.0, 2595 * np.log10(1 + (SR / 2) / 700)
    mel_pts = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((N_FFT + 1) * hz_pts / SR).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        l, c, r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return np.log(fb @ spec.T + 1e-6)


plt.rcParams.update({"font.size": 8})

for cond, clip_id in CLIPS:
    clean = log_mel(load_audio("clean", clip_id))
    cond_spec = log_mel(load_audio(cond, clip_id))
    vmin = min(clean.min(), cond_spec.min())
    vmax = max(clean.max(), cond_spec.max())

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2))
    for ax, spec, title in zip(axes, [clean, cond_spec], ["Clean", LABELS[cond]]):
        ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"clip {clip_id}", fontsize=8, y=1.02)
    fig.tight_layout()
    out_path = OUT_DIR / f"{cond}_{clip_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
