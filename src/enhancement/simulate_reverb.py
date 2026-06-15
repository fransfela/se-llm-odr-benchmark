"""Convolve clean speech with a room impulse response and save as int16 WAV."""

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from math import gcd
from pathlib import Path


def _estimate_rt60(rir: np.ndarray, sr: int) -> float:
    """Estimate RT60 via Schroeder backward integration of the RIR energy."""
    # Schroeder backward integral: cumulative sum of squared RIR in reverse
    energy = rir ** 2
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder_db = 10.0 * np.log10(schroeder / (schroeder[0] + 1e-12) + 1e-12)
    # Find where level drops to -60 dB
    idx = np.argmax(schroeder_db <= -60.0)
    if idx == 0:
        # -60 dB not reached; extrapolate from -20 dB point
        idx_20 = np.argmax(schroeder_db <= -20.0)
        if idx_20 == 0:
            return float("nan")
        return float(idx_20 / sr * 3.0)  # RT60 ≈ 3 × RT20
    return float(idx / sr)


def apply_reverb(
    clean_path: str | Path,
    rir_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Convolve clean speech with a RIR and save the reverberant signal."""

    clean_path = Path(clean_path)
    rir_path = Path(rir_path)
    output_path = Path(output_path)

    # Load clean speech (must be 16kHz mono)
    clean, sr_clean = sf.read(clean_path, dtype="float32", always_2d=False)
    assert sr_clean == 16000, f"Expected 16kHz, got {sr_clean}Hz: {clean_path}"
    assert clean.ndim == 1, f"Expected mono, got shape {clean.shape}: {clean_path}"

    # Load RIR — resample to 16kHz if needed
    rir, sr_rir = sf.read(rir_path, dtype="float32", always_2d=False)
    if rir.ndim > 1:
        rir = rir[:, 0]  # take first channel if multi-channel
    if sr_rir != 16000:
        g = gcd(16000, sr_rir)
        rir = resample_poly(rir, 16000 // g, sr_rir // g).astype(np.float32)

    # Convolve and trim to original length
    reverberant = fftconvolve(clean, rir, mode="full")[: len(clean)].astype(np.float32)

    # Peak-normalise to 0.95
    peak = np.max(np.abs(reverberant))
    if peak > 0.0:
        reverberant = reverberant * (0.95 / peak)

    assert np.max(np.abs(reverberant)) <= 1.0, "Post-normalise clip check failed"

    # Save as int16
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, (reverberant * 32767).astype(np.int16), 16000, subtype="PCM_16")

    rt60 = _estimate_rt60(rir, 16000)

    return {
        "output_path": str(output_path),
        "rt60_s": rt60,
        "duration_s": float(len(clean) / 16000),
    }
