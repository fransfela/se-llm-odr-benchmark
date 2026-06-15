"""Add calibrated additive noise to a clean speech file at a specified SNR."""

import numpy as np
import soundfile as sf
from pathlib import Path


def add_noise_at_snr(
    clean_path: str | Path,
    noise_path: str | Path,
    snr_db: float,
    output_path: str | Path,
) -> dict:
    """Mix clean speech with noise at a target SNR and save as int16 WAV."""

    clean_path = Path(clean_path)
    noise_path = Path(noise_path)
    output_path = Path(output_path)

    # Load clean audio
    clean, sr_clean = sf.read(clean_path, dtype="float32", always_2d=False)
    assert sr_clean == 16000, f"Expected 16kHz, got {sr_clean}Hz: {clean_path}"
    assert clean.ndim == 1, f"Expected mono, got shape {clean.shape}: {clean_path}"

    # Load noise audio
    noise, sr_noise = sf.read(noise_path, dtype="float32", always_2d=False)
    assert sr_noise == 16000, f"Expected 16kHz, got {sr_noise}Hz: {noise_path}"
    assert noise.ndim == 1, f"Expected mono, got shape {noise.shape}: {noise_path}"

    # Wrap-around or trim noise to exactly match clean length
    n = len(clean)
    if len(noise) < n:
        repeats = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:n]

    # Compute RMS levels
    rms_clean = np.sqrt(np.mean(clean ** 2))
    rms_noise = np.sqrt(np.mean(noise ** 2))

    # Scale noise so 20*log10(rms_clean / rms_noise_scaled) == snr_db
    # → rms_noise_scaled = rms_clean / 10^(snr_db/20)
    target_rms_noise = rms_clean / (10.0 ** (snr_db / 20.0))
    scale = target_rms_noise / (rms_noise + 1e-9)
    noise_scaled = noise * scale

    # Mix
    noisy = clean + noise_scaled

    # Check for clipping; rescale to 0.99 peak if needed
    peak = np.max(np.abs(noisy))
    clipped = bool(peak > 1.0)
    if clipped:
        noisy = noisy * (0.99 / peak)

    assert np.max(np.abs(noisy)) <= 1.0, "Post-scale clip check failed"

    # Measure actual SNR after possible rescale (ratio unchanged by global scale)
    rms_noisy_noise = np.sqrt(np.mean(noise_scaled ** 2))
    actual_snr_db = float(20.0 * np.log10(rms_clean / (rms_noisy_noise + 1e-9)))

    # Save as int16
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, (noisy * 32767).astype(np.int16), 16000, subtype="PCM_16")

    return {
        "output_path": str(output_path),
        "actual_snr_db": actual_snr_db,
        "clipped": clipped,
        "duration_s": float(n / 16000),
    }
