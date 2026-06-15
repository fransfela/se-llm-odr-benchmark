"""Simulate acoustic echo by mixing near-end speech with echoed far-end signal."""
# SER definition per ITU-T P.1110

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from pathlib import Path


def simulate_echo(
    near_end_path: str | Path,
    far_end_path: str | Path,
    rir_path: str | Path,
    ser_db: float,
    out_mic_path: str | Path,
    out_ref_path: str | Path,
) -> dict:
    """Produce a mic mix of near-end speech with echoed far-end at a target SER."""

    # Load near-end (SLURP test clip)
    near_end, sr = sf.read(near_end_path, dtype="float32", always_2d=False)
    assert sr == 16000 and near_end.ndim == 1, f"near_end must be 16kHz mono: {near_end_path}"

    # Load far-end (DNS loudspeaker signal causing echo)
    far_end, sr_fe = sf.read(far_end_path, dtype="float32", always_2d=False)
    assert sr_fe == 16000 and far_end.ndim == 1, f"far_end must be 16kHz mono: {far_end_path}"

    # Load RIR
    rir, sr_rir = sf.read(rir_path, dtype="float32", always_2d=False)
    assert sr_rir == 16000, f"RIR must be 16kHz: {rir_path}"
    if rir.ndim > 1:
        rir = rir[:, 0]

    n = len(near_end)

    # Trim/pad far_end reference to near_end length (wrap-around)
    if len(far_end) < n:
        far_end = np.tile(far_end, int(np.ceil(n / len(far_end))))
    far_end_ref = far_end[:n].copy()

    # Step 1: echo = convolve(far_end, rir)[:len(near_end)]
    echo = fftconvolve(far_end_ref, rir, mode="full")[:n].astype(np.float32)

    # Step 2: scale echo so 20*log10(rms_near / rms_echo_scaled) == ser_db
    # → rms_echo_scaled = rms_near / 10^(ser_db/20)
    rms_near = np.sqrt(np.mean(near_end ** 2))
    rms_echo = np.sqrt(np.mean(echo ** 2))
    echo_scaled = echo * (rms_near / (10.0 ** (ser_db / 20.0))) / (rms_echo + 1e-9)
    actual_ser_db = float(20.0 * np.log10(rms_near / (np.sqrt(np.mean(echo_scaled ** 2)) + 1e-9)))

    # Step 3: mic = near_end + scaled_echo
    mic = near_end + echo_scaled
    peak = np.max(np.abs(mic))
    if peak > 1.0:
        mic = mic * (0.99 / peak)
    assert np.max(np.abs(mic)) <= 1.0

    # Step 4: save mic signal and far_end reference (unmodified loudspeaker signal)
    for path, sig in [(out_mic_path, mic), (out_ref_path, far_end_ref)]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (sig * 32767).astype(np.int16), 16000, subtype="PCM_16")

    return {
        "out_mic_path": str(out_mic_path),
        "out_ref_path": str(out_ref_path),
        "actual_ser_db": actual_ser_db,
    }
