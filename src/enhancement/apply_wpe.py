"""Apply WPE single-channel dereverberation (Nakatani et al. 2010).

Uses a fully-vectorised batch-numpy implementation so that all STFT frequency
bins are processed in a single batched np.linalg.solve call instead of a
per-bin Python loop (which would be ~900 s/clip on CPU).
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from nara_wpe.utils import stft, istft

STFT_SIZE = 512
STFT_SHIFT = 128


def _build_y_tilde(Y: np.ndarray, taps: int, delay: int) -> np.ndarray:
    """Build the delayed observation tensor.

    Args:
        Y: [freq_bins, frames] complex
        taps: filter order
        delay: prediction delay (frames)
    Returns:
        Y_tilde: [freq_bins, taps, frames] complex  (zero-padded at start)
    """
    freq_bins, frames = Y.shape
    Y_tilde = np.zeros((freq_bins, taps, frames), dtype=Y.dtype)
    for k in range(taps):
        s = delay + k
        if s < frames:
            Y_tilde[:, k, s:] = Y[:, : frames - s]
    return Y_tilde


def _wpe_fast(Y: np.ndarray, taps: int = 5, delay: int = 3, iterations: int = 2) -> np.ndarray:
    """Vectorised single-channel WPE (all freq bins solved in one LAPACK call).

    Args:
        Y: [freq_bins, frames, 1] complex STFT
        taps, delay, iterations: WPE hyper-parameters
    Returns:
        Z: [freq_bins, frames, 1] complex STFT
    """
    freq_bins, frames, _ = Y.shape
    Yf = Y[:, :, 0]                                          # [F, T]

    for _ in range(iterations):
        power = np.maximum(np.abs(Yf) ** 2, 1e-10)          # [F, T]

        Y_tilde = _build_y_tilde(Yf, taps, delay)           # [F, K, T]
        inv_power = 1.0 / power                              # [F, T]

        # R[f] = sum_t Y_tilde[f,:,t] * Y_tilde[f,:,t]^H / power[f,t]
        # einsum: fkt,fjt->fkj  (outer product summed over t)
        Y_tilde_w = Y_tilde * inv_power[:, np.newaxis, :]    # [F, K, T]
        R = np.einsum("fkt,fjt->fkj", Y_tilde_w, Y_tilde.conj())   # [F, K, K]

        # P[f] = sum_t Y_tilde[f,:,t] * conj(Y[f,t]) / power[f,t]
        P = np.einsum("fkt,ft->fk", Y_tilde_w, Yf.conj())   # [F, K]

        # Tikhonov regularisation to avoid singular matrices
        R += np.eye(taps, dtype=R.dtype)[np.newaxis] * (1e-6 * np.mean(np.abs(R)))

        # Batch solve: R[f] @ G[f] = P[f]  →  G: [F, K]
        G = np.linalg.solve(R, P[..., np.newaxis])[..., 0]  # [F, K]

        # Dereverberated signal: Z[f,t] = Y[f,t] - G[f]^H · Y_tilde[f,:,t]
        Yf = Yf - np.einsum("fk,fkt->ft", G.conj(), Y_tilde)

    return Yf[:, :, np.newaxis]


def apply_wpe_dereverb(input_path: str | Path, output_path: str | Path) -> dict:
    """Dereverb a 16kHz mono WAV using WPE."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz: {input_path}"
    assert audio.ndim == 1, f"Expected mono, got shape {audio.shape}: {input_path}"

    Y = stft(audio, size=STFT_SIZE, shift=STFT_SHIFT)       # [frames, freq_bins]
    Y = Y.T[:, :, np.newaxis]                               # [freq_bins, frames, 1]
    Z = _wpe_fast(Y, taps=5, delay=3, iterations=2)

    dereverbed = istft(
        Z[:, :, 0].T, size=STFT_SIZE, shift=STFT_SHIFT
    ).astype(np.float32)[: len(audio)]

    peak = np.max(np.abs(dereverbed))
    if peak > 0.0:
        dereverbed = dereverbed * (0.95 / peak)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, (dereverbed * 32767).astype(np.int16), 16000, subtype="PCM_16")

    return {"output_path": str(output_path)}

