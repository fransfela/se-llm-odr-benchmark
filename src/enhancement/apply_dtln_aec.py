"""Apply DTLN-AEC (TF-Lite) for acoustic echo cancellation.

Based on https://github.com/breizhn/DTLN-aec (MIT license).
Uses the 512-unit model (10.4M params, 3rd place AEC-Challenge ICASSP 2021).
"""

import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
_MODEL_DIR = ROOT / "tmp" / "pretrained_models" / "dtln-aec"
_MODEL_BASE = str(_MODEL_DIR / "dtln_aec_512")

_interpreters = None


def _get_interpreters():
    global _interpreters
    if _interpreters is None:
        from ai_edge_litert.interpreter import Interpreter

        interp_1 = Interpreter(model_path=_MODEL_BASE + "_1.tflite")
        interp_1.allocate_tensors()
        interp_2 = Interpreter(model_path=_MODEL_BASE + "_2.tflite")
        interp_2.allocate_tensors()
        _interpreters = (interp_1, interp_2)
    return _interpreters


def enhance(mic_path: str | Path, ref_path: str | Path, output_path: str | Path) -> None:
    """Cancel echo from mic signal using far-end reference.

    Args:
        mic_path: path to near-end microphone WAV (16 kHz mono).
        ref_path: path to far-end reference/loopback WAV (16 kHz mono).
        output_path: destination for echo-cancelled WAV.
    """
    mic_path = Path(mic_path)
    ref_path = Path(ref_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio, fs = sf.read(str(mic_path), dtype="float32")
    lpb, fs2 = sf.read(str(ref_path), dtype="float32")
    assert fs == 16000 and fs2 == 16000, f"Expected 16 kHz, got {fs}/{fs2}"
    assert audio.ndim == 1 and lpb.ndim == 1, "Only mono files supported"

    # Align lengths
    min_len = min(len(audio), len(lpb))
    audio = audio[:min_len]
    lpb = lpb[:min_len]

    block_len = 512
    block_shift = 128

    interpreter_1, interpreter_2 = _get_interpreters()
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    # Pad audio
    len_audio = len(audio)
    padding = np.zeros(block_len - block_shift, dtype="float32")
    audio = np.concatenate((padding, audio, padding))
    lpb = np.concatenate((padding, lpb, padding))

    # Preallocate
    states_1 = np.zeros(input_details_1[1]["shape"], dtype="float32")
    states_2 = np.zeros(input_details_2[1]["shape"], dtype="float32")
    out_file = np.zeros(len(audio), dtype="float32")
    in_buffer = np.zeros(block_len, dtype="float32")
    in_buffer_lpb = np.zeros(block_len, dtype="float32")
    out_buffer = np.zeros(block_len, dtype="float32")

    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift

    for idx in range(num_blocks):
        # Shift mic buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx * block_shift: idx * block_shift + block_shift]

        # Shift loopback buffer
        in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
        in_buffer_lpb[-block_shift:] = lpb[idx * block_shift: idx * block_shift + block_shift]

        # FFT of mic
        in_block_fft = np.fft.rfft(in_buffer).astype("complex64")
        in_mag = np.abs(in_block_fft).reshape(1, 1, -1).astype("float32")

        # FFT of loopback
        lpb_block_fft = np.fft.rfft(in_buffer_lpb).astype("complex64")
        lpb_mag = np.abs(lpb_block_fft).reshape(1, 1, -1).astype("float32")

        # First model: frequency-domain masking
        interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
        interpreter_1.set_tensor(input_details_1[2]["index"], lpb_mag)
        interpreter_1.set_tensor(input_details_1[1]["index"], states_1)
        interpreter_1.invoke()
        out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
        states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])

        # Apply mask and IFFT
        estimated_block = np.fft.irfft(in_block_fft * out_mask)
        estimated_block = estimated_block.reshape(1, 1, -1).astype("float32")
        in_lpb = in_buffer_lpb.reshape(1, 1, -1).astype("float32")

        # Second model: time-domain refinement
        interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
        interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
        interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
        interpreter_2.invoke()
        out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])

        # Overlap-add
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = 0.0
        out_buffer += np.squeeze(out_block)
        out_file[idx * block_shift: idx * block_shift + block_shift] = out_buffer[:block_shift]

    # Trim to original length
    predicted = out_file[block_len - block_shift: block_len - block_shift + len_audio]
    if np.max(np.abs(predicted)) > 1:
        predicted = predicted / np.max(np.abs(predicted)) * 0.99

    sf.write(str(output_path), predicted, 16000, subtype="PCM_16")
