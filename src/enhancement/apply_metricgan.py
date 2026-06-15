"""Apply MetricGAN+ noise suppression (SpeechBrain speechbrain/metricgan-plus-voicebank)."""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
_SAVEDIR = str(ROOT / "tmp" / "pretrained_models" / "metricgan-plus")
_MODEL_ID = "speechbrain/metricgan-plus-voicebank"

_model = None


def _get_model():
    global _model
    if _model is None:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        from speechbrain.utils.fetching import LocalStrategy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SpectralMaskEnhancement.from_hparams(
            source=_MODEL_ID,
            savedir=_SAVEDIR,
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
        )
    return _model


def batch_enhance(pairs: list) -> None:
    """Enhance (input_path, output_path) pairs in-place using MetricGAN+.

    Args:
        pairs: list of (input_path, output_path) — both must be 16 kHz mono WAV.
    """
    model = _get_model()
    device = next(model.parameters()).device
    for input_path, output_path in pairs:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio directly with soundfile to avoid SpeechBrain's path-joining bug on Windows
        audio_np, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
        noisy = torch.from_numpy(audio_np).unsqueeze(0).to(device)  # [1, T]
        lengths = torch.ones(1, device=device)

        with torch.no_grad():
            enhanced = model.enhance_batch(noisy, lengths=lengths)  # [1, T]

        enhanced_np = enhanced.squeeze(0).cpu().numpy()
        sf.write(str(output_path), enhanced_np, sr)

