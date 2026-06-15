"""Apply SGMSE+ diffusion-based speech enhancement (sp-uhh/sgmse-plus)."""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
_CKPT_DIR = ROOT / "tmp" / "pretrained_models" / "sgmse-plus"
_CKPT_CACHE = _CKPT_DIR / "train_vb_29nqe0uh_epoch=115.ckpt"
_HF_REPO = "sp-uhh/speech-enhancement-sgmse"
_HF_FILENAME = "train_vb_29nqe0uh_epoch=115.ckpt"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sgmse.model import ScoreModel
        from huggingface_hub import hf_hub_download

        _CKPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path = _CKPT_CACHE
        if not ckpt_path.exists():
            downloaded = hf_hub_download(
                repo_id=_HF_REPO,
                filename=_HF_FILENAME,
                local_dir=str(_CKPT_CACHE.parent),
            )
            ckpt_path = Path(downloaded)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = ScoreModel.load_from_checkpoint(
            str(ckpt_path), map_location=device, base_dir="", batch_size=1,
            num_workers=0, kwargs=dict(gpu=False), weights_only=False,
        )
        _model.eval()
        _model = _model.to(device)
    return _model


def enhance(input_path: str | Path, output_path: str | Path) -> None:
    """Enhance a single 16 kHz mono WAV using SGMSE+ diffusion model.

    Args:
        input_path: path to noisy 16 kHz mono WAV.
        output_path: destination path for enhanced WAV.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    assert sr == 16000, f"Expected 16kHz, got {sr} Hz: {input_path}"

    model = _get_model()
    device = next(model.parameters()).device

    with torch.no_grad():
        y_tensor = torch.from_numpy(y).unsqueeze(0).to(device)   # [1, T]
        enhanced = model.enhance(
            y_tensor, sampler_type="pc", predictor="reverse_diffusion",
            corrector="ald", N=30, corrector_steps=1, snr=0.5,
        )

    enhanced_np = enhanced.squeeze()
    if isinstance(enhanced_np, torch.Tensor):
        enhanced_np = enhanced_np.cpu().numpy()
    else:
        enhanced_np = np.asarray(enhanced_np).squeeze()
    # Clip to [-1, 1] to avoid float overflow artifacts
    enhanced_np = np.clip(enhanced_np, -1.0, 1.0)
    sf.write(str(output_path), enhanced_np, samplerate=16000, subtype="PCM_16")
