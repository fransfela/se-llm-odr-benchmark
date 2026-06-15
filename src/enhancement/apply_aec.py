"""AEC via DTLN-AEC (Westhausen & Meyer, ICASSP 2021)."""

from pathlib import Path

from src.enhancement.apply_dtln_aec import enhance as dtln_enhance


def apply_aec(
    mic_path: str | Path,
    ref_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Run DTLN-AEC echo cancellation on mic + reference pair."""
    mic_path = Path(mic_path)
    output_path = Path(output_path)

    dtln_enhance(mic_path, ref_path, output_path)

    return {
        "output_path": str(output_path),
        "method": "DTLN-AEC-512",
    }
