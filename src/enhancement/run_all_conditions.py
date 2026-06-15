"""Master orchestrator: run all 7 enhancement conditions for every SLURP test clip."""

import argparse
import json
import random
import shutil

import numpy as np
import soundfile as sf
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from src.enhancement.simulate_noise import add_noise_at_snr
from src.enhancement.simulate_reverb import apply_reverb
from src.enhancement.simulate_echo import simulate_echo
from src.enhancement.apply_wpe import apply_wpe_dereverb
from src.enhancement.apply_aec import apply_aec

ROOT = Path(__file__).parent.parent.parent
SLURP_TEST = ROOT / "data" / "raw" / "slurp" / "test"
ENHANCED = ROOT / "data" / "enhanced"
LOG_PATH = ROOT / "results" / "logs" / "conditions_run.jsonl"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

CONDITIONS = ["clean", "noisy", "ns_metricgan", "ns_diffusion",
              "dereverb", "echo_sim", "aec_full", "ns_aec_combined"]


def _load_configs():
    with open(ROOT / "configs" / "conditions.yaml") as f:
        cond = yaml.safe_load(f)["conditions"]
    with open(ROOT / "configs" / "experiment.yaml") as f:
        exp = yaml.safe_load(f)
    return cond, exp


def _log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def _pick(files: list, clip_id: str):
    """Deterministically select a file from a list based on clip_id."""
    return files[hash(clip_id) % len(files)]


def _dur(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def _run(clip_id: str, clip_path: Path, condition: str, cfg: dict) -> dict:
    out = ENHANCED / condition / f"{clip_id}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"  SKIP: {clip_id} {condition}")
        return {"clip_id": clip_id, "condition": condition, "status": "skip",
                "output_path": str(out), "duration_s": _dur(out), "error": None}

    try:
        if condition == "clean":
            shutil.copy2(clip_path, out)

        elif condition == "noisy":
            c = cfg["noisy"]
            snr = c["snr_db"][len(c["snr_db"]) // 2]
            noise = _pick(sorted((ROOT / c["noise_source"]).glob("*.wav")), clip_id)
            add_noise_at_snr(clip_path, noise, snr, out)

        elif condition == "ns_metricgan":
            from src.enhancement.apply_metricgan import batch_enhance  # heavy model: lazy import
            noisy = ENHANCED / "noisy" / f"{clip_id}.wav"
            batch_enhance([(noisy, out)])

        elif condition == "ns_diffusion":
            from src.enhancement.apply_sgmse import enhance  # heavy model: lazy import
            noisy = ENHANCED / "noisy" / f"{clip_id}.wav"
            enhance(noisy, out)

        elif condition == "dereverb":
            c = cfg["dereverb"]
            rir = _pick(sorted((ROOT / c["rir_source"]).glob("*.wav")), clip_id)
            tmp = out.with_suffix(".reverb_tmp.wav")
            apply_reverb(clip_path, rir, tmp)
            apply_wpe_dereverb(tmp, out)
            tmp.unlink(missing_ok=True)

        elif condition == "echo_sim":
            c = cfg["echo_sim"]
            ser = c["ser_db"][len(c["ser_db"]) // 2]
            rir = _pick(sorted((ROOT / c["rir_source"]).glob("*.wav")), clip_id)
            fe  = _pick(sorted((ROOT / c["far_end_source"]).glob("*.wav")), clip_id)
            mic     = ENHANCED / "echo_sim" / f"{clip_id}_mic.wav"
            ref_sig = ENHANCED / "echo_sim" / f"{clip_id}_ref.wav"
            simulate_echo(clip_path, fe, rir, ser, mic, ref_sig)
            # echo_sim output = uncancelled mic signal (worst-case echo baseline)
            shutil.copy2(mic, out)

        elif condition == "aec_full":
            c = cfg["aec_full"]
            ser = c["ser_db"][len(c["ser_db"]) // 2]
            rir = _pick(sorted((ROOT / c["rir_source"]).glob("*.wav")), clip_id)
            fe  = _pick(sorted((ROOT / c["far_end_source"]).glob("*.wav")), clip_id)
            mic     = ENHANCED / "aec_full" / f"{clip_id}_mic.wav"
            ref_sig = ENHANCED / "aec_full" / f"{clip_id}_ref.wav"
            simulate_echo(clip_path, fe, rir, ser, mic, ref_sig)
            apply_aec(mic, ref_sig, out)

        elif condition == "ns_aec_combined":
            from src.enhancement.apply_metricgan import batch_enhance  # heavy model: lazy import
            aec = ENHANCED / "aec_full" / f"{clip_id}.wav"
            batch_enhance([(aec, out)])

        return {"clip_id": clip_id, "condition": condition, "status": "ok",
                "output_path": str(out), "duration_s": _dur(out), "error": None}

    except Exception as e:
        return {"clip_id": clip_id, "condition": condition, "status": "error",
                "output_path": str(out), "duration_s": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run enhancement conditions for SLURP test clips.")
    parser.add_argument("--condition", default="all", choices=["all"] + CONDITIONS)
    args = parser.parse_args()

    cond_cfg, _ = _load_configs()
    target = CONDITIONS if args.condition == "all" else [args.condition]
    wavs = sorted((SLURP_TEST / "audio").glob("*.wav"))

    for wav in tqdm(wavs, desc="clips"):
        clip_id = wav.stem
        for condition in target:
            print(f"  {condition}", end="\r", flush=True)
            row = _run(clip_id, wav, condition, cond_cfg)
            print(f"  {condition}: {row['status'].upper():<6}")
            _log(row)


if __name__ == "__main__":
    main()
