"""Compute AECMOS and ERLE for the aec_sim condition."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tmp" / "aecmos"))
from aecmos import AECMOSEstimator

ROOT = Path(__file__).parent.parent.parent
ONNX = ROOT / "tmp" / "aecmos" / "Run_1663829550_Stage_0.onnx"
ERROR_LOG = ROOT / "results" / "logs" / "aecmos_errors.jsonl"
FIELDNAMES = ["clip_id", "condition", "echo_mos", "deg_mos", "erle"]


def compute_erle(mic, enhanced):
    n = min(len(mic), len(enhanced))
    mic, enhanced = mic[:n], enhanced[:n]
    power_mic = np.mean(mic ** 2) + 1e-10
    power_enh = np.mean(enhanced ** 2) + 1e-10
    return float(10 * np.log10(power_mic / power_enh))


def compute_aecmos_batch(condition="aec_sim"):
    estimator = AECMOSEstimator(str(ONNX))
    enh_dir = ROOT / "data" / "enhanced" / condition
    output_csv = ROOT / "results" / "metrics_aecmos.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if output_csv.exists():
        with open(output_csv, newline="") as f:
            existing = {r["clip_id"] for r in csv.DictReader(f)}

    # Enhanced files (exclude _mic and _ref intermediates)
    wavs = sorted(f for f in enh_dir.glob("*.wav")
                   if "_mic" not in f.stem and "_ref" not in f.stem)

    write_hdr = not output_csv.exists()
    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()

        for wav_path in tqdm(wavs, desc=f"AECMOS {condition}"):
            clip_id = wav_path.stem
            if clip_id in existing:
                continue

            mic_path = enh_dir / f"{clip_id}_mic.wav"
            ref_path = enh_dir / f"{clip_id}_ref.wav"

            if not mic_path.exists() or not ref_path.exists():
                with open(ERROR_LOG, "a") as el:
                    el.write(json.dumps({"clip_id": clip_id,
                        "error": "missing _mic or _ref"}) + "\n")
                continue

            try:
                lpb_sig, mic_sig, enh_sig = estimator.read_and_process_audio_files(
                    str(ref_path), str(mic_path), str(wav_path))
                echo_mos, deg_mos = estimator.run(None, lpb_sig, mic_sig, enh_sig)

                mic_raw, _ = sf.read(mic_path, dtype="float32")
                enh_raw, _ = sf.read(wav_path, dtype="float32")
                erle = compute_erle(mic_raw, enh_raw)
            except Exception as e:
                with open(ERROR_LOG, "a") as el:
                    el.write(json.dumps({"clip_id": clip_id,
                        "error": str(e)}) + "\n")
                echo_mos, deg_mos, erle = float("nan"), float("nan"), float("nan")

            writer.writerow({
                "clip_id": clip_id, "condition": condition,
                "echo_mos": round(echo_mos, 4) if echo_mos == echo_mos else "nan",
                "deg_mos": round(deg_mos, 4) if deg_mos == deg_mos else "nan",
                "erle": round(erle, 4) if erle == erle else "nan",
            })
            csvfile.flush()

    print(f"Saved {output_csv}")


if __name__ == "__main__":
    compute_aecmos_batch()
