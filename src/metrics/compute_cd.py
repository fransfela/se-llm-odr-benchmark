"""Compute Mel-Cepstral Distance (MCD) for the dereverb condition."""

import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
ERROR_LOG = ROOT / "results" / "logs" / "cd_errors.jsonl"
FIELDNAMES = ["clip_id", "condition", "cd"]


def compute_cd(enhanced_path, clean_path, n_mfcc=13, n_fft=1024, hop=512):
    enh, sr1 = sf.read(enhanced_path, dtype="float32")
    cln, sr2 = sf.read(clean_path, dtype="float32")
    assert sr1 == 16000 and sr2 == 16000
    n = min(len(enh), len(cln))
    enh, cln = enh[:n], cln[:n]

    mfcc_enh = librosa.feature.mfcc(y=enh, sr=16000, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop)
    mfcc_cln = librosa.feature.mfcc(y=cln, sr=16000, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop)
    # Exclude c0, align frame counts
    t = min(mfcc_enh.shape[1], mfcc_cln.shape[1])
    diff = mfcc_enh[1:, :t] - mfcc_cln[1:, :t]
    mcd = (10 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
    return float(mcd)


def compute_cd_batch(condition="dereverb"):
    enh_dir = ROOT / "data" / "enhanced" / condition
    cln_dir = ROOT / "data" / "enhanced" / "clean"
    output_csv = ROOT / "results" / "metrics_cd.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if output_csv.exists():
        with open(output_csv, newline="") as f:
            existing = {r["clip_id"] for r in csv.DictReader(f)}

    wavs = sorted(enh_dir.glob("*.wav"))
    write_hdr = not output_csv.exists()

    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()

        for wav_path in tqdm(wavs, desc=f"CD {condition}"):
            clip_id = wav_path.stem
            if clip_id in existing:
                continue
            clean_path = cln_dir / f"{clip_id}.wav"
            if not clean_path.exists():
                continue
            try:
                cd = compute_cd(wav_path, clean_path)
            except Exception as e:
                with open(ERROR_LOG, "a") as el:
                    el.write(json.dumps({"clip_id": clip_id,
                        "error": str(e)}) + "\n")
                cd = float("nan")

            writer.writerow({
                "clip_id": clip_id, "condition": condition,
                "cd": round(cd, 4) if cd == cd else "nan",
            })
            csvfile.flush()

    print(f"Saved {output_csv}")


if __name__ == "__main__":
    compute_cd_batch()
