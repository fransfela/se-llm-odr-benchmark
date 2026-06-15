"""Compute 6 core metrics for aec_full condition and append to metrics_core.csv."""

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

ROOT = Path(__file__).parent.parent
ENHANCED_DIR = ROOT / "data" / "enhanced"
CLEAN_DIR = ENHANCED_DIR / "clean"
AEC_FULL_DIR = ENHANCED_DIR / "aec_full"
METRICS_CSV = ROOT / "results" / "metrics_core.csv"
ERROR_LOG = ROOT / "results" / "logs" / "metric_errors.jsonl"

FIELDNAMES = ["clip_id", "condition", "pesq", "stoi", "snr", "si_sdr", "srmr", "squim_mos"]


def load_audio(path):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == 16000 and a.ndim == 1, f"Expected 16kHz mono: {path}"
    return a


def log_err(clip_id, category, exc):
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(json.dumps({"clip_id": clip_id, "condition": "aec_full",
                            "category": category, "error": str(exc)}) + "\n")


def compute_clip(clip_id, enh_path, ref_path):
    enh = load_audio(enh_path)
    ref = load_audio(ref_path)
    n = min(len(enh), len(ref))
    enh, ref = enh[:n], ref[:n]

    row = {"clip_id": clip_id, "condition": "aec_full"}

    # SNR
    try:
        row["snr"] = float(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - enh) ** 2) + 1e-8)))
    except Exception as exc:
        log_err(clip_id, "snr", exc)
        row["snr"] = float("nan")

    # SI-SDR
    try:
        from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        e = torch.tensor(enh).unsqueeze(0)
        r = torch.tensor(ref).unsqueeze(0)
        row["si_sdr"] = float(scale_invariant_signal_distortion_ratio(e, r))
    except Exception as exc:
        log_err(clip_id, "si_sdr", exc)
        row["si_sdr"] = float("nan")

    # PESQ
    try:
        from pesq import pesq
        row["pesq"] = float(pesq(16000, ref, enh, "wb"))
    except Exception as exc:
        log_err(clip_id, "pesq", exc)
        row["pesq"] = float("nan")

    # STOI
    try:
        from pystoi import stoi
        row["stoi"] = float(stoi(ref, enh, 16000, extended=False))
    except Exception as exc:
        log_err(clip_id, "stoi", exc)
        row["stoi"] = float("nan")

    # SRMR
    try:
        from srmrpy import srmr
        row["srmr"] = float(srmr(enh, 16000)[0])
    except Exception as exc:
        log_err(clip_id, "srmr", exc)
        row["srmr"] = float("nan")

    # SQUIM MOS (subjective, needs reference)
    try:
        import torchaudio
        model = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
        e = torch.tensor(enh).unsqueeze(0)
        r = torch.tensor(ref).unsqueeze(0)
        with torch.no_grad():
            mos = model(e, r)
        row["squim_mos"] = float(mos)
    except Exception as exc:
        log_err(clip_id, "squim_mos", exc)
        row["squim_mos"] = float("nan")

    return row


def main():
    # Check existing
    existing = set()
    if METRICS_CSV.exists():
        with open(METRICS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if r["condition"] == "aec_full":
                    existing.add(r["clip_id"])

    # Get base clips (exclude _mic and _ref)
    clips = sorted([
        p for p in AEC_FULL_DIR.glob("*.wav")
        if not p.stem.endswith("_mic") and not p.stem.endswith("_ref")
    ])
    clips = [p for p in clips if p.stem not in existing]
    print(f"Computing metrics for {len(clips)} aec_full clips ({len(existing)} already done)")

    if not clips:
        print("Nothing to compute.")
        return

    write_hdr = not METRICS_CSV.exists()
    with open(METRICS_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()
        for p in tqdm(clips, desc="aec_full metrics"):
            clip_id = p.stem
            ref_path = CLEAN_DIR / f"{clip_id}.wav"
            if not ref_path.exists():
                log_err(clip_id, "missing_ref", f"No clean reference: {ref_path}")
                continue
            row = compute_clip(clip_id, p, ref_path)
            writer.writerow(row)
            csvfile.flush()

    # Print summary
    import pandas as pd
    df = pd.read_csv(METRICS_CSV)
    af = df[df.condition == "aec_full"]
    print(f"\naec_full metrics summary ({len(af)} clips):")
    for m in ["pesq", "stoi", "snr", "si_sdr", "srmr", "squim_mos"]:
        print(f"  {m}: mean={af[m].mean():.2f}, std={af[m].std():.2f}")


if __name__ == "__main__":
    main()
