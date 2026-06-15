"""Compute core audio quality metrics for every enhanced clip.

Phase 1 metrics: PESQ, STOI, SNR, SI-SDR, SRMR, ScoreQ-NR, SQUIM-MOS.
"""

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

ROOT = Path(__file__).parent.parent.parent
ENHANCED_DIR = ROOT / "data" / "enhanced"
METRICS_CSV = ROOT / "results" / "metrics_core.csv"
ERROR_LOG = ROOT / "results" / "logs" / "metric_errors.jsonl"
MODEL_VERSIONS = ROOT / "results" / "model_versions.txt"

FIELDNAMES = [
    "clip_id", "condition",
    "pesq", "stoi", "snr", "si_sdr", "srmr", "squim_mos",
]

_RANGES = {
    "pesq": (-0.5, 4.5),
    "stoi": (0.0, 1.0),
    "srmr": (0.0, float("inf")),
    "squim_mos": (1.0, 5.0),
}

# Conditions to process (skip ns_diffusion — 0 files)
CONDITIONS = ["clean", "noisy", "ns_metricgan", "dereverb", "aec_sim", "ns_aec_combined"]

# Singletons
_squim_subj = None


def _log_version(name, ver):
    MODEL_VERSIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_VERSIONS, "a") as f:
        f.write(f"{name} | {ver}\n")


def _load(path: Path) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == 16000 and a.ndim == 1, f"Expected 16kHz mono: {path}"
    return a


def _log_err(clip_id, condition, metric, exc):
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(json.dumps({"clip_id": clip_id, "condition": condition,
                            "metric": metric, "error": str(exc)}) + "\n")


def compute_core_metrics(clip_id, condition, enhanced_path, clean_path):
    """Compute 7 core metrics for one clip."""
    enh_path, ref_path = Path(enhanced_path), Path(clean_path)
    enh = _load(enh_path)
    ref = _load(ref_path)
    n = min(len(enh), len(ref))
    enh, ref = enh[:n], ref[:n]

    row = {"clip_id": clip_id, "condition": condition}

    # PESQ (intrusive)
    try:
        from pesq import pesq
        row["pesq"] = float(pesq(16000, ref, enh, "wb"))
    except Exception as exc:
        _log_err(clip_id, condition, "pesq", exc)
        row["pesq"] = float("nan")

    # STOI (intrusive)
    try:
        from pystoi import stoi
        row["stoi"] = float(stoi(ref, enh, 16000, extended=False))
    except Exception as exc:
        _log_err(clip_id, condition, "stoi", exc)
        row["stoi"] = float("nan")

    # SNR (intrusive, direct computation)
    try:
        noise = ref - enh
        row["snr"] = float(10 * np.log10(np.sum(ref ** 2) / (np.sum(noise ** 2) + 1e-8)))
    except Exception as exc:
        _log_err(clip_id, condition, "snr", exc)
        row["snr"] = float("nan")

    # SI-SDR (intrusive)
    try:
        from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        e_t = torch.tensor(enh).unsqueeze(0)
        r_t = torch.tensor(ref).unsqueeze(0)
        row["si_sdr"] = float(scale_invariant_signal_distortion_ratio(e_t, r_t))
    except Exception as exc:
        _log_err(clip_id, condition, "si_sdr", exc)
        row["si_sdr"] = float("nan")

    # SRMR (non-intrusive)
    try:
        from srmrpy import srmr as _srmr
        row["srmr"] = float(_srmr(enh, 16000)[0])
    except Exception as exc:
        _log_err(clip_id, condition, "srmr", exc)
        row["srmr"] = float("nan")

    # SQUIM MOS (non-intrusive, subjective)
    try:
        global _squim_subj
        if _squim_subj is None:
            import torchaudio
            _squim_subj = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
            _squim_subj.eval()
            _log_version("squim_subjective", "torchaudio SQUIM_SUBJECTIVE")
        e_t = torch.tensor(enh).unsqueeze(0)
        r_t = torch.tensor(ref).unsqueeze(0)
        with torch.no_grad():
            mos = _squim_subj(e_t, r_t)
        row["squim_mos"] = float(mos)
    except Exception as exc:
        _log_err(clip_id, condition, "squim_mos", exc)
        row["squim_mos"] = float("nan")

    # Range assertions
    for metric, (lo, hi) in _RANGES.items():
        v = row.get(metric)
        if v is not None and not math.isnan(v):
            if not (lo <= v <= hi):
                _log_err(clip_id, condition, f"{metric}_range", f"{v} not in [{lo},{hi}]")

    return row


def _existing_keys():
    if not METRICS_CSV.exists():
        return set()
    with open(METRICS_CSV, newline="") as f:
        return {(r["clip_id"], r["condition"]) for r in csv.DictReader(f)}


def main(limit=None):
    """Run core metrics on all conditions. limit=N processes only N clips per condition."""
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_keys()
    write_hdr = not METRICS_CSV.exists()
    clean_dir = ENHANCED_DIR / "clean"

    jobs = []
    for cond in CONDITIONS:
        cond_dir = ENHANCED_DIR / cond
        if not cond_dir.exists():
            print(f"SKIP {cond}: directory not found")
            continue
        wavs = sorted(cond_dir.glob("*.wav"))
        if limit:
            wavs = wavs[:limit]
        for p in wavs:
            if (p.stem, cond) not in existing:
                clean_path = clean_dir / f"{p.stem}.wav"
                if clean_path.exists():
                    jobs.append((p.stem, cond, p, clean_path))

    print(f"Processing {len(jobs)} clip-condition pairs")
    with open(METRICS_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()
        for args in tqdm(jobs, desc="core metrics"):
            row = compute_core_metrics(*args)
            writer.writerow(row)
            csvfile.flush()


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
