"""Compute all 22 audio quality metrics for every enhanced clip."""

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

ROOT         = Path(__file__).parent.parent.parent
ENHANCED_DIR = ROOT / "data" / "enhanced"
METRICS_CSV  = ROOT / "results" / "metrics.csv"
ERROR_LOG    = ROOT / "results" / "logs" / "metric_errors.jsonl"
MODEL_VERSIONS = ROOT / "results" / "model_versions.txt"

FIELDNAMES = [
    "clip_id", "condition",
    "snr", "si_snr", "si_sdr", "sdr",
    "pesq", "stoi", "visqol", "warpq",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "nisqa", "noresqa", "noresqa_mos",
    "scoreq_nr", "nomad", "squim_mos", "squim_stoi", "squim_pesq", "squim_si_sdr",
    "srmr", "emotion_sim",
]

_RANGES = {
    "pesq": (-0.5, 4.5), "stoi": (0.0, 1.0), "visqol": (0.0, 5.0), "warpq": (0.0, 1.0),
    "dnsmos_sig": (1.0, 5.0), "dnsmos_bak": (1.0, 5.0), "dnsmos_ovrl": (1.0, 5.0),
    "nisqa": (1.0, 5.0), "noresqa": (0.0, 1.0), "noresqa_mos": (1.0, 5.0),
    "scoreq_nr": (1.0, 5.0), "nomad": (0.0, 1.0), "squim_mos": (1.0, 5.0),
    "squim_stoi": (0.0, 1.0), "squim_pesq": (-0.5, 4.5), "srmr": (0.0, float("inf")),
    "emotion_sim": (0.0, 1.0),
}

# ── Singletons (one load per worker process) ─────────────────────────────────
_dnsmos = _nisqa_fn = _scoreq_fn = _squim_obj = _squim_subj = None


def _log_version(name, ver):
    MODEL_VERSIONS.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(MODEL_VERSIONS, "a") as f:
        f.write(f"{ts} | {name} | {ver}\n")


def _load(path: Path) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == 16000 and a.ndim == 1, f"Expected 16kHz mono: {path}"
    return a


def _log_err(clip_id, condition, category, exc):
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(json.dumps({"clip_id": clip_id, "condition": condition,
                            "category": category, "error": str(exc)}) + "\n")


def _compute_distortion(enh: np.ndarray, ref: np.ndarray, clip_id: str, condition: str) -> dict:
    try:
        from torchmetrics.functional.audio import (
            scale_invariant_signal_noise_ratio as _si_snr,
            scale_invariant_signal_distortion_ratio as _si_sdr,
            signal_distortion_ratio as _sdr,
        )
        e = torch.tensor(enh).unsqueeze(0)
        r = torch.tensor(ref).unsqueeze(0)
        snr = float(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - enh) ** 2) + 1e-8)))
        return {"snr": snr, "si_snr": float(_si_snr(e, r)),
                "si_sdr": float(_si_sdr(e, r)), "sdr": float(_sdr(e, r))}
    except Exception as exc:
        _log_err(clip_id, condition, "distortion", exc)
        return {k: float("nan") for k in ("snr", "si_snr", "si_sdr", "sdr")}


def _compute_perceptual(enh_path, ref_path, enh, ref, sr, clip_id, condition) -> dict:
    try:
        from pesq import pesq as _pesq
        from pystoi import stoi as _stoi
        import warp_q
        dur = len(enh) / sr
        try:
            if dur < 1.0:
                raise ValueError(f"too short ({dur:.2f}s < 1.0s required for ViSQOL)")
            import visqol
            visqol_v = float(visqol.visqol(str(ref_path), str(enh_path), mode="speech"))
        except Exception as ve:
            _log_err(clip_id, condition, "visqol", ve)
            visqol_v = float("nan")
        return {"pesq": float(_pesq(sr, ref, enh, "wb")),
                "stoi": float(_stoi(ref, enh, sr, extended=False)),
                "visqol": visqol_v,
                "warpq": float(warp_q.WARPQ(str(ref_path), str(enh_path)).compute())}
    except Exception as exc:
        _log_err(clip_id, condition, "perceptual", exc)
        return {k: float("nan") for k in ("pesq", "stoi", "visqol", "warpq")}


def _compute_non_intrusive(enh: np.ndarray, ref: np.ndarray, enh_path, sr, clip_id, condition) -> dict:
    global _dnsmos, _nisqa_fn, _scoreq_fn, _squim_obj, _squim_subj
    try:
        import speechmetrics
        import torchaudio
        from noresqa import score as _noresqa
        from scoreq import scoreq_predict
        from versa.utterance_metrics.nomad import score_nomad
        if _dnsmos is None:
            _dnsmos = speechmetrics.load("absolute", sr)
            _log_version("dnsmos", "speechmetrics DNSMOS P.835 ONNX")
        if _nisqa_fn is None:
            from nisqa.NISQA_model import nisqa_predict
            _nisqa_fn = nisqa_predict
            _log_version("nisqa", "nisqa GitHub")
        if _scoreq_fn is None:
            _scoreq_fn = scoreq_predict
            _log_version("scoreq_nr", "scoreq GitHub")
        if _squim_obj is None:
            _squim_obj = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
            _log_version("squim_objective", "torchaudio SQUIM_OBJECTIVE")
        if _squim_subj is None:
            _squim_subj = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
            _log_version("squim_subjective", "torchaudio SQUIM_SUBJECTIVE")
        dns = _dnsmos(str(enh_path))
        e_t = torch.tensor(enh).unsqueeze(0)
        r_t = torch.tensor(ref).unsqueeze(0)
        with torch.no_grad():
            stoi_p, pesq_p, sisnr_p = _squim_obj(e_t)
            mos_p = _squim_subj(e_t, r_t)
        return {
            "dnsmos_sig":   float(dns.get("SIG",  math.nan)),
            "dnsmos_bak":   float(dns.get("BAK",  math.nan)),
            "dnsmos_ovrl":  float(dns.get("OVRL", math.nan)),
            "nisqa":        float(_nisqa_fn(str(enh_path))),
            "noresqa":      float(_noresqa(enh, sr, metric_type=0)),
            "noresqa_mos":  float(_noresqa(enh, sr, metric_type=1)),
            "scoreq_nr":    float(_scoreq_fn(str(enh_path), mode="nr")),
            "nomad":        float(score_nomad(enh, sr)),
            "squim_mos":    float(mos_p),
            "squim_stoi":   float(stoi_p),
            "squim_pesq":   float(pesq_p),
            "squim_si_sdr": float(sisnr_p),
        }
    except Exception as exc:
        _log_err(clip_id, condition, "non_intrusive", exc)
        return {k: float("nan") for k in (
            "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "nisqa", "noresqa", "noresqa_mos",
            "scoreq_nr", "nomad", "squim_mos", "squim_stoi", "squim_pesq", "squim_si_sdr")}


def _compute_special(enh: np.ndarray, ref: np.ndarray, sr, clip_id, condition) -> dict:
    try:
        from srmrpy import srmr as _srmr
        from versa.utterance_metrics.emotion import emotion_similarity
        return {
            "srmr":        float(_srmr(enh, sr)[0]),
            "emotion_sim": float(emotion_similarity(ref, enh, sr)["emotion_similarity"]),
        }
    except Exception as exc:
        _log_err(clip_id, condition, "special", exc)
        return {"srmr": float("nan"), "emotion_sim": float("nan")}


def _assert_ranges(row: dict, clip_id: str) -> None:
    for metric, (lo, hi) in _RANGES.items():
        v = row.get(metric)
        if v is not None and not math.isnan(v):
            assert lo <= v <= hi, f"{clip_id} {metric}={v} out of range [{lo}, {hi}]"


def compute_all_metrics(clip_id: str, condition: str,
                        enhanced_path: str | Path, clean_path: str | Path) -> dict:
    """Compute all 22 metrics for one clip. clean_path = original SLURP recording."""
    enh_path, ref_path = Path(enhanced_path), Path(clean_path)
    enh = _load(enh_path)
    ref = _load(ref_path)
    n = min(len(enh), len(ref))
    enh, ref = enh[:n], ref[:n]
    row = {"clip_id": clip_id, "condition": condition}
    row.update(_compute_distortion(enh, ref, clip_id, condition))
    row.update(_compute_perceptual(enh_path, ref_path, enh, ref, 16000, clip_id, condition))
    row.update(_compute_non_intrusive(enh, ref, enh_path, 16000, clip_id, condition))
    row.update(_compute_special(enh, ref, 16000, clip_id, condition))
    _assert_ranges(row, clip_id)
    return row


# ── Parallel main ─────────────────────────────────────────────────────────────
def _compute_row(args: tuple) -> dict:
    return compute_all_metrics(*args)


def _existing_keys() -> set:
    if not METRICS_CSV.exists():
        return set()
    with open(METRICS_CSV, newline="") as f:
        return {(r["clip_id"], r["condition"]) for r in csv.DictReader(f)}


def main() -> None:
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing  = _existing_keys()
    write_hdr = not METRICS_CSV.exists()
    clean_dir = ENHANCED_DIR / "clean"
    jobs = [
        (p.stem, p.parent.name, p, clean_dir / f"{p.stem}.wav")
        for cond_dir in sorted(ENHANCED_DIR.iterdir()) if cond_dir.is_dir()
        for p in sorted(cond_dir.glob("*.wav"))
        if (p.stem, cond_dir.name) not in existing
    ]
    with open(METRICS_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_hdr:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=4) as pool:
            for row in tqdm(pool.map(_compute_row, jobs), total=len(jobs), desc="metrics"):
                writer.writerow(row)
                csvfile.flush()


if __name__ == "__main__":
    main()
