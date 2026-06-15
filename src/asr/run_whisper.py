"""Run Whisper transcription across all conditions using faster-whisper (CTranslate2).

Usage:
    python -m src.asr.run_whisper [limit] [model_size]
    python -m src.asr.run_whisper 10 medium    # test: 10 clips, medium model
    python -m src.asr.run_whisper               # full: all clips, large-v3

Auto-detects CUDA GPU. With RTX A2000: ~1s/clip (large-v3, int8_float16).
"""

import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import numpy as np
from tqdm import tqdm

from src.data.slurp_utils import load_slurp_references

np.random.seed(42)
random.seed(42)

ROOT = Path(__file__).parent.parent.parent
ENHANCED_DIR = ROOT / "data" / "enhanced"
TRANSCRIPT_DIR = ROOT / "data" / "processed" / "transcripts"
REFERENCE_JSONL = ROOT / "data" / "raw" / "slurp" / "test" / "slurp_test.jsonl"
MODEL_VERSIONS = ROOT / "results" / "model_versions.txt"
EMPTY_LOG = ROOT / "results" / "logs" / "whisper_empty.txt"

CONDITIONS = ["clean", "noisy", "ns_metricgan", "dereverb", "aec_sim", "ns_aec_combined"]


def _log_version(name, ver):
    MODEL_VERSIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_VERSIONS, "a") as f:
        f.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | {name} | {ver}\n")


def _existing_keys(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {r["clip_id"] for r in csv.DictReader(f)}


def main(limit=None, model_size="large-v3"):
    """Transcribe all conditions with faster-whisper. limit=N for test runs."""
    import torch
    from faster_whisper import WhisperModel

    refs = load_slurp_references(REFERENCE_JSONL)
    print(f"Loaded {len(refs)} reference transcripts")

    if torch.cuda.is_available():
        device, compute = "cuda", "int8_float16"
    else:
        device, compute = "cpu", "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute)
    _log_version("whisper", f"faster-whisper {model_size} {compute} ({device})")
    print(f"faster-whisper {model_size} loaded ({compute}, {device})")

    EMPTY_LOG.parent.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:
        cond_dir = ENHANCED_DIR / condition
        if not cond_dir.exists():
            print(f"SKIP {condition}: directory not found")
            continue

        out_dir = TRANSCRIPT_DIR / condition
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = ROOT / "results" / f"transcripts_{condition}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _existing_keys(csv_path)

        wavs = sorted(cond_dir.glob("*.wav"))
        if limit:
            wavs = wavs[:limit]

        todo = [w for w in wavs if w.stem not in existing]
        if not todo:
            print(f"{condition}: all {len(wavs)} clips done, skipping")
            continue

        print(f"{condition}: {len(todo)} clips to transcribe")
        write_hdr = not csv_path.exists()

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["clip_id", "condition", "whisper_text", "ref_text", "wer"],
            )
            if write_hdr:
                writer.writeheader()

            for wav_path in tqdm(todo, desc=f"whisper {condition}"):
                clip_id = wav_path.stem
                segments, _ = model.transcribe(
                    str(wav_path), language="en", beam_size=5, temperature=0
                )
                text = " ".join(s.text.strip() for s in segments)

                if text.strip() == "":
                    with open(EMPTY_LOG, "a") as ef:
                        ef.write(f"{clip_id}\t{condition}\n")

                ref_text = refs.get(clip_id, {}).get("transcript", "")
                wer = jiwer.wer(ref_text, text) if ref_text else float("nan")

                (out_dir / f"{clip_id}.txt").write_text(text, encoding="utf-8")

                writer.writerow({
                    "clip_id": clip_id,
                    "condition": condition,
                    "whisper_text": text,
                    "ref_text": ref_text,
                    "wer": round(wer, 6) if wer == wer else "nan",
                })
                csvfile.flush()


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    model = sys.argv[2] if len(sys.argv) > 2 else "large-v3"
    main(limit=lim, model_size=model)
