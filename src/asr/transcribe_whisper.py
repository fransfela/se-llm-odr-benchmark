"""Transcribe all WAV files in a condition directory with Whisper large-v3."""

import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import numpy as np
import torch
import whisper
from tqdm import tqdm

from src.data.slurp_utils import load_slurp_references

ROOT = Path(__file__).parent.parent.parent
MODEL_VERSIONS = ROOT / "results" / "model_versions.txt"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def _log_model_version(name: str, version_str: str) -> None:
    MODEL_VERSIONS.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(MODEL_VERSIONS, "a") as f:
        f.write(f"{ts} | {name} | {version_str}\n")


def transcribe_condition(
    condition_dir: str | Path,
    output_dir:    str | Path,
    reference_jsonl: str | Path,
) -> None:
    """Transcribe every WAV in condition_dir with Whisper; write per-clip text and WER CSV."""
    condition_dir  = Path(condition_dir)
    output_dir     = Path(output_dir)
    reference_jsonl = Path(reference_jsonl)

    condition = condition_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference transcripts
    refs = load_slurp_references(reference_jsonl)

    # Load model once
    model = whisper.load_model("large-v3")
    _log_model_version("whisper", "openai/whisper large-v3")

    # Append-mode CSV: skip already-processed clip_ids
    csv_path = ROOT / "results" / f"transcripts_{condition}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing = {r["clip_id"] for r in csv.DictReader(f)}

    empty_log = ROOT / "results" / "logs" / "whisper_empty.txt"
    empty_log.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(
        f for f in condition_dir.glob("*.wav")
        if "_mic" not in f.stem and "_ref" not in f.stem
    )
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["clip_id", "condition", "whisper_text", "ref_text", "wer"],
        )
        if not existing:
            writer.writeheader()

        for wav_path in tqdm(wavs, desc=f"whisper {condition}"):
            clip_id = wav_path.stem
            if clip_id in existing:
                continue

            result = model.transcribe(
                str(wav_path), language="en", beam_size=5, temperature=0
            )
            text = result["text"]

            if text.strip() == "":
                with open(empty_log, "a") as ef:
                    ef.write(clip_id + "\n")

            ref_text = refs.get(clip_id, {}).get("transcript", "")
            wer = jiwer.wer(ref_text, text) if ref_text else float("nan")

            # Save per-clip transcript text
            (output_dir / f"{clip_id}.txt").write_text(text, encoding="utf-8")

            writer.writerow({
                "clip_id":      clip_id,
                "condition":    condition,
                "whisper_text": text,
                "ref_text":     ref_text,
                "wer":          round(wer, 6) if wer == wer else "nan",
            })
            csvfile.flush()
