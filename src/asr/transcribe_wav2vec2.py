"""Transcribe WAV files with wav2vec2-large CTC model."""

import csv
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.data.slurp_utils import load_slurp_references

ROOT = Path(__file__).parent.parent.parent
MODEL_VERSIONS = ROOT / "results" / "model_versions.txt"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
MODEL_VERSIONS.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_VERSIONS, "a") as f:
    f.write(f"{ts} | wav2vec2 | {MODEL_ID}\n")

FIELDNAMES = ["clip_id", "condition", "wav2vec2_transcript", "ref_text", "wer"]


def transcribe_condition_wav2vec2(condition, output_csv=None):
    wav_dir = ROOT / "data" / "enhanced" / condition
    ref_jsonl = ROOT / "data" / "raw" / "slurp" / "test" / "slurp_test.jsonl"
    refs = load_slurp_references(ref_jsonl)

    if output_csv is None:
        output_csv = ROOT / "results" / f"transcripts_wav2vec2_{condition}.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if output_csv.exists():
        with open(output_csv, newline="") as f:
            existing = {r["clip_id"] for r in csv.DictReader(f)}

    empty_log = ROOT / "results" / "logs" / "wav2vec2_empty.txt"
    empty_log.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(
        f for f in wav_dir.glob("*.wav")
        if "_mic" not in f.stem and "_ref" not in f.stem
    )
    write_hdr = not output_csv.exists()

    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()

        for wav_path in tqdm(wavs, desc=f"wav2vec2 {condition}"):
            clip_id = wav_path.stem
            if clip_id in existing:
                continue

            audio, sr = sf.read(wav_path, dtype="float32")
            assert sr == 16000

            inputs = processor(audio, sampling_rate=16000,
                               return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = processor.batch_decode(predicted_ids)[0].lower()

            if not transcript.strip():
                with open(empty_log, "a") as ef:
                    ef.write(f"{clip_id}\n")

            ref_text = refs.get(clip_id, {}).get("transcript", "")
            wer = jiwer.wer(ref_text.lower(), transcript) if ref_text else float("nan")

            writer.writerow({
                "clip_id": clip_id,
                "condition": condition,
                "wav2vec2_transcript": transcript,
                "ref_text": ref_text,
                "wer": round(wer, 6) if wer == wer else "nan",
            })
            csvfile.flush()


if __name__ == "__main__":
    cond = "clean"
    if "--condition" in sys.argv:
        cond = sys.argv[sys.argv.index("--condition") + 1]
    transcribe_condition_wav2vec2(cond)
