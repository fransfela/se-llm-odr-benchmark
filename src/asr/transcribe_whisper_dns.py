"""Transcribe DNS blind test clips with Whisper large-v3.

WER is computed relative to the Whisper transcript of the corresponding clean clip
(no external ground-truth transcripts exist for the DNS blind test).
"""

import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import numpy as np
import torch
import whisper
from tqdm import tqdm

from src.data.dns_utils import load_dns_clip_pairs, DNS_DIR

ROOT           = Path(__file__).parent.parent.parent
MODEL_VERSIONS = ROOT / "results" / "model_versions.txt"
ERROR_LOG      = ROOT / "results" / "logs" / "whisper_dns_errors.jsonl"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

FIELDNAMES = ["clip_id", "condition_name", "whisper_text", "clean_reference_text", "wer_relative"]


def _log_model_version(name: str, version_str: str) -> None:
    MODEL_VERSIONS.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(MODEL_VERSIONS, "a") as f:
        f.write(f"{ts} | {name} | {version_str}\n")


def _existing_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(r["clip_id"], r["condition_name"]) for r in csv.DictReader(f)}


def transcribe_dns_condition(
    condition_name: str,
    dns_blind_dir: str | Path = DNS_DIR,
    output_dir: str | Path | None = None,
) -> None:
    """Transcribe one DNS condition (e.g. 'dns_noisy' or a system name) with Whisper."""
    dns_blind_dir = Path(dns_blind_dir)
    if output_dir is None:
        output_dir = ROOT / "data" / "processed" / "transcripts" / "whisper" / f"dns_{condition_name}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)

    csv_path = ROOT / "results" / f"transcripts_dns_{condition_name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_keys(csv_path)

    model = whisper.load_model("large-v3")
    _log_model_version("whisper", "openai/whisper large-v3 (dns)")

    clip_pairs = load_dns_clip_pairs(dns_blind_dir)

    # Pre-transcribe all clean clips once (cached to text files)
    clean_txt_dir = ROOT / "data" / "processed" / "transcripts" / "whisper" / "dns_clean"
    clean_txt_dir.mkdir(parents=True, exist_ok=True)
    clean_refs: dict[str, str] = {}
    for pair in clip_pairs:
        cid = pair["clip_id"]
        txt_file = clean_txt_dir / f"{cid}.txt"
        if txt_file.exists():
            clean_refs[cid] = txt_file.read_text(encoding="utf-8").strip()
        elif pair["clean_path"]:
            try:
                res = model.transcribe(str(pair["clean_path"]), language="en",
                                       beam_size=5, temperature=0)
                text = res["text"].strip()
                txt_file.write_text(text, encoding="utf-8")
                clean_refs[cid] = text
            except Exception as exc:
                import json
                with open(ERROR_LOG, "a") as f:
                    f.write(json.dumps({"clip_id": cid, "condition": "dns_clean",
                                        "stage": "whisper", "error": str(exc)}) + "\n")

    write_hdr = not csv_path.exists()
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()

        for pair in tqdm(clip_pairs, desc=f"whisper dns_{condition_name}"):
            cid = pair["clip_id"]
            if (cid, condition_name) in existing:
                continue

            # resolve audio path for this condition
            if condition_name == "dns_noisy":
                wav_path = pair["noisy_path"]
            else:
                sys_key  = condition_name.removeprefix("dns_")
                wav_path = pair["enhanced_paths"].get(sys_key)

            if wav_path is None or not Path(wav_path).exists():
                continue

            try:
                res  = model.transcribe(str(wav_path), language="en", beam_size=5, temperature=0)
                text = res["text"].strip()
            except Exception as exc:
                import json
                with open(ERROR_LOG, "a") as f:
                    f.write(json.dumps({"clip_id": cid, "condition": condition_name,
                                        "stage": "whisper", "error": str(exc)}) + "\n")
                continue

            (output_dir / f"{cid}.txt").write_text(text, encoding="utf-8")
            clean_ref = clean_refs.get(cid, "")
            wer_rel   = jiwer.wer(clean_ref, text) if clean_ref else float("nan")

            writer.writerow({
                "clip_id":              cid,
                "condition_name":       condition_name,
                "whisper_text":         text,
                "clean_reference_text": clean_ref,
                "wer_relative":         wer_rel,
            })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="dns_noisy",
                        help="Condition name, e.g. dns_noisy or dns_DTLN")
    parser.add_argument("--dns_dir", default=str(DNS_DIR))
    args = parser.parse_args()
    transcribe_dns_condition(args.condition, dns_blind_dir=args.dns_dir)
