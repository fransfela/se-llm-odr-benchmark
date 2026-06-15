"""Classify whisper transcripts with Llama-3-70B-Instruct (local inference)."""

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
MODEL_VERSIONS = ROOT / "results" / "model_versions.txt"
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"

FIELDNAMES = ["clip_id", "condition", "predicted_intent", "is_valid", "inference_time_s"]

SYSTEM_PROMPT = (
    "You are a spoken language understanding classifier. "
    "Output ONLY one intent label from the list. No punctuation. No explanation."
)

# ── Singleton ────────────────────────────────────────────────────────────────
_pipeline = None


def _log_model_version(name: str, version_str: str) -> None:
    MODEL_VERSIONS.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(MODEL_VERSIONS, "a") as f:
        f.write(f"{ts} | {name} | {version_str}\n")


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import transformers
        _pipeline = transformers.pipeline(
            "text-generation",
            model=MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _log_model_version(
            "llama3-70b",
            f"{MODEL_ID} | transformers=={transformers.__version__}",
        )
    return _pipeline


# ── Helpers ─────────────────────────────────────────────────────────────────
def _build_user_prompt(whisper_text: str, intents: list[str]) -> str:
    return (
        f"Utterance: {whisper_text}\n"
        f"Valid intents: {', '.join(intents)}\n"
        "Intent:"
    )


def _load_existing(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()
    with open(output_csv, newline="") as f:
        return {(r["clip_id"], r["condition"]) for r in csv.DictReader(f)}


# ── Main function ────────────────────────────────────────────────────────────
def classify_transcripts_llama(
    transcript_csv: str | Path,
    intent_list_path: str | Path,
    output_csv: str | Path,
) -> None:
    """Classify each row in transcript_csv with Llama-3-70B and append to output_csv."""
    transcript_csv   = Path(transcript_csv)
    intent_list_path = Path(intent_list_path)
    output_csv       = Path(output_csv)

    with open(intent_list_path) as f:
        intent_list = [line.strip() for line in f if line.strip()]
    intent_set = set(intent_list)

    with open(transcript_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    existing  = _load_existing(output_csv)
    write_hdr = not output_csv.exists()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pipe = _get_pipeline()

    with open(output_csv, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()

        for row in tqdm(rows, desc="llama-intent"):
            clip_id   = row["clip_id"]
            condition = row["condition"]
            if (clip_id, condition) in existing:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(
                    row["whisper_text"], intent_list)},
            ]

            t0 = time.perf_counter()
            output = pipe(
                messages,
                max_new_tokens=15,
                temperature=0.0,
                do_sample=False,
            )
            inference_time_s = time.perf_counter() - t0

            raw = output[0]["generated_text"]
            # Extract only the assistant's new tokens (last turn)
            if isinstance(raw, list):
                label = raw[-1].get("content", "").strip()
            else:
                label = str(raw).strip()
            label = label.splitlines()[0].strip()

            if label not in intent_set:
                label = "INVALID"

            writer.writerow({
                "clip_id":          clip_id,
                "condition":        condition,
                "predicted_intent": label,
                "is_valid":         label != "INVALID",
                "inference_time_s": round(inference_time_s, 4),
            })
            out_f.flush()
