"""Classify GPT-4o sentiment (positive / neutral / negative) for DNS transcript CSVs."""

import csv
import json
import time
from pathlib import Path

import openai
from tqdm import tqdm

ROOT      = Path(__file__).parent.parent.parent
ERROR_LOG = ROOT / "results" / "logs" / "gpt_sentiment_errors.jsonl"

FIELDNAMES = ["clip_id", "condition_name", "predicted_sentiment",
              "prompt_tokens", "completion_tokens"]

SYSTEM_PROMPT = (
    "You are a sentiment classifier. Classify the sentiment of this spoken utterance. "
    "Output ONLY one of: positive, neutral, negative. No punctuation. No explanation."
)

VALID_SENTIMENTS = {"positive", "neutral", "negative"}


def _load_existing(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()
    with open(output_csv, newline="") as f:
        return {(r["clip_id"], r["condition_name"]) for r in csv.DictReader(f)}


def _call_gpt(client: openai.OpenAI, text: str, retries: int = 3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Utterance: {text}"},
    ]
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                temperature=0,
                max_tokens=10,
                messages=messages,
            )
        except openai.OpenAIError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def classify_sentiment(
    transcript_csv: str | Path,
    output_csv: str | Path,
    condition_name: str | None = None,
) -> None:
    """Classify sentiment for every row in transcript_csv; append to output_csv.

    transcript_csv must have columns: clip_id, condition_name (or override via param),
    whisper_text.
    """
    transcript_csv = Path(transcript_csv)
    output_csv     = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)

    existing  = _load_existing(output_csv)
    write_hdr = not output_csv.exists()
    client    = openai.OpenAI()

    with open(transcript_csv, newline="") as src, \
         open(output_csv, "a", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()

        rows = list(reader)
        for row in tqdm(rows, desc=f"sentiment {output_csv.stem}"):
            cid   = row["clip_id"].strip()
            cname = condition_name or row.get("condition_name", "").strip()
            if (cid, cname) in existing:
                continue

            text = row.get("whisper_text", "").strip()
            if not text:
                with open(ERROR_LOG, "a") as ef:
                    ef.write(json.dumps({"clip_id": cid, "condition": cname,
                                         "error": "empty transcript"}) + "\n")
                continue

            try:
                resp      = _call_gpt(client, text)
                predicted = resp.choices[0].message.content.strip().lower()
                pt        = resp.usage.prompt_tokens
                ct        = resp.usage.completion_tokens
            except Exception as exc:
                with open(ERROR_LOG, "a") as ef:
                    ef.write(json.dumps({"clip_id": cid, "condition": cname,
                                         "error": str(exc)}) + "\n")
                continue

            if predicted not in VALID_SENTIMENTS:
                with open(ERROR_LOG, "a") as ef:
                    ef.write(json.dumps({"clip_id": cid, "condition": cname,
                                         "error": f"invalid sentiment: {predicted!r}"}) + "\n")
                # still write so it shows in analysis as INVALID equivalent
                predicted = "INVALID"

            writer.writerow({
                "clip_id":            cid,
                "condition_name":     cname,
                "predicted_sentiment": predicted,
                "prompt_tokens":      pt,
                "completion_tokens":  ct,
            })
            existing.add((cid, cname))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_csv", required=True)
    parser.add_argument("--output_csv",     required=True)
    parser.add_argument("--condition_name", default=None)
    args = parser.parse_args()
    classify_sentiment(args.transcript_csv, args.output_csv, args.condition_name)
