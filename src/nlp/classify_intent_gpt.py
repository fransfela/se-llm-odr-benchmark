"""Classify whisper transcripts via OpenAI-compatible API against the SLURP intent list.

Supports Gemini, GPT, and on-prem models via OpenAI-compatible API.
Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI, OpenAIError
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
ERROR_LOG = ROOT / "results" / "logs" / "gpt_intent_errors.jsonl"

FIELDNAMES = ["clip_id", "condition", "model_name", "predicted_intent",
              "is_valid", "prompt_tokens", "completion_tokens"]

SYSTEM_PROMPT = (
    "You are a spoken language understanding classifier. "
    "Output ONLY one intent label from the list. No punctuation. No explanation."
)

CONDITIONS = ["clean", "noisy", "ns_metricgan", "aec_sim", "dereverb"]


def _model_slug(model_name):
    return model_name.replace("/", "_").replace(".", "_")


def get_client():
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise ValueError("LLM_API_KEY not set in .env")
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    base_url = os.getenv("LLM_BASE_URL", "")

    if "qwen" in model.lower():
        client = OpenAI(
            api_key=key,
            base_url=base_url or "http://localhost:8000/v1",
            default_headers={"api-key": key},
            http_client=httpx.Client(verify=False),
        )
    elif "gemini" in model.lower():
        # Gemini OpenAI-compatible route uses /gemini base and google/ prefix
        api_model = model if model.startswith("google/") else f"google/{model}"
        client = OpenAI(
            api_key=key,
            base_url=base_url or "https://generativelanguage.googleapis.com/v1beta",
            default_headers={"api-key": key},
        )
        model = api_model
    else:
        client = OpenAI(
            api_key=key,
            base_url=base_url or "https://api.openai.com/v1",
            default_headers={"api-key": key},
        )
    return client, model


THINKING_MODELS = {"google/gemini-2.5-pro", "google/gemini-2.5-flash"}


def make_completion(client, model, messages, retries=3):
    for attempt in range(retries):
        try:
            if model.startswith("gpt-5"):
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    extra_body={"max_completion_tokens": 15},
                )
            elif model in THINKING_MODELS:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=500,
                )
            else:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=15,
                )
        except OpenAIError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _build_user_prompt(whisper_text, intents):
    return (
        f"Utterance: {whisper_text}\n"
        f"Valid intents: {', '.join(intents)}\n"
        "Intent:"
    )


def _load_existing(output_csv):
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()
    import pandas as pd
    df = pd.read_csv(output_csv)
    # Use only clip_id (each file is condition-specific).
    # Cast to str for consistent comparison with csv.DictReader rows.
    return set(df["clip_id"].dropna().astype(str).tolist())


def classify_transcripts(transcript_csv, intent_list_path, output_csv,
                         client=None, model=None):
    transcript_csv = Path(transcript_csv)
    intent_list_path = Path(intent_list_path)
    output_csv = Path(output_csv)

    with open(intent_list_path) as f:
        intent_list = [line.strip() for line in f if line.strip()]
    intent_set = set(intent_list)

    with open(transcript_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if client is None or model is None:
        client, model = get_client()

    existing = _load_existing(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not output_csv.exists() or output_csv.stat().st_size == 0
    total_tokens = 0

    with open(output_csv, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()

        for row in tqdm(rows, desc=f"intent ({model})"):
            clip_id = row["clip_id"]
            condition = row["condition"]
            if str(clip_id) in existing:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(row["whisper_text"], intent_list)},
            ]

            try:
                resp = make_completion(client, model, messages)
                msg = resp.choices[0].message
                label = msg.content.strip() if msg and msg.content else "INVALID"
                p_tok = resp.usage.prompt_tokens
                c_tok = resp.usage.completion_tokens
                total_tokens += p_tok + c_tok
            except Exception as e:
                ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(ERROR_LOG, "a") as el:
                    el.write(json.dumps({"clip_id": clip_id, "condition": condition,
                                         "error": str(e)}) + "\n")
                label, p_tok, c_tok = "INVALID", 0, 0

            if label not in intent_set:
                label = "INVALID"

            writer.writerow({
                "clip_id": clip_id,
                "condition": condition,
                "model_name": model,
                "predicted_intent": label,
                "is_valid": label != "INVALID",
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
            })
            existing.add(str(clip_id))
            out_f.flush()
            time.sleep(60 / 400)

    print(f"Done. {total_tokens} tokens used.")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    conds = CONDITIONS
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--condition" and len(sys.argv) > 2:
            arg = sys.argv[2]
        if arg != "all":
            conds = [arg]

    client, model = get_client()
    slug = _model_slug(model)
    print(f"Model: {model}  slug: {slug}")

    intent_path = ROOT / "data" / "raw" / "slurp" / "intents.txt"
    if not intent_path.exists():
        from src.data.slurp_utils import load_intent_list
        load_intent_list(ROOT / "data" / "raw" / "slurp" / "test" / "slurp_test.jsonl")

    for cond in conds:
        transcript_csv = ROOT / "results" / f"transcripts_{cond}.csv"
        if not transcript_csv.exists():
            print(f"SKIP {cond}: {transcript_csv.name} not found")
            continue
        output_csv = ROOT / "results" / f"intents_{slug}_{cond}.csv"
        print(f"\n{'='*40}\n{cond} -> {output_csv.name}\n{'='*40}")
        classify_transcripts(transcript_csv, intent_path, output_csv,
                             client=client, model=model)


if __name__ == "__main__":
    main()
