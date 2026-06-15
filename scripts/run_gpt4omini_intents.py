"""Run Gemini 2.5 Pro intent classification on all 6 conditions.

This serves as a cross-model robustness check against the primary
Gemini 2.5 Flash Lite results.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import os
os.environ["LLM_MODEL"] = "gemini-2.5-pro"

from src.nlp.classify_intent_gpt import classify_transcripts, get_client, _model_slug

CONDITIONS = ["clean", "noisy", "ns_metricgan", "aec_sim", "aec_full", "dereverb"]

def main():
    client, model = get_client()
    slug = _model_slug(model)
    print(f"Model: {model}  slug: {slug}")

    intent_path = ROOT / "data" / "raw" / "slurp" / "intents.txt"

    for cond in CONDITIONS:
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
