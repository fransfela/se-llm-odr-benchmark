"""Classify wav2vec2 transcripts via GN APIM gateway."""

import csv
import sys
from pathlib import Path

from src.nlp.classify_intent_gpt import (
    classify_transcripts, get_client, _model_slug,
)

ROOT = Path(__file__).parent.parent.parent
CONDITIONS = ["clean", "noisy", "ns_metricgan", "aec_sim", "dereverb"]


def classify_wav2vec2_transcripts(condition, client=None, model=None):
    transcript_csv = ROOT / "results" / f"transcripts_wav2vec2_{condition}.csv"
    intent_path = ROOT / "data" / "raw" / "slurp" / "intents.txt"
    output_csv = ROOT / "results" / f"intents_wav2vec2_{condition}.csv"

    if not transcript_csv.exists():
        print(f"SKIP {condition}: {transcript_csv.name} not found")
        return

    if client is None or model is None:
        client, model = get_client()

    # Patch: classify_transcripts reads "whisper_text" column —
    # create a temp CSV with that column name mapped
    import pandas as pd
    df = pd.read_csv(transcript_csv)
    df["whisper_text"] = df["wav2vec2_transcript"]
    tmp = transcript_csv.parent / f"_tmp_w2v_{condition}.csv"
    df.to_csv(tmp, index=False)

    classify_transcripts(tmp, intent_path, output_csv,
                         client=client, model=model)
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    cond = "clean"
    if "--condition" in sys.argv:
        cond = sys.argv[sys.argv.index("--condition") + 1]

    client, model = get_client()
    print(f"Model: {model}")

    if cond == "all":
        for c in CONDITIONS:
            classify_wav2vec2_transcripts(c, client, model)
    else:
        classify_wav2vec2_transcripts(cond, client, model)
