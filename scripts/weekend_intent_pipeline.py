"""Weekend intent pipeline — polls ASR completion and auto-starts Gemini intent classification.

Usage:
    python scripts/weekend_intent_pipeline.py

Polls results/transcripts_{condition}.csv for 2974 rows.
When complete, starts Gemini intent classification via GN APIM gateway.
Idempotent — skips already-classified clips via the CSV append logic.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
EXPECTED = 2974
CONDITIONS = ["ns_diffusion", "aec_full"]
POLL_INTERVAL = 300  # 5 minutes


def transcript_count(condition: str) -> int:
    csv = ROOT / "results" / f"transcripts_{condition}.csv"
    if not csv.exists():
        return 0
    lines = csv.read_text().strip().split("\n")
    return max(0, len(lines) - 1)  # subtract header


def intent_done(condition: str) -> bool:
    """Check if intent classification is complete for this condition."""
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    slug = model.replace("/", "_").replace(".", "_")
    csv = ROOT / "results" / f"intents_{slug}_{condition}.csv"
    if not csv.exists():
        return False
    lines = csv.read_text().strip().split("\n")
    return len(lines) - 1 >= EXPECTED


def run_intent(condition: str):
    from src.nlp.classify_intent_gpt import get_client, classify_transcripts, _model_slug

    print(f"\n{'='*50}")
    print(f"Starting intent classification for {condition}")
    print(f"{'='*50}\n")

    client, model = get_client()
    slug = _model_slug(model)

    intent_path = ROOT / "data" / "raw" / "slurp" / "intents.txt"
    transcript_csv = ROOT / "results" / f"transcripts_{condition}.csv"
    output_csv = ROOT / "results" / f"intents_{slug}_{condition}.csv"

    classify_transcripts(transcript_csv, intent_path, output_csv,
                         client=client, model=model)
    print(f"\nDone: {condition}")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    done = set()
    print(f"Weekend intent pipeline — polling {CONDITIONS} for {EXPECTED} transcripts")
    print(f"Poll interval: {POLL_INTERVAL}s")

    while len(done) < len(CONDITIONS):
        for cond in CONDITIONS:
            if cond in done:
                continue

            if intent_done(cond):
                print(f"[{cond}] Intent classification already complete — skipping")
                done.add(cond)
                continue

            n = transcript_count(cond)
            if n >= EXPECTED:
                print(f"[{cond}] ASR complete ({n}/{EXPECTED}) — starting intent classification")
                run_intent(cond)
                done.add(cond)
            else:
                print(f"[{cond}] {n}/{EXPECTED} transcripts — waiting...")

        if len(done) < len(CONDITIONS):
            time.sleep(POLL_INTERVAL)

    print("\nAll conditions classified!")


if __name__ == "__main__":
    main()
