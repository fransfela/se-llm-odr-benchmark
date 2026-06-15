"""Master runner for the wav2vec2 ASR + intent classification pipeline."""

import subprocess
import sys
from pathlib import Path

CONDITIONS = ["clean", "noisy", "ns_metricgan", "aec_sim", "dereverb"]


def run(cmd, desc):
    print(f"\n>>> {desc}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)


def check_asr_done(condition):
    p = Path(f"results/transcripts_wav2vec2_{condition}.csv")
    if not p.exists():
        return False
    import pandas as pd
    return len(pd.read_csv(p)) >= 2900


def check_intents_done(condition):
    p = Path(f"results/intents_wav2vec2_{condition}.csv")
    if not p.exists():
        return False
    import pandas as pd
    return len(pd.read_csv(p)) >= 2900


if __name__ == "__main__":
    print("=== wav2vec2 pipeline ===")
    print("Step 1: ASR transcription (all 5 conditions)")
    for c in CONDITIONS:
        if check_asr_done(c):
            print(f"  SKIP {c} (already done)")
            continue
        run(f"python -m src.asr.transcribe_wav2vec2 "
            f"--condition {c}", f"ASR: {c}")

    print("\nStep 2: Intent classification (all 5 conditions)")
    for c in CONDITIONS:
        if check_intents_done(c):
            print(f"  SKIP {c} (already done)")
            continue
        run(f"python -m src.nlp.classify_intent_wav2vec2 "
            f"--condition {c}", f"Intents: {c}")

    print("\nStep 3: ODR analysis")
    run("python scripts/compute_odr_wav2vec2.py",
        "Compute wav2vec2 ODR")

    print("\nDone. Results: results/divergence_wav2vec2.csv")
