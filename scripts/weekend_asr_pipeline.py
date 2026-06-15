"""Weekend ASR pipeline — polls enhancement dirs and auto-starts Whisper transcription.

Usage:
    python scripts/weekend_asr_pipeline.py

Polls data/enhanced/{ns_diffusion,aec_full} every 5 minutes.
When a condition reaches 2974 WAVs, starts Whisper large-v3 transcription.
Idempotent — skips already-transcribed clips via the CSV append logic.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
EXPECTED = 2974
CONDITIONS = ["ns_diffusion", "aec_full"]
POLL_INTERVAL = 300  # 5 minutes


def count_wavs(condition: str) -> int:
    d = ROOT / "data" / "enhanced" / condition
    if not d.exists():
        return 0
    # Count only base wavs (not _mic/_ref intermediates)
    return len([f for f in d.glob("*.wav")
                if "_mic" not in f.stem and "_ref" not in f.stem])


def transcription_done(condition: str) -> bool:
    csv = ROOT / "results" / f"transcripts_{condition}.csv"
    if not csv.exists():
        return False
    lines = csv.read_text().strip().split("\n")
    return len(lines) - 1 >= EXPECTED  # header + data rows


def run_transcription(condition: str):
    from src.asr.transcribe_whisper import transcribe_condition

    print(f"\n{'='*50}")
    print(f"Starting Whisper transcription for {condition}")
    print(f"{'='*50}\n")

    transcribe_condition(
        condition_dir=ROOT / "data" / "enhanced" / condition,
        output_dir=ROOT / "data" / "processed" / "transcripts" / condition,
        reference_jsonl=ROOT / "data" / "raw" / "slurp" / "test" / "slurp_test.jsonl",
    )
    print(f"\nDone: {condition}")


def main():
    done = set()
    print(f"Weekend ASR pipeline — polling {CONDITIONS} for {EXPECTED} WAVs")
    print(f"Poll interval: {POLL_INTERVAL}s")

    while len(done) < len(CONDITIONS):
        for cond in CONDITIONS:
            if cond in done:
                continue

            if transcription_done(cond):
                print(f"[{cond}] Transcription already complete — skipping")
                done.add(cond)
                continue

            n = count_wavs(cond)
            if n >= EXPECTED:
                print(f"[{cond}] Enhancement complete ({n}/{EXPECTED}) — starting ASR")
                run_transcription(cond)
                done.add(cond)
            else:
                print(f"[{cond}] {n}/{EXPECTED} WAVs — waiting...")

        if len(done) < len(CONDITIONS):
            time.sleep(POLL_INTERVAL)

    print("\nAll conditions transcribed!")


if __name__ == "__main__":
    main()
