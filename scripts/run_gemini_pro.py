"""Resume Gemini 2.5 Pro intent classification.

Usage:
    python scripts/run_gemini_pro.py YOUR_API_KEY
    python scripts/run_gemini_pro.py YOUR_API_KEY --condition noisy
    python scripts/run_gemini_pro.py YOUR_API_KEY --rerun-invalid noisy

The script sets LLM_API_KEY and LLM_MODEL, then calls the existing
classify_intent_gpt.py pipeline.

--rerun-invalid COND:
  Removes rows where is_valid==False from the output CSV for COND,
  then re-runs classification for those clips only.
  Use this to fix the noisy/ns_metricgan runs that used max_tokens=15.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def rerun_invalid(cond, slug='google_gemini-2_5-pro'):
    """Remove invalid rows from output CSV so the resume logic re-processes them."""
    import pandas as pd
    csv_path = ROOT / 'results' / f'intents_{slug}_{cond}.csv'
    if not csv_path.exists():
        print(f'{csv_path.name} not found — nothing to clean')
        return
    df = pd.read_csv(csv_path)
    n_before = len(df)
    n_invalid = (df.is_valid == False).sum()
    df_valid = df[df.is_valid == True].copy()
    df_valid.to_csv(csv_path, index=False)
    print(f'{cond}: removed {n_invalid} invalid rows '
          f'({n_before} -> {len(df_valid)}), will re-classify them')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    api_key = sys.argv[1]
    os.environ['LLM_API_KEY'] = api_key
    os.environ['LLM_MODEL'] = 'gemini-2.5-pro'

    # Parse optional args
    conditions = None
    rerun_cond = None
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--condition' and i + 1 < len(sys.argv):
            conditions = [sys.argv[i + 1]]
            i += 2
        elif sys.argv[i] == '--rerun-invalid' and i + 1 < len(sys.argv):
            rerun_cond = sys.argv[i + 1]
            i += 2
        else:
            conditions = [sys.argv[i]]
            i += 1

    if rerun_cond:
        rerun_invalid(rerun_cond)
        conditions = [rerun_cond]

    # Default: all conditions that need work
    if conditions is None:
        conditions = ['noisy', 'ns_metricgan', 'aec_sim', 'dereverb', 'aec_full']

    # Import and run the existing pipeline
    sys.path.insert(0, str(ROOT))
    from src.nlp.classify_intent_gpt import (
        get_client, classify_transcripts, _model_slug, ROOT as SRC_ROOT
    )

    client, model = get_client()
    slug = _model_slug(model)
    print(f'Model: {model}  slug: {slug}')
    print(f'Conditions: {conditions}')
    print(f'max_tokens: 500 (thinking model)')
    print()

    intent_path = SRC_ROOT / 'data' / 'raw' / 'slurp' / 'intents.txt'

    for cond in conditions:
        transcript_csv = SRC_ROOT / 'results' / f'transcripts_{cond}.csv'
        if not transcript_csv.exists():
            print(f'SKIP {cond}: {transcript_csv.name} not found')
            continue
        output_csv = SRC_ROOT / 'results' / f'intents_{slug}_{cond}.csv'
        print(f"\n{'='*40}")
        print(f'{cond} -> {output_csv.name}')
        print(f'{"="*40}')
        classify_transcripts(transcript_csv, intent_path, output_csv,
                             client=client, model=model)


if __name__ == '__main__':
    main()
