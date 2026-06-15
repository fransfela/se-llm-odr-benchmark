"""Verify SLURP reference label loading and intent list consistency."""

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SLURP_JSONL  = ROOT / "data" / "raw" / "slurp" / "test" / "slurp_test.jsonl"
INTENTS_TXT  = ROOT / "data" / "raw" / "slurp" / "intents.txt"


def test_slurp_references_load():
    if not SLURP_JSONL.exists():
        pytest.skip("slurp_test.jsonl not present yet")

    from src.data.slurp_utils import load_slurp_references
    refs = load_slurp_references(SLURP_JSONL)

    assert len(refs) > 0, "No entries loaded from slurp_test.jsonl"

    required_keys = {"transcript", "intent", "domain", "action"}
    for clip_id, entry in refs.items():
        missing = required_keys - entry.keys()
        assert not missing, f"clip_id={clip_id} missing keys: {missing}"
        assert entry["transcript"].strip() != "", f"clip_id={clip_id} has empty transcript"
        assert entry["intent"].strip()     != "", f"clip_id={clip_id} has empty intent"


def test_intents_txt_matches_jsonl():
    if not SLURP_JSONL.exists():
        pytest.skip("slurp_test.jsonl not present yet")
    if not INTENTS_TXT.exists():
        pytest.skip("intents.txt not generated yet — run load_intent_list first")

    from src.data.slurp_utils import load_intent_list

    intents_jsonl = set(load_intent_list(SLURP_JSONL))

    with open(INTENTS_TXT) as f:
        intents_txt = {line.strip() for line in f if line.strip()}

    assert intents_txt == intents_jsonl, (
        f"Mismatch:\n  only in txt:   {intents_txt - intents_jsonl}\n"
        f"  only in jsonl: {intents_jsonl - intents_txt}"
    )
    assert len(intents_txt) > 50, (
        f"Only {len(intents_txt)} intents found — expected ~90 for SLURP"
    )
