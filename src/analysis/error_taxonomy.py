"""Classify divergent clips into error taxonomy categories."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
ERROR_TYPES = ["domain_shift", "action_confusion", "complete_failure", "other"]


def _load_intent_meta(intents_jsonl: Path) -> dict[str, dict]:
    """Return {intent_label: {domain, action}} from slurp_test.jsonl."""
    meta: dict[str, dict] = {}
    with open(intents_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta[obj["intent"]] = {"domain": obj["scenario"], "action": obj["action"]}
    return meta


def classify_errors(
    clip_divergence_csv: str | Path,
    intents_jsonl: str | Path,
) -> pd.DataFrame:
    """Tag each divergent clip with one of: domain_shift, action_confusion, complete_failure, other.

    Note: negation_flip and entity_loss require entity-level NER — tagged as 'other' for now
    (future work).
    """
    clip_df = pd.read_csv(Path(clip_divergence_csv))
    diverged = clip_df[clip_df["diverged"] == 1].copy()
    meta = _load_intent_meta(Path(intents_jsonl))

    types = []
    for row in diverged.itertuples():
        ic = row.intent_clean
        io = row.intent_condition

        if str(io).upper() == "INVALID":
            types.append("complete_failure")
            continue

        dom_c = meta.get(ic, {}).get("domain")
        dom_o = meta.get(io, {}).get("domain")
        act_c = meta.get(ic, {}).get("action")
        act_o = meta.get(io, {}).get("action")

        if dom_c and dom_o and dom_c != dom_o:
            types.append("domain_shift")
        elif dom_c == dom_o and act_c and act_o and act_c != act_o:
            types.append("action_confusion")
        else:
            types.append("other")

    diverged = diverged.copy()
    diverged["error_type"] = types

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    diverged[["clip_id", "condition", "error_type"]].to_csv(
        out / "error_taxonomy.csv", index=False)

    # Condition-level summary
    summary_rows = []
    for cond in diverged["condition"].unique():
        sub = diverged[diverged["condition"] == cond]
        row = {"condition": cond, "total_divergent": len(sub)}
        for et in ERROR_TYPES:
            row[f"{et}_n"] = int((sub["error_type"] == et).sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "error_taxonomy_summary.csv", index=False)
    return summary
