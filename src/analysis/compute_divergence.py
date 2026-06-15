"""Compute ODR, wODR, and WER-ODR gap per enhancement condition."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util as st_util

ROOT = Path(__file__).parent.parent.parent
NON_CLEAN = ["noisy", "ns_metricgan", "ns_diffusion", "dereverb", "aec_sim", "aec_full", "ns_aec_combined"]

np.random.seed(42)
random.seed(42)

_embed_model = None


def _get_embed():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _bootstrap_ci(arr: np.ndarray, n: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, (n, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _mean_wer(condition: str) -> float:
    p = ROOT / "results" / f"transcripts_{condition}.csv"
    if not p.exists():
        return float("nan")
    return float(pd.read_csv(p)["wer"].replace("nan", np.nan).dropna().astype(float).mean())


def compute_odr(intents_csv: str | Path) -> pd.DataFrame:
    """Compute per-condition ODR, wODR, and WER-ODR gap. Saves two CSVs."""
    raw = pd.read_csv(Path(intents_csv))
    # normalise is_valid to bool regardless of CSV storage format
    raw["is_valid"] = raw["is_valid"].astype(str).str.lower().isin(["true", "1"])
    valid = raw[raw["is_valid"]].copy()

    clean = (valid[valid["condition"] == "clean"][["clip_id", "predicted_intent"]]
             .rename(columns={"predicted_intent": "intent_clean"}))

    model = _get_embed()
    embs = {lbl: model.encode(lbl, convert_to_tensor=True)
            for lbl in valid["predicted_intent"].unique()}

    rows, clip_rows = [], []
    for cond in NON_CLEAN:
        sub = valid[valid["condition"] == cond].merge(clean, on="clip_id", how="inner")
        if sub.empty:
            continue
        diverged = (sub["predicted_intent"] != sub["intent_clean"]).astype(int).values
        odr = diverged.mean()
        ci_lo, ci_hi = _bootstrap_ci(diverged)

        div_sub = sub[diverged.astype(bool)]
        if len(div_sub):
            sims = [float(st_util.cos_sim(embs[r.predicted_intent], embs[r.intent_clean]))
                    for r in div_sub.itertuples()]
            wodr = float(np.mean([1 - s for s in sims]))
        else:
            wodr = 0.0

        for r in sub.itertuples():
            div = int(r.predicted_intent != r.intent_clean)
            sem = float(1 - st_util.cos_sim(embs[r.predicted_intent], embs[r.intent_clean])) if div else 0.0
            clip_rows.append({"clip_id": r.clip_id, "condition": cond,
                               "intent_clean": r.intent_clean, "intent_condition": r.predicted_intent,
                               "diverged": div, "semantic_distance": round(sem, 6)})

        mwer = _mean_wer(cond)
        gap = float(odr) - min(float(mwer), 1.0) if not np.isnan(mwer) else np.nan
        inv = float((raw[raw["condition"] == cond]["is_valid"] == False).mean())  # noqa: E712
        rows.append({"condition": cond, "odr": round(float(odr), 6),
                     "odr_ci_low": round(ci_lo, 6), "odr_ci_high": round(ci_hi, 6),
                     "wodr": round(wodr, 6),
                     "mean_wer": round(float(mwer), 6) if not np.isnan(mwer) else np.nan,
                     "wer_odr_gap": round(float(gap), 6) if not np.isnan(gap) else np.nan,
                     "n_clips": len(sub), "n_divergent": int(diverged.sum()),
                     "invalid_rate": round(inv, 6)})

    result  = pd.DataFrame(rows)
    clip_df = pd.DataFrame(clip_rows)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    result.to_csv(ROOT / "results" / "divergence.csv", index=False)
    clip_df.to_csv(ROOT / "results" / "clip_level_divergence.csv", index=False)
    return result


if __name__ == "__main__":
    df = compute_odr(ROOT / "results" / "intents_gpt.csv")
    parts = ["clean=0.000"] + [f"{r.condition}={r.odr:.3f}" for r in df.itertuples()]
    print(" | ".join(parts))
