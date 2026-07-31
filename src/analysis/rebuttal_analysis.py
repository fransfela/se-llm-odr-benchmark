"""Rebuttal analyses (July 2026 review cycle):
1. correction_vs_regression: does SE fix or break LLM intent predictions relative to ground truth?
2. metric_failure_analysis: quantify PESQ false positives/negatives against ODR per condition.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
CONDITIONS = ["noisy", "ns_metricgan", "aec_sim", "aec_full", "dereverb"]


def _load_ground_truth() -> pd.DataFrame:
    rows = []
    with open(ROOT / "data" / "raw" / "slurp" / "test" / "slurp_test.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"clip_id": obj["id"], "intent_truth": obj["intent"]})
    df = pd.DataFrame(rows)
    df["clip_id"] = df["clip_id"].astype(str)
    return df.drop_duplicates(subset="clip_id", keep="first")


def _load_intents(condition: str) -> pd.DataFrame:
    p = ROOT / "results" / f"intents_google_gemini-2_5-flash-lite_{condition}.csv"
    df = pd.read_csv(p)
    df["is_valid"] = df["is_valid"].astype(str).str.lower().isin(["true", "1"])
    df = df[df["is_valid"]].copy()
    df["clip_id"] = df["clip_id"].astype(str)
    return df[["clip_id", "predicted_intent"]]


def correction_vs_regression() -> pd.DataFrame:
    truth = _load_ground_truth()
    clean = _load_intents("clean").rename(columns={"predicted_intent": "intent_clean"})
    rows = []
    for cond in CONDITIONS:
        cond_df = _load_intents(cond).rename(columns={"predicted_intent": "intent_condition"})
        merged = truth.merge(clean, on="clip_id", how="inner").merge(cond_df, on="clip_id", how="inner")
        n = len(merged)
        clean_correct = merged["intent_clean"] == merged["intent_truth"]
        cond_correct = merged["intent_condition"] == merged["intent_truth"]
        correction = int((~clean_correct & cond_correct).sum())
        regression = int((clean_correct & ~cond_correct).sum())
        both_wrong_diff = int((~clean_correct & ~cond_correct &
                               (merged["intent_clean"] != merged["intent_condition"])).sum())
        unaffected = n - correction - regression - both_wrong_diff
        rows.append({
            "condition": cond, "n": n,
            "correction_n": correction, "correction_pct": round(correction / n, 4),
            "regression_n": regression, "regression_pct": round(regression / n, 4),
            "both_wrong_diff_n": both_wrong_diff, "both_wrong_diff_pct": round(both_wrong_diff / n, 4),
            "unaffected_n": unaffected, "unaffected_pct": round(unaffected / n, 4),
            "harm_benefit_ratio": round(regression / correction, 2) if correction else float("inf"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "correction_vs_regression.csv", index=False)
    return out


def metric_failure_analysis() -> pd.DataFrame:
    metrics = pd.read_csv(ROOT / "results" / "metrics_core.csv")
    metrics["clip_id"] = metrics["clip_id"].astype(str)
    div = pd.read_csv(ROOT / "results" / "clip_level_divergence.csv")
    div["clip_id"] = div["clip_id"].astype(str)

    rows = []
    for cond in CONDITIONS:
        m = metrics[metrics["condition"] == cond][["clip_id", "pesq"]]
        d = div[div["condition"] == cond][["clip_id", "diverged"]]
        merged = m.merge(d, on="clip_id", how="inner").dropna(subset=["pesq"])
        median_pesq = merged["pesq"].median()
        good_quality = merged["pesq"] >= median_pesq
        diverged = merged["diverged"] == 1
        good_but_diverged = int((good_quality & diverged).sum())
        poor_but_not_diverged = int((~good_quality & ~diverged).sum())
        n = len(merged)
        n_above = int(good_quality.sum())   # actual above-median count (accounts for ties)
        n_below = n - n_above
        # FN rate: P(diverge | PESQ >= median) — PESQ misses real damage
        fn_rate = round(good_but_diverged / n_above, 4) if n_above > 0 else float("nan")
        # FP rate: P(not diverge | PESQ < median) — PESQ over-warns on safe clips
        fp_rate = round(poor_but_not_diverged / n_below, 4) if n_below > 0 else float("nan")
        rows.append({
            "condition": cond, "n": n, "median_pesq": round(median_pesq, 3),
            "n_above_median": n_above, "n_below_median": n_below,
            "good_quality_but_diverged_n": good_but_diverged,
            "fn_rate": fn_rate,   # conditional: / n_above
            "poor_quality_but_not_diverged_n": poor_but_not_diverged,
            "fp_rate": fp_rate,   # conditional: / n_below
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "metric_failure_analysis.csv", index=False)
    return out


if __name__ == "__main__":
    print(correction_vs_regression().to_string(index=False))
    print()
    print(metric_failure_analysis().to_string(index=False))
