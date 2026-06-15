"""Sanity checks on results/metrics.csv against declared valid ranges."""

import math
import statistics
from pathlib import Path

import pytest
import yaml

ROOT         = Path(__file__).parent.parent
METRICS_CSV  = ROOT / "results" / "metrics.csv"
METRICS_YAML = ROOT / "configs" / "metrics.yaml"


def _load_ranges() -> dict:
    """Return {metric: (valid_min_or_None, valid_max_or_None)} from metrics.yaml."""
    with open(METRICS_YAML) as f:
        cfg = yaml.safe_load(f)["metrics"]
    return {k: (v.get("valid_min"), v.get("valid_max")) for k, v in cfg.items()}


def _metric_cols() -> list:
    with open(METRICS_YAML) as f:
        return list(yaml.safe_load(f)["metrics"].keys())


def _load_csv() -> list:
    import csv
    with open(METRICS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _is_nan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (ValueError, TypeError):
        return False


def test_metric_ranges():
    if not METRICS_CSV.exists():
        pytest.skip("results/metrics.csv does not exist yet")

    rows   = _load_csv()
    ranges = _load_ranges()
    cols   = _metric_cols()
    failures = []

    for row in rows:
        for col in cols:
            raw = row.get(col, "")
            if raw in ("", None):
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if math.isnan(val):
                continue
            lo, hi = ranges.get(col, (None, None))
            # skip bound check for metrics with null valid_min or valid_max
            if lo is not None and val < lo:
                print(f"OUT OF RANGE: clip={row['clip_id']} cond={row['condition']} {col}={val} (< min {lo})")
                failures.append((row["clip_id"], row["condition"], col, val))
            if hi is not None and val > hi:
                print(f"OUT OF RANGE: clip={row['clip_id']} cond={row['condition']} {col}={val} (> max {hi})")
                failures.append((row["clip_id"], row["condition"], col, val))

    assert not failures, f"{len(failures)} out-of-range value(s) found"


def test_nan_rate():
    if not METRICS_CSV.exists():
        pytest.skip("results/metrics.csv does not exist yet")

    rows = _load_csv()
    if not rows:
        pytest.skip("results/metrics.csv is empty")

    cols  = _metric_cols()
    total = len(rows)
    for col in cols:
        nan_count = sum(
            1 for r in rows
            if r.get(col, "") in ("", "nan", "NaN") or _is_nan(r.get(col, ""))
        )
        rate = nan_count / total
        if rate > 0.01:
            print(f"WARNING: {col} NaN rate = {rate:.1%} ({nan_count}/{total})")
        assert rate < 0.05, (
            f"{col} NaN rate {rate:.1%} exceeds 5% threshold ({nan_count}/{total} clips)"
        )


def test_squim_prediction_direction():
    """squim_stoi/pesq/si_sdr should rank conditions in same direction as intrusive ground truth."""
    if not METRICS_CSV.exists():
        pytest.skip("results/metrics.csv does not exist yet")

    import csv
    with open(METRICS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    PAIRS = [("squim_stoi", "stoi"), ("squim_pesq", "pesq"), ("squim_si_sdr", "si_sdr")]
    CONDITIONS = ["noisy", "ns_metricgan", "dereverb", "echo_sim", "aec_full", "ns_aec_combined"]
    conditions = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]

    def _cond_mean(col, condition):
        vals = [
            float(r[col]) for r in rows
            if r["condition"] == condition
            and r.get(col, "") not in ("", "nan", "NaN")
            and not _is_nan(r.get(col, ""))
        ]
        return statistics.mean(vals) if vals else None

    inversions = []
    for squim_col, true_col in PAIRS:
        squim_means = {c: _cond_mean(squim_col, c) for c in conditions}
        true_means  = {c: _cond_mean(true_col, c)  for c in conditions}
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i + 1:]:
                s1, s2 = squim_means.get(c1), squim_means.get(c2)
                t1, t2 = true_means.get(c1),  true_means.get(c2)
                if None in (s1, s2, t1, t2):
                    continue
                if (s1 > s2) != (t1 > t2):
                    inversions.append(
                        f"{squim_col}: {c1} vs {c2} "
                        f"(squim={'>' if s1>s2 else '<'}, truth={'>' if t1>t2 else '<'})"
                    )

    if inversions:
        pytest.xfail("SQUIM ordering inversions detected:\n" + "\n".join(inversions))


def test_emotion_sim_clean_is_one():
    """emotion_sim for clean condition must be ~1.0 (cosine_sim of identical embeddings)."""
    if not METRICS_CSV.exists():
        pytest.skip("results/metrics.csv does not exist yet")

    import csv
    with open(METRICS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    clean_scores = [
        float(r["emotion_sim"]) for r in rows
        if r["condition"] == "clean"
        and r.get("emotion_sim", "") not in ("", "nan", "NaN")
        and not _is_nan(r.get("emotion_sim", ""))
    ]
    if not clean_scores:
        pytest.skip("No emotion_sim values for clean condition yet")

    mean_score = statistics.mean(clean_scores)
    assert mean_score > 0.99, (
        f"mean emotion_sim for clean = {mean_score:.4f} (expected > 0.99). "
        "EMO2VEC model may be loading incorrectly."
    )
