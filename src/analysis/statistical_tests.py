"""Statistical tests: ODR significance, pairwise McNemar, WER-ODR gap, cascade vs E2E."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).parent.parent.parent
CONDITIONS = ["noisy", "ns_metricgan", "ns_diffusion", "dereverb", "aec_sim", "ns_aec_combined"]
FIELDNAMES = ["test_name", "statistic", "p_value", "p_corrected", "effect_size", "significant"]


def _cohens_h(p1: float, p2: float = 0.0) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def _fmt(v) -> str:
    return f"{v:.4f}" if (v is not None and v == v) else "---"


def _to_latex(df: pd.DataFrame) -> str:
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\begin{tabular}{lrrrrc}", r"\toprule",
             r"Test & Stat & $p$ & $p_{\mathrm{corr}}$ & Effect & Sig.\\",
             r"\midrule"]
    for _, r in df.iterrows():
        sig = r"\checkmark" if r["significant"] else ""
        lines.append(f"{r['test_name']} & {_fmt(r['statistic'])} & {_fmt(r['p_value'])} "
                     f"& {_fmt(r['p_corrected'])} & {_fmt(r['effect_size'])} & {sig} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def run_all_tests(
    divergence_csv: str | Path,
    clip_divergence_csv: str | Path,
) -> pd.DataFrame:
    """Run all four statistical tests; save CSV and LaTeX table."""
    div_df  = pd.read_csv(Path(divergence_csv))
    clip_df = pd.read_csv(Path(clip_divergence_csv))

    rows = []

    # ── Test 1: each condition ODR > 0 (binomtest) ──────────────────────────
    for _, r in div_df.iterrows():
        res = stats.binomtest(int(r["n_divergent"]), int(r["n_clips"]), p=0.0, alternative="greater")
        rows.append({"test_name": f"odr>0:{r['condition']}",
                     "statistic": round(float(res.statistic), 4),
                     "p_value":   round(float(res.pvalue), 6), "p_corrected": None,
                     "effect_size": round(_cohens_h(r["odr"]), 4),
                     "significant": bool(res.pvalue < 0.05)})

    # ── Test 2: pairwise McNemar with Holm-Bonferroni (21 pairs) ────────────
    pivot = clip_df.pivot_table(index="clip_id", columns="condition", values="diverged")
    pairs = list(combinations(CONDITIONS, 2))
    mc_stats_list, p_raw = [], []
    for c1, c2 in pairs:
        if c1 not in pivot.columns or c2 not in pivot.columns:
            mc_stats_list.append((0.0, 1.0)); p_raw.append(1.0); continue
        sub = pivot[[c1, c2]].dropna()
        n00 = int(((sub[c1] == 0) & (sub[c2] == 0)).sum())
        n11 = int(((sub[c1] == 1) & (sub[c2] == 1)).sum())
        b   = int(((sub[c1] == 1) & (sub[c2] == 0)).sum())
        c   = int(((sub[c1] == 0) & (sub[c2] == 1)).sum())
        if b + c == 0:
            mc_stats_list.append((0.0, 1.0)); p_raw.append(1.0); continue
        res = mcnemar_test(np.array([[n00, b], [c, n11]]), exact=True)
        mc_stats_list.append((float(res.statistic) if res.statistic else 0.0, float(res.pvalue)))
        p_raw.append(float(res.pvalue))

    _, p_holm, _, _ = multipletests(p_raw, method="holm")
    for (c1, c2), (stat, pv), pc in zip(pairs, mc_stats_list, p_holm):
        rows.append({"test_name": f"mcnemar:{c1}_vs_{c2}",
                     "statistic": round(stat, 4), "p_value": round(pv, 6),
                     "p_corrected": round(float(pc), 6), "effect_size": None,
                     "significant": bool(float(pc) < 0.05)})

    # ── Test 3: WER-ODR gap > 0 (one-sample t-test) ─────────────────────────
    gaps = div_df["wer_odr_gap"].dropna().values.astype(float)
    if len(gaps) > 1:
        t_res = stats.ttest_1samp(gaps, popmean=0, alternative="greater")
        rows.append({"test_name": "wer_odr_gap>0",
                     "statistic": round(float(t_res.statistic), 4),
                     "p_value":   round(float(t_res.pvalue), 6), "p_corrected": None,
                     "effect_size": round(float(gaps.mean()), 4),
                     "significant": bool(t_res.pvalue < 0.05)})

    # ── Test 4: cascade vs E2E (McNemar; skip gracefully if E2E not ready) ──
    for src, label in [("intents_gpt.csv", "cascade"), ("intents_e2e.csv", "e2e")]:
        if not (ROOT / "results" / src).exists():
            break
    else:
        gpt = pd.read_csv(ROOT / "results" / "intents_gpt.csv")
        e2e = pd.read_csv(ROOT / "results" / "intents_e2e.csv")
        # Build paired divergence using clean reference for each model
        clean_gpt = gpt[gpt["condition"] == "clean"][["clip_id", "predicted_intent"]].rename(
            columns={"predicted_intent": "clean_gpt"})
        clean_e2e = e2e[e2e["condition"] == "clean"][["clip_id", "predicted_intent"]].rename(
            columns={"predicted_intent": "clean_e2e"})
        paired = (gpt[gpt["condition"] != "clean"]
                  .merge(e2e[e2e["condition"] != "clean"], on=["clip_id", "condition"],
                         suffixes=("_gpt", "_e2e"))
                  .merge(clean_gpt, on="clip_id").merge(clean_e2e, on="clip_id"))
        dg = (paired["predicted_intent_gpt"] != paired["clean_gpt"]).astype(int)
        de = (paired["predicted_intent_e2e"] != paired["clean_e2e"]).astype(int)
        table = np.array([[((dg == 0) & (de == 0)).sum(), ((dg == 0) & (de == 1)).sum()],
                          [((dg == 1) & (de == 0)).sum(), ((dg == 1) & (de == 1)).sum()]])
        res = mcnemar_test(table, exact=True)
        rows.append({"test_name": "mcnemar:cascade_vs_e2e",
                     "statistic": round(float(res.statistic) if res.statistic else 0.0, 4),
                     "p_value": round(float(res.pvalue), 6), "p_corrected": None,
                     "effect_size": None, "significant": bool(res.pvalue < 0.05)})

    result = pd.DataFrame(rows, columns=FIELDNAMES)
    result.to_csv(ROOT / "results" / "statistical_tests.csv", index=False)
    tex_path = ROOT / "results" / "tables" / "stats_table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(_to_latex(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    df = run_all_tests(
        ROOT / "results" / "divergence.csv",
        ROOT / "results" / "clip_level_divergence.csv",
    )
    gap_row = df[df["test_name"] == "wer_odr_gap>0"]
    if not gap_row.empty:
        r = gap_row.iloc[0]
        verdict = (
            "enhancement causes MORE semantic damage than WER reveals"
            if r["p_value"] < 0.05 and r["effect_size"] > 0
            else "gap not significant"
        )
        print(f"WER-ODR gap test: t={r['statistic']:.3f}, p={r['p_value']:.4f} — {verdict}")
