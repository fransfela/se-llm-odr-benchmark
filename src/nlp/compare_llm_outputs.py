"""Compare GPT-4o and Llama-3-70B intent classification outputs per condition."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
CONDITIONS = ["clean", "noisy", "ns_metricgan", "ns_diffusion",
              "dereverb", "aec_sim", "ns_aec_combined"]


def compare_outputs(
    gpt_csv: str | Path,
    llama_csv: str | Path,
) -> pd.DataFrame:
    """Merge GPT and Llama predictions; compute agreement and ODR metrics per condition."""
    gpt_csv   = Path(gpt_csv)
    llama_csv = Path(llama_csv)

    gpt   = pd.read_csv(gpt_csv)
    llama = pd.read_csv(llama_csv)

    # Merge on clip_id + condition; keep only valid predictions for ODR denominator
    merged = gpt[["clip_id", "condition", "predicted_intent", "is_valid"]].merge(
        llama[["clip_id", "condition", "predicted_intent", "is_valid"]],
        on=["clip_id", "condition"],
        suffixes=("_gpt", "_llama"),
    )

    # Fetch clean-condition predictions for each model (ODR reference)
    clean_gpt = (
        merged.loc[merged["condition"] == "clean", ["clip_id", "predicted_intent_gpt"]]
        .rename(columns={"predicted_intent_gpt": "clean_gpt"})
    )
    clean_llama = (
        merged.loc[merged["condition"] == "clean", ["clip_id", "predicted_intent_llama"]]
        .rename(columns={"predicted_intent_llama": "clean_llama"})
    )

    merged = merged.merge(clean_gpt,   on="clip_id", how="left")
    merged = merged.merge(clean_llama, on="clip_id", how="left")

    rows = []
    for condition in CONDITIONS:
        sub = merged[merged["condition"] == condition].copy()
        if sub.empty:
            continue

        # ODR denominators: only valid predictions
        valid_gpt   = sub[sub["is_valid_gpt"]   == True]  # noqa: E712
        valid_llama = sub[sub["is_valid_llama"] == True]  # noqa: E712

        agreement_rate = (
            sub["predicted_intent_gpt"] == sub["predicted_intent_llama"]
        ).mean()

        gpt_odr   = ((valid_gpt["predicted_intent_gpt"]   != valid_gpt["clean_gpt"]).mean()
                     if len(valid_gpt)   > 0 else float("nan"))
        llama_odr = ((valid_llama["predicted_intent_llama"] != valid_llama["clean_llama"]).mean()
                     if len(valid_llama) > 0 else float("nan"))

        import math
        odr_delta = (abs(gpt_odr - llama_odr)
                     if not (math.isnan(gpt_odr) or math.isnan(llama_odr))
                     else float("nan"))

        rows.append({
            "condition":      condition,
            "agreement_rate": round(agreement_rate, 4),
            "gpt_odr":        round(gpt_odr,        4) if gpt_odr == gpt_odr     else float("nan"),
            "llama_odr":      round(llama_odr,       4) if llama_odr == llama_odr else float("nan"),
            "odr_delta":      round(odr_delta,       4) if odr_delta == odr_delta else float("nan"),
        })

    result = pd.DataFrame(rows, columns=["condition", "agreement_rate",
                                          "gpt_odr", "llama_odr", "odr_delta"])

    out_path = ROOT / "results" / "llm_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    mean_agreement = result["agreement_rate"].mean()
    verdict = (
        "consistent"
        if mean_agreement > 0.85
        else "divergent, investigate before reporting"
    )
    print(f"Model agreement: {mean_agreement:.1%} — findings are '{verdict}'")

    return result
