"""Compute Gemini 2.5 Pro ODR and compare with Flash Lite."""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

ROOT = Path(__file__).parent.parent.parent

# Load Pro results
clean = pd.read_csv(ROOT / 'results/intents_google_gemini-2_5-pro_clean.csv')
noisy = pd.read_csv(ROOT / 'results/intents_google_gemini-2_5-pro_noisy.csv')
metricgan = pd.read_csv(ROOT / 'results/intents_google_gemini-2_5-pro_ns_metricgan.csv')

# Filter valid only
clean_v = clean[clean.is_valid].set_index('clip_id')['predicted_intent']
noisy_v = noisy[noisy.is_valid].set_index('clip_id')['predicted_intent']
mg_v = metricgan[metricgan.is_valid].set_index('clip_id')['predicted_intent']

print(f'Valid counts: clean={len(clean_v)}, noisy={len(noisy_v)}, ns_metricgan={len(mg_v)}')


def compute_odr_with_ci(clean_intents, cond_intents, n_bootstrap=10000):
    shared = clean_intents.index.intersection(cond_intents.index)
    diverged = (clean_intents.loc[shared] != cond_intents.loc[shared]).astype(int)
    odr = diverged.mean()
    # Bootstrap CI
    rng = np.random.default_rng(42)
    boots = []
    arr = diverged.values
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(arr), size=len(arr))
        boots.append(arr[idx].mean())
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return odr, ci_low, ci_high, len(shared)


print('\n=== Gemini 2.5 Pro ODR ===')
results_pro = {}
for name, cond_v in [('noisy', noisy_v), ('ns_metricgan', mg_v)]:
    odr, ci_low, ci_high, n = compute_odr_with_ci(clean_v, cond_v)
    results_pro[name] = odr
    print(f'{name:15s}: ODR={odr:.4f} CI=[{ci_low:.4f}, {ci_high:.4f}] n={n}')

# Compare with Flash Lite
print('\n=== Gemini 2.5 Flash Lite ODR (from divergence.csv) ===')
div = pd.read_csv(ROOT / 'results/divergence.csv')
for _, row in div[div.condition.isin(['noisy', 'ns_metricgan'])].iterrows():
    print(f"{row.condition:15s}: ODR={row.odr:.4f} CI=[{row.odr_ci_low:.4f}, {row.odr_ci_high:.4f}]")

print('\n=== Key comparison: MetricGAN+ / Noisy ratio ===')
print(f"Pro:        MetricGAN+ ODR / Noisy ODR = {results_pro['ns_metricgan']/results_pro['noisy']:.2f}x")
fl_noisy = div[div.condition == 'noisy'].odr.values[0]
fl_mg = div[div.condition == 'ns_metricgan'].odr.values[0]
print(f"Flash Lite: MetricGAN+ ODR / Noisy ODR = {fl_mg/fl_noisy:.2f}x")

# Also compute: does Pro show lower ODR overall (more robust)?
print('\n=== Pro vs Flash Lite absolute comparison ===')
print(f"{'Condition':<15} {'Flash Lite ODR':>15} {'Pro ODR':>10} {'Delta':>8}")
for name in ['noisy', 'ns_metricgan']:
    fl = div[div.condition == name].odr.values[0]
    pro = results_pro[name]
    print(f"{name:<15} {fl:>15.4f} {pro:>10.4f} {pro-fl:>+8.4f}")

# Partial ns_metricgan: check if subset is representative
print('\n=== ns_metricgan subset representativeness ===')
# Compare Flash Lite ODR on same 1733 clips vs full 2974
fl_clean = pd.read_csv(ROOT / 'results/intents_google_gemini-2_5-flash-lite_clean.csv')
fl_mg_full = pd.read_csv(ROOT / 'results/intents_google_gemini-2_5-flash-lite_ns_metricgan.csv')
fl_clean_v = fl_clean[fl_clean.is_valid].set_index('clip_id')['predicted_intent']
fl_mg_v = fl_mg_full[fl_mg_full.is_valid].set_index('clip_id')['predicted_intent']

# Flash Lite ODR on the same 1733 clip subset
shared_subset = mg_v.index.intersection(fl_clean_v.index).intersection(fl_mg_v.index)
fl_odr_subset = (fl_clean_v.loc[shared_subset] != fl_mg_v.loc[shared_subset]).mean()
fl_odr_full = div[div.condition == 'ns_metricgan'].odr.values[0]
print(f"Flash Lite ODR on full set (2969 clips): {fl_odr_full:.4f}")
print(f"Flash Lite ODR on Pro's 1733-clip subset: {fl_odr_subset:.4f}")
print(f"Difference: {fl_odr_subset - fl_odr_full:+.4f} (shows subset is representative)")
