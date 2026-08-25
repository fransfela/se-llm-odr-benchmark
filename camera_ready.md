# Camera-Ready Checklist — EMNLP 2026 Industry Track #461

**Deadline:** August 30, AoE  
**Submission portal:** https://openreview.net/forum?id=NUmVKOsUq3  
**Page budget:** 7 pages main body + 1 page Limitations/Ethics + unlimited refs/appendix

---

## Status

- [x] De-anonymize paper (`\usepackage[hyperref]{acl}`, compact two-author block, `GN Group, Ballerup, Denmark`)
- [x] Replace anonymous repo URL → `https://github.com/fransfela/se-llm-odr-benchmark`
- [x] Remove deprecated `\aclfinalcopy` (not in current acl.sty)
- [x] Compile clean: 11 pages, 0 errors, 0 overfull hboxes
- [x] Page count verified: main body = 7 pages (Sections 1–6, pp. 1–7)
- [x] README updated: remove "anonymous, under review", add BibTeX citation
- [ ] Fix malformed `chen2024voicebench` bib entry (see §B below)
- [ ] Manual spot-check: §B numbers below
- [ ] Verify arXiv references exist at stated IDs (§C below)
- [ ] ORCID IDs for both authors in OpenReview profiles
- [ ] Fill OpenReview camera-ready form (portal open Aug 23–30)

---

## A. Numbers verified against results CSVs

| Claim | Ground truth | OK? |
|---|---|---|
| 2,974 SLURP clips, 77 intents | CLAUDE.md | ✅ |
| MetricGAN+ ODR 0.318 (CI [0.301, 0.335]) | divergence.csv: 0.3183, [0.3014, 0.3351] | ✅ |
| Noisy ODR 0.135 (CI [0.123, 0.147]) | divergence.csv: 0.1351, [0.1226, 0.1472] | ✅ |
| Dereverb ODR 0.108 (CI [0.097, 0.119]) | divergence.csv: 0.1077, [0.0970, 0.1189] | ✅ |
| wav2vec2 MetricGAN+ 0.474, Noisy 0.352 | divergence_wav2vec2.csv: 0.4744, 0.3516 | ✅ |
| Echo sim ODR 0.836 (CI [0.823, 0.849]) | divergence.csv: 0.8363, [0.8232, 0.8494] | ✅ |
| Echo sim wav2vec2 ODR 0.853 | divergence_wav2vec2.csv: 0.8527 | ✅ |
| AEC ODR 0.404 (CI [0.386, 0.422], Gap −0.245) | divergence.csv: 0.4042, [0.3864, 0.4218], gap −0.2445 | ✅ |
| Whisper WER gaps −0.092 to −0.400 | divergence.csv: −0.0922 to −0.3996 | ✅ |
| wav2vec2 WER gaps −0.114 to −0.150 | divergence_wav2vec2.csv: −0.1137 to −0.1497 | ✅ |
| WER drop 1.448→0.700 = −51.6% | (1−0.700/1.448) = 51.7% | ✅ |
| ODR drop 0.836→0.404 = −51.7% | (1−0.404/0.836) = 51.7% | ✅ |
| PESQ 1.76→1.94 (+0.18), STOI 0.87→0.80 (−0.07) | table2_metrics.tex | ✅ |
| SI-SDR −5.01→−26.13 (AEC) | table2_metrics.tex | ✅ |
| PESQ ρ = −0.467 | correlations.csv | ✅ |
| SQUIM-MOS ρ_ODR = −0.068, ρ_WER = −0.069 | correlations.csv | ✅ |
| Within-condition max \|ρ\| = 0.34 (STOI / MetricGAN+) | table8: −0.340 | ✅ |
| PESQ within-condition \|ρ\| = 0.26 | table8: −0.259 (rounded) | ✅ |
| 4,149 divergent clips | error_taxonomy.csv rows | ✅ |
| Regression ratios 3.3:1 / 7.6:1 / 23.1:1 / 10.0:1 / 1.9:1 | table5_correction_regression.tex | ✅ |
| Domain ODR: Lists 0.419, Alarm 0.386, News 0.319, QA 0.315 | table7_domain_odr.tex | ✅ |

---

## B. Numbers requiring manual spot-check

These cannot be verified from the CSV summaries alone — check the raw intent files:

- [ ] "48.4% of the **562 regressions** under MetricGAN+ land in `general_quirky` (13.4% vs. 5.2%) or `general_greet` (6.8% vs. 2.0%)"  
  → cross-check `error_taxonomy.csv` + `intents_google_gemini-2_5-flash-lite_ns_metricgan.csv`

- [ ] "90.4% represent domain-level shifts, 9.6% within-domain action confusion"  
  → check `error_type` column in `error_taxonomy.csv`

- [ ] McNemar χ²=427, p<10⁻⁹⁰ (MetricGAN+ vs. Noisy)  
  → rerun `src/analysis/rebuttal_analysis.py` or spot-check

---

## C. Bibliography — one confirmed fix needed

**`chen2024voicebench`** has a malformed entry in `paper/custom.bib`:

```bibtex
% WRONG (current):
journal={URL https://arxiv. org/abs/2410},
volume={17196}

% CORRECT (fix to):
journal={arXiv preprint arXiv:2410.17196},
```

**arXiv IDs to manually verify exist:**

| Key | arXiv ID | Notes |
|---|---|---|
| `chondhekar2025noising` | 2512.17562 | "When De-noising Hurts" |
| `islam2026denoising` | 2603.04710 | "When Denoising Hinders" |
| `saijo2025interspeech` | 2505.23212 | INTERSPEECH 2025 URGENT |
| `zhang2024urgent` | 2406.04660 | URGENT Challenge |
| `chen2026voicebench` | TACL vol.14 pp.378–398 | Confirm published in TACL |

---

## D. OpenReview submission form

**Have ready before opening the portal:**

| Field | Value |
|---|---|
| Presenting author email | (Randy or Pejman?) |
| Country of residence | Denmark |
| Visa required? | No (EU/Schengen) |
| In-person or virtual? | |
| Oral or poster preference? | |
| Anticipated travel dates to Budapest | |
| ORCID — Randy Frans Fela | Add to OpenReview profile |
| ORCID — Pejman Mowlaee | Add to OpenReview profile |

---

## E. Figure captions (visual check in PDF)

- [ ] **Fig 1** — ODR bar chart: all 5 conditions shown for both ASR models, CIs visible
- [ ] **Fig 2** — WER vs ODR scatter: ★ clean baseline markers present, all 10 points below diagonal
- [ ] **Fig 3a/3b** — Correlation dot plots: SQUIM-MOS near zero, error bars present
