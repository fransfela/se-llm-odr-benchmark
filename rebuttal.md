# Rebuttal / Revision Tracker

Paper: "Perceptually Better, Semantically Worse: Measuring Speech Enhancement
Impact on LLM-Based Voice Systems" (EMNLP 2026 Industry Track, Submission #461)

Reviews received: 21 Jul (zPYo), 17 Jul (i7kA), 11 Jul (mUUj)
Revision deadline: 2 Aug 2026
Plan created: 29 Jul 2026

This file tracks every change made to the submission in response to reviewer
feedback, ordered easiest -> hardest. Update this file whenever the paper,
tables, or analysis scripts change as part of the revision.

---

## Status legend
- [x] Done
- [~] Partial / in progress
- [ ] Not started (needs a decision or more time)

---

## Easy (done 29 Jul 2026)

- [x] **Table 2 caption referenced ODR but the table had no ODR column** (mUUj,
  "Questions"). Added an ODR column to `paper/tables/table2_metrics.tex` with
  per-condition Whisper ODR values; caption unchanged in substance.
- [x] **No conclusion section, paper hard to read** (mUUj, "Questions"). Added
  a short `\section{Conclusion}` in `paper/emnlp2026_industry.tex` (before
  Limitations) summarizing the ODR contribution, the MetricGAN+/echo
  findings, the correlation results, and the new correction-vs-regression
  finding.
- [x] **ODR requires a clean reference, limiting real-world deployment use**
  (i7kA, reason to reject). Added a new Limitations bullet ("Clean-reference
  requirement") framing ODR as an offline/development-time evaluation tool
  (like PESQ/STOI) rather than an online monitor, and pointing to
  reference-free proxies as future work.
- [x] **Limited SE coverage (only MetricGAN+ as discriminative SE)** (zPYo,
  reason to reject). Added Limitations bullet ("SE method coverage").
- [x] **No encoder-based classifier tested, conclusions may differ with other
  classifiers** (mUUj, reason to reject). Added Limitations bullet
  ("Classifier architecture").
- [x] **Named-entity / proper-noun question** (zPYo, "Questions"). Added
  Limitations bullet ("Entity-level effects") explaining that the SLURP test
  release used here does not expose entity annotations, that intent labels
  are domain/action pairs that often survive entity-level ASR substitutions,
  and that ODR may therefore understate SE's impact on entity-dependent
  tasks (slot filling / NER) -- flagged explicitly as future work rather than
  claimed as answered.

## Medium (done 29 Jul 2026) — new analyses from existing data, no new experiments

- [x] **No analysis of SE impact on correct predictions; ODR conflates harm
  with beneficial correction** (zPYo, reason to reject — the most substantive
  methodological critique). Added `src/analysis/rebuttal_analysis.py`
  (`correction_vs_regression`), which joins SLURP ground-truth intents with
  the clean and per-condition Gemini predictions and splits every divergent
  clip into **correction** (clean wrong -> condition correct), **regression**
  (clean correct -> condition wrong), or **both-wrong** (mutually different,
  both wrong). Output: `results/correction_vs_regression.csv`.
  - Regressions outnumber corrections in every condition: 3.3:1 (Noisy),
    7.6:1 (MetricGAN+), 23.1:1 (Echo sim), 10.0:1 (Echo+DTLN-AEC), 1.9:1
    (Dereverb). This directly answers the critique: ODR is overwhelmingly
    capturing harm, not benign correction.
  - Added new paragraph "Correct vs. harmful divergence" in
    `\subsection{ODR Across Conditions}` (main body) + full breakdown as
    Table 6 in a new Appendix section `app:correction`
    (`paper/tables/table6_correction_regression.tex`).
- [x] **Report where quality metrics fail (false positive / false negative
  cases)** (i7kA, reason to reject). Added `metric_failure_analysis` in the
  same script: dichotomizes clips by within-condition median PESQ and
  reports (a) % of above-median-PESQ clips that still diverge (PESQ misses
  real damage) and (b) % of below-median-PESQ clips that do not diverge
  (PESQ over-warns). Output: `results/metric_failure_analysis.csv`.
  - Echo (sim): 41.2% false-negative rate (PESQ gives no warning exactly
    where damage is worst). Noisy/Dereverb: 42-44% false-positive rate
    (PESQ over-warns on clips the LLM handles fine).
  - Added new paragraph "Where PESQ fails as an early-warning signal" in
    `\subsection{Quality Metrics as ODR Predictors}` + Table 7 in a new
    Appendix section `app:metricfail`
    (`paper/tables/table7_metric_failure.tex`).
- [x] Recompiled paper (`compile.ps1`): 10 pages total (appendix does not
  count toward the 6-page limit), no LaTeX errors, no undefined
  references/citations, no overfull hboxes.

## Hard — Gemini 2.5 Pro second-LLM replication (addresses ALL 3 reviewers)

**Status:** Partially complete. Quota exhausted after clean + noisy + partial
ns_metricgan. Remaining budget = $0 (x-ratelimit-remaining-cost: 0).

**Final data state (verified 30 Jul):**
  - `clean`: 2974/2974 valid ✓
  - `noisy`: 2974/2974 valid ✓
  - `ns_metricgan`: 1733/2974 valid (58%) -- subset is representative
    (Flash Lite ODR on same subset = 0.305, vs full-set 0.318)
  - `aec_sim`, `dereverb`, `aec_full`: 0 valid (all quota failures)

**Results:**
  - Pro noisy ODR = 0.154, Pro MetricGAN+ ODR = 0.294
  - MetricGAN+ / Noisy ratio: 1.91x (cf. Flash Lite 2.36x)
  - Central finding replicates: MetricGAN+ nearly doubles ODR under Pro

**Decision:** Use in author response as supplementary evidence + update
Limitations bullet. Do NOT add as full subsection in paper body.

**Steps completed:**
- [x] Check if APIM quota has reset (yes, on 29 Jul)
- [x] Purge failed rows from noisy CSV (kept 839 valid)
- [x] Delete ns_metricgan CSV (0 valid -> reran from scratch)
- [x] Run classification: noisy complete, ns_metricgan 58%, others quota-blocked
- [x] Compute ODR (src/analysis/gemini_pro_odr.py)
- [x] Update Limitations bullet in paper (softened "Single LLM")
- [x] Mention in author response with numbers + repo pointer

## Out of scope for Aug 2

- [ ] **Real-world (non-simulated) recordings** (zPYo, reason to reject). No
  real device-compound-distortion data is available in this repo; addressed
  only as an existing Limitations bullet ("Simulated conditions"). Would
  require new data collection — out of scope.
- [ ] **Additional generative task beyond closed-set intent classification**
  (zPYo, reason to reject). Not attempted — would require a new task
  definition, prompts, and ground truth; out of scope. Left as a Limitations
  point ("Closed-set classification").

## Author response (to write after experiments settle)

- [ ] Draft OpenReview author response addressing each reviewer's concerns,
  pointing to specific paper sections/tables that now answer them.

---

## Reviewer clarifications (for author response)

- **mUUj: "no encoder-based classification comparison"** — The paper DOES test
  wav2vec2-large (self-supervised encoder + CTC, Section 4.3). The reviewer
  likely means an encoder-based *intent classifier* (BERT/RoBERTa joint
  intent+slot model), which is a different and fair point — addressed in
  Limitations ("Classifier architecture"). Will clarify in author response.
- **zPYo: "correlation analysis methodological concerns"** — Already addressed
  in paper: "Pooled correlations overstate predictive value" paragraph
  explicitly disaggregates within-condition vs between-condition effects, and
  the PESQ failure analysis (Table 7) provides the clip-level evidence.
- **zPYo: "simulated conditions"** — Acknowledged limitation; no real-world
  data available. However, DNS Challenge noise/RIRs are the standard
  evaluation conditions used by the SE community (URGENT, DNS Challenge),
  making our results directly comparable to published SE benchmarks.

---

## Files touched

- `paper/emnlp2026_industry.tex` — Conclusion section, Limitations (4 new
  bullets), 2 new Results paragraphs, 2 new Appendix sections.
- `paper/tables/table2_metrics.tex` — added ODR column.
- `paper/tables/table6_correction_regression.tex` — new.
- `paper/tables/table7_metric_failure.tex` — new.
- `src/analysis/rebuttal_analysis.py` — new analysis script.
- `results/correction_vs_regression.csv` — new.
- `results/metric_failure_analysis.csv` — new.
