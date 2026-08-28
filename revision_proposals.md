# Revision Proposals — Before / After

All excerpts are quoted verbatim from `paper/emnlp2026_industry.tex`.
Each item states the issue, shows the exact passage to replace, and
provides the corrected version ready to paste in.

---

## P1-A — Factual error: regression-ratio range excludes Dereverb 1.9:1

**Issue.** The sentence implies 3.3:1 (Noisy) is the floor of the
range, but Table 5 shows Dereverb at 1.9:1, which is the actual minimum.

**BEFORE**
```latex
Regressions outnumber corrections in
every condition, from 3.3\,:\,1 under Noisy to 23.1\,:\,1 under
Echo (sim), confirming that ODR predominantly captures harmful
divergence rather than benign correction.
```

**AFTER**
```latex
Regressions outnumber corrections in
every condition, ranging from 1.9\,:\,1 under Dereverb to
23.1\,:\,1 under Echo~(sim), with Noisy at 3.3\,:\,1, confirming
that ODR predominantly captures harmful divergence rather than
benign correction.
```

---

## P1-B — Fig. 2 caption overgeneralises "LLM robustness" for Echo (sim)

**Issue.** The body text correctly says the Echo (sim) gap reflects
WER saturation, not LLM robustness. The caption contradicts this by
saying "all conditions."

**BEFORE**
```latex
\caption{WER vs.\ ODR for Whisper (filled markers) and wav2vec2 (open markers).
\textbf{$\star$} marks the clean baseline (ODR\,=\,0 by construction,
WER reflecting Whisper/wav2vec2 accuracy on spontaneous SLURP speech).
All enhancement conditions fall below $y\!=\!x$ (dashed), indicating
partial LLM robustness to transcription errors across all conditions
and architectures.}
```

**AFTER**
```latex
\caption{WER vs.\ ODR for Whisper (filled markers) and wav2vec2 (open markers).
\textbf{$\star$} marks the clean baseline (ODR\,=\,0 by construction,
WER reflecting Whisper/wav2vec2 accuracy on spontaneous SLURP speech).
All points fall below $y\!=\!x$ (dashed); for spectral distortion
conditions this reflects partial LLM robustness to transcription
errors, while for Echo~(sim) the negative gap reflects WER ceiling
saturation rather than LLM recovery.}
```

---

## P2-A — Table 2: raw WER vs. capped WER trap for gap calculation

**Issue.** Table 2 shows WER = 1.448 (raw) for Echo (sim) in bold,
but the gap formula uses the capped mean of 0.928. A reader computing
the gap from the table gets −0.612, not −0.092. The caption footnote
is insufficient; a second column makes the distinction unambiguous.

**BEFORE** (in `paper/tables/table2_metrics.tex`)
```latex
    \textbf{Condition} & \textbf{PESQ} & \textbf{STOI} &
    \textbf{SNR} & \textbf{SI-SDR} &
    \textbf{SRMR} & \textbf{SQUIM} & \textbf{WER} & \textbf{ODR} \\
    \hline
    Clean           & 4.64 & 1.00 & 105.12 & 94.35 & 8.56 & 4.22 & 0.507 & -- \\
    Noisy           & 1.76 & 0.87 & 9.92 & 10.00 & 6.24 & 3.77 & 0.531 & 0.135 \\
    MetricGAN+      & 1.94 & 0.80 & 1.81 & 4.23 & 10.96 & 3.87 & 0.647 & \textbf{0.318} \\
    Echo (sim)      & 1.11 & 0.52 & $-$2.69 & $-$5.01 & 6.24 & 3.69 & \textbf{1.448} & \textbf{0.836} \\
    Echo + DTLN-AEC & 1.53 & 0.47 & $-$1.66 & $-$26.13 & 10.98 & 3.92 & 0.700 & 0.404 \\
    Dereverb        & 2.43 & 0.71 & $-$4.72 & $-$23.04 & 7.79 & 3.94 & 0.517 & 0.108 \\
```

**AFTER** — add a "WER$_\text{cap}$" column (used in the gap formula)
```latex
    \textbf{Condition} & \textbf{PESQ} & \textbf{STOI} &
    \textbf{SNR} & \textbf{SI-SDR} &
    \textbf{SRMR} & \textbf{SQUIM} & \textbf{WER} & \textbf{WER$_\text{cap}$} & \textbf{ODR} \\
    \hline
    Clean           & 4.64 & 1.00 & 105.12 & 94.35 & 8.56 & 4.22 & 0.507 & 0.507 & -- \\
    Noisy           & 1.76 & 0.87 & 9.92 & 10.00 & 6.24 & 3.77 & 0.531 & 0.531 & 0.135 \\
    MetricGAN+      & 1.94 & 0.80 & 1.81 & 4.23 & 10.96 & 3.87 & 0.647 & 0.647 & \textbf{0.318} \\
    Echo (sim)      & 1.11 & 0.52 & $-$2.69 & $-$5.01 & 6.24 & 3.69 & \textbf{1.448} & 0.928 & \textbf{0.836} \\
    Echo + DTLN-AEC & 1.53 & 0.47 & $-$1.66 & $-$26.13 & 10.98 & 3.92 & 0.700 & 0.700 & 0.404 \\
    Dereverb        & 2.43 & 0.71 & $-$4.72 & $-$23.04 & 7.79 & 3.94 & 0.517 & 0.517 & 0.108 \\
```

Also update the caption to name the column:

**BEFORE** (caption excerpt)
```latex
    Bold marks the highest ODR values
    (Echo~(sim) and MetricGAN{+}) and the saturating WER
    (Echo~(sim) 1.448, raw Whisper WER before per-clip capping at 1.0).}
```

**AFTER**
```latex
    WER: raw mean per-clip word error rate. WER$_\text{cap}$: mean of
    $\min(\text{WER}_i, 1.0)$ per clip, used in the WER--ODR gap
    (Eq.~\ref{eq:gap}). Bold marks the highest ODR values
    (Echo~(sim) and MetricGAN{+}) and the saturating raw WER
    (Echo~(sim) 1.448).}
```

---

## P2-B — LLM robustness conclusion too strong for a 58 % partial replication

**Issue.** 1733/2974 clips (58 %) with a 1.91× vs. 2.36× amplification
difference (23 % relative) does not support "confirms the finding is
not an artefact." Soften to directional preservation.

**BEFORE**
```latex
The doubling effect is preserved: a substantially more capable LLM
still registers nearly twice as much semantic damage from
MetricGAN{+} as from unenhanced noise, confirming the finding is
not an artefact of Flash Lite's limited capacity.
```

**AFTER**
```latex
The doubling direction is preserved: a substantially more capable LLM
still registers nearly twice as much semantic damage from
MetricGAN{+} as from unenhanced noise (1.91$\times$ vs.\
2.36$\times$ under Flash~Lite), consistent with the core finding
being robust to classifier capacity, though a full five-condition
Pro replication remains future work.
```

---

## P2-C — Dereverb metric anomaly unexplained

**Issue.** Table 2 shows Dereverb SNR = −4.72 dB and SI-SDR = −23.04 dB
(worse than Echo (sim) on both signal-level metrics) yet ODR = 0.108,
the lowest of all conditions. The paper explains why AEC breaks
intrusive metrics but offers no equivalent explanation for Dereverb.
A reviewer will ask why WPE produces negative SNR against a clean reference.
Add one sentence in the AEC decoupling paragraph that also covers Dereverb.

**BEFORE**
```latex
Note that SI-SDR, SRMR, and STOI are not designed for AEC evaluation, where the output is intentionally gain- and phase-modified relative to the near-end reference. Taken together with the MetricGAN{+} case, these results show
that metrics from the same category (intrusive signal-level:
SNR, SI-SDR) can disagree sharply about the direction of an
enhancement's effect, independent of its actual impact on the
downstream task.
```

**AFTER**
```latex
Note that SI-SDR, SRMR, and STOI are not designed for AEC evaluation,
where the output is intentionally gain- and phase-modified relative to
the near-end reference.
A parallel argument applies to Dereverb: intrusive metrics compare
the WPE output against the dry clean recording, but the dereverberated
signal is temporally smeared relative to that reference, producing
severely negative SNR ($-$4.72\,dB) and SI-SDR ($-$23.04\,dB) even
though ODR is the lowest of all conditions (0.108), again showing
that signal-level metrics do not track LLM task failure.
Taken together with the MetricGAN{+} case, these results show
that metrics from the same category (intrusive signal-level:
SNR, SI-SDR) can disagree sharply about the direction of an
enhancement's effect, independent of its actual impact on the
downstream task.
```

---

## P3-A — Contribution (4) headline overstates predictive utility

**Issue.** The contribution as written implies metrics are useful ODR
predictors "across conditions." The body text immediately qualifies
this as a between-condition separation artefact with no per-clip power.
The contribution list is what reviewers read first; it should carry
the caveat.

**BEFORE**
```latex
\item We demonstrate that standard audio quality metrics correlate
  with ODR across conditions but fail to predict which individual
  clips will diverge within a given SE pipeline.
```

**AFTER**
```latex
\item We show that standard audio quality metrics reflect
  between-condition ODR separation but have near-zero predictive
  power for which individual clips will diverge once the enhancement
  condition is fixed, making them unreliable per-clip early-warning
  signals.
```

---

## P3-B — "Central finding" fragment disrupts flow

**Issue.** "This is the central finding." is a one-sentence paragraph
fragment that reads as an announcement rather than analysis. The
following sentence already states the substance.

**BEFORE**
```latex
Under Whisper, MetricGAN{+} produces ODR of 0.318
(CI\,[0.301,\,0.335]), more than doubling the unenhanced noisy
baseline (0.135, CI\,[0.123,\,0.147]).
This is the central finding. MetricGAN{+} substantially improves PESQ
and STOI yet introduces additional LLM semantic damage beyond what
noise alone causes, demonstrating that perceptual quality improvement
and LLM task performance are not aligned.
```

**AFTER**
```latex
Under Whisper, MetricGAN{+} produces ODR of 0.318
(CI\,[0.301,\,0.335]), more than doubling the unenhanced noisy
baseline (0.135, CI\,[0.123,\,0.147]).
MetricGAN{+} raises PESQ yet simultaneously worsens STOI and more
than doubles ODR, demonstrating that perceptual quality improvement
and LLM task performance are not aligned.
```

---

## P3-C — MetricGAN+ PESQ improvement overstated without absolute context

**Issue.** PESQ rises from 1.76 to 1.94 — both values are in the
"bad" range of the 1–4.5 scale. Describing the gain without this
context overstates the enhancement's perceptual benefit.

**BEFORE**
```latex
MetricGAN{+} raises PESQ from 1.76 to 1.94
relative to Noisy ($\Delta\!=\!+0.18$), a modest
improvement, while simultaneously \emph{reducing}
STOI from 0.87 to 0.80 ($\Delta\!=\!-0.07$), and
more than doubling ODR from 0.135 to 0.318.
This triple divergence is particularly damaging for
practitioners because the metric that MetricGAN{+} is
explicitly trained to maximize (PESQ) improves, the
intelligibility proxy (STOI) worsens, and the
downstream LLM task failure rate doubles.
```

**AFTER**
```latex
MetricGAN{+} raises PESQ from 1.76 to 1.94
relative to Noisy ($\Delta\!=\!+0.18$), a modest gain that keeps
both values within the ``bad'' range of the 1--4.5 scale,
while simultaneously \emph{reducing}
STOI from 0.87 to 0.80 ($\Delta\!=\!-0.07$) and
more than doubling ODR from 0.135 to 0.318.
This triple divergence is particularly damaging for
practitioners: the metric MetricGAN{+} is explicitly trained to
maximise (PESQ) improves, the intelligibility proxy (STOI)
worsens, and the downstream LLM task failure rate doubles.
```

---

## P3-D — ODR offline-only limitation missing from Deployment Implications

**Issue.** The Deployment Implications section recommends ODR "as a
standard evaluation criterion" but does not state it requires a clean
reference and is therefore an offline development-time tool. This
caveat is only in Limitations.

**BEFORE**
```latex
\paragraph{WER is an insufficient SE evaluation metric.}
Our results show WER can overestimate LLM damage (MetricGAN{+}:
ODR\,=\,0.318, Gap\,=\,$-0.296$) or saturate while
ODR remains catastrophic (Echo (sim): WER\,=\,1.448,
ODR\,=\,0.836).
We recommend ODR alongside WER as a standard evaluation criterion
for any SE front-end deployed before an LLM, with particular urgency
for CTC-based ASR systems where our results suggest ODR is
systematically higher.
```

**AFTER**
```latex
\paragraph{WER is an insufficient SE evaluation metric.}
Our results show WER can overestimate LLM damage (MetricGAN{+}:
ODR\,=\,0.318, Gap\,=\,$-0.296$) or saturate while
ODR remains catastrophic (Echo (sim): WER\,=\,1.448,
ODR\,=\,0.836).
We recommend ODR alongside WER as an offline evaluation criterion
for comparing SE configurations before deployment; like PESQ and
STOI, it requires a clean reference recording and is therefore
a development-time tool rather than an online per-utterance monitor.
This recommendation carries particular urgency for CTC-based ASR
systems, where our results suggest ODR is systematically higher.
```

---

## P3-E — Conclusion mixes recommendation and release in one sentence

**Issue.** The final conclusion sentence conflates a methodological
recommendation with a software release statement.

**BEFORE**
```latex
We recommend ODR alongside WER as a standard criterion for SE
front-ends before an LLM, and release the full pipeline for evaluation across
additional LLMs, tasks, and SE systems.
```

**AFTER**
```latex
We recommend ODR alongside WER as an offline evaluation criterion
for SE front-ends deployed before an LLM.
The full pipeline is publicly released to support evaluation across
additional LLMs, tasks, and SE systems.
```

---

## P3-F — Abstract "significant ODR" is ambiguous

**Issue.** "Every condition produces significant ODR (p < 0.001)" could
mean the condition's ODR is significantly above zero, or that pairwise
comparisons are significant. Clarify the test type.

**BEFORE**
```latex
Every condition produces significant ODR ($p < 0.001$).
```

**AFTER**
```latex
Every condition produces ODR significantly above zero
($p < 0.001$, binomial test).
```

---

## P3-G — AEC "quality optimization" vague in Deployment Implications

**Issue.** "Not just a quality optimization" is imprecise. The intent
is that AEC is often treated as optional polish rather than a
functional requirement.

**BEFORE**
```latex
In any conferencing deployment, functional AEC must be treated as a
prerequisite for LLM integration, not just a quality optimization.
```

**AFTER**
```latex
In any conferencing deployment, functional AEC must be treated as a
prerequisite for LLM integration, not an optional quality refinement
that can be deferred when latency or cost is constrained.
```

---

## Summary table

| ID  | Priority | Location                        | Issue                                          |
|-----|----------|---------------------------------|------------------------------------------------|
| P1-A | **P1** | Sec. 4.1 body                  | Regression range misses Dereverb 1.9:1 minimum |
| P1-B | **P1** | Fig. 2 caption                 | "All conditions" LLM robustness overgeneralises |
| P2-A | **P2** | Table 2 + caption              | Raw vs. capped WER trap                        |
| P2-B | **P2** | Sec. 4.4 LLM Robustness        | Conclusion too strong for 58 % replication     |
| P2-C | **P2** | Sec. 4.1 AEC paragraph         | Dereverb metric anomaly unexplained            |
| P3-A | P3     | Contribution list item 4       | Headline overstates metric predictive utility  |
| P3-B | P3     | Sec. 4.1 MetricGAN+ paragraph  | "Central finding" fragment — cut or merge      |
| P3-C | P3     | Sec. 4.1 MetricGAN+ paragraph  | PESQ gain context missing (stays in "bad" range)|
| P3-D | P3     | Sec. 5 Deployment Implications | ODR offline-only caveat missing                |
| P3-E | P3     | Conclusion                     | Recommendation and release in one sentence     |
| P3-F | P3     | Abstract                       | "Significant ODR" ambiguous                    |
| P3-G | P3     | Sec. 5 Deployment Implications | "Quality optimization" vague                   |
