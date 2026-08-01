# Author Response -- EMNLP 2026 Industry Track Submission #461

We thank all three reviewers for their constructive feedback. Below we
address each concern and describe revisions made to the paper.

---

## Common concern: Single downstream LLM (all reviewers)

We acknowledge this as the primary limitation. To test whether our
findings depend on classifier capacity, we ran a partial replication
with Gemini 2.5 Pro (a reasoning-enabled model, substantially more
capable than Flash Lite) on the two central conditions:

| Condition    | Flash Lite ODR | Pro ODR | Ratio vs. Noisy |
|--------------|---------------|---------|-----------------|
| Noisy        | 0.135         | 0.154   | 1.0x (baseline) |
| MetricGAN+   | 0.318         | 0.294   | 1.91x           |

The central finding replicates: MetricGAN+ nearly doubles ODR versus
unenhanced noise under both models (2.36x with Flash Lite, 1.91x with
Pro). The effect is not an artifact of limited classifier capacity.
Full Pro results (clean, noisy, partial MetricGAN+) are available in
our repository. We have updated the Limitations section accordingly.

We note that our benchmark already tests robustness across two
architecturally distinct ASR models (encoder-decoder Whisper and
CTC-based wav2vec2), producing identical condition rankings
(Spearman rho = 1.0). Combined with the LLM replication above, this
provides evidence across three axes of the pipeline (ASR architecture,
ASR model, LLM capacity).

---

## Reviewer zPYo

**ODR conflates harmful divergence with beneficial correction.**

We agree this is methodologically important. In the revised paper
(Section 4.1, "Correct vs. harmful divergence"), we decompose each
condition's divergent clips against SLURP ground-truth intents into
corrections, regressions, and both-wrong shifts. Regressions
outnumber corrections in every condition: 3.3:1 (Noisy), 7.6:1
(MetricGAN+), 23.1:1 (Echo sim), 10.0:1 (Echo + DTLN-AEC), 1.9:1
(Dereverb). This confirms that ODR predominantly captures harmful
divergence. Full breakdown in Appendix Table 5.

**Correlation analysis methodological concerns (binary ODR, pooled
across conditions).**

We address this directly in Section 4.2: "Pooled correlations
overstate predictive value." Recomputing within each condition
separately yields |rho| <= 0.34 (vs. pooled -0.467), confirming that
no metric provides meaningful per-clip signal once the enhancement
condition is fixed. We additionally report PESQ false-negative and
false-positive rates (Appendix Table 6): under Echo (sim), 82.3% of
above-median-PESQ clips still diverge; under Noisy/Dereverb, 84--88%
of below-median-PESQ clips do not diverge.

**Simulated conditions only.**

Acknowledged as a limitation. We note that our conditions use DNS
Challenge noise and RIRs, the standard evaluation corpora used by the
SE community (URGENT Challenge, DNS Challenge, INTERSPEECH 2025 SE
benchmarks), making our results directly comparable to published SE
evaluations. Real-device compound distortions remain future work.

**Limited SE coverage.**

Acknowledged. We test one discriminative suppressor (MetricGAN+),
which is representative of the most widely deployed SE class
(GAN-based perceptual optimizers). Diffusion-based and commercial
systems remain future work (noted in Limitations).

**Named entities and proper nouns.**

SLURP intent labels are domain/action pairs (e.g.,
"transport_traffic") that typically survive entity-level ASR
substitutions (e.g., mishearing a city name). ODR may therefore
understate SE impact on entity-dependent tasks such as slot filling.
We have added this to Limitations ("Entity-level effects") and flag
slot-level F1 evaluation as future work.

---

## Reviewer i7kA

**Where do quality metrics fail (false positives and false
negatives)?**

Added in Section 4.2 ("Where PESQ fails as an early-warning signal")
and Appendix Table 6. Key finding: PESQ failure mode is
condition-dependent. Under Echo (sim), the false-negative rate is
82.3% (PESQ gives no safety margin where damage is worst). Under
Dereverb and Noisy, the false-positive rate is 84--88% (PESQ
over-warns on clips the LLM handles correctly). No single PESQ
threshold separates safe from unsafe clips across conditions.

**ODR requires clean speech reference, limiting deployment use.**

Correct. Like PESQ and STOI, ODR is an offline development-time
criterion for comparing SE front-ends before deployment, not an
online monitor. We have clarified this in Limitations
("Clean-reference requirement") and point to reference-free proxies
(transcript stability, ASR confidence) as future work.

---

## Reviewer mUUj

**Table 2 caption mentions ODR but table does not include it.**

Fixed. We have added an ODR column to Table 2 with per-condition
Whisper ODR values.

**No conclusion section.**

Added (Section 6).

**No encoder-based classification.**

We clarify that the paper tests two architecturally distinct ASR
front-ends: encoder-decoder (Whisper) and encoder-based CTC
(wav2vec2). The reviewer may be referring to encoder-based intent
classifiers (e.g., BERT joint intent/slot models). We have added
this as a Limitation ("Classifier architecture") and note that
conclusions could differ under such architectures.

**Single language model.**

Addressed above (common concern). Preliminary Gemini 2.5 Pro
replication confirms the central finding.

---

## Summary of revisions

- Added Conclusion section (Section 6)
- Added ODR column to Table 2
- Added "Correct vs. harmful divergence" paragraph (Section 4.1)
  with Appendix Table 5
- Added "Where PESQ fails" paragraph (Section 4.2) with Appendix
  Table 6
- Updated Limitations: Single LLM (with Pro replication numbers),
  clean-reference requirement, SE method coverage, classifier
  architecture, entity-level effects
- Gemini 2.5 Pro replication data available in repository
