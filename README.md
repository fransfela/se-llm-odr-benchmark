# Perceptually Better, Semantically Worse

Measuring Speech Enhancement Impact on LLM-Based Voice
Systems. Code and evaluation pipeline for the EMNLP 2026
Industry Track submission (anonymous, under review).

---

## Overview

Speech enhancement (SE) front-ends improve perceptual quality (PESQ, STOI)
but can silently damage the semantic content that downstream LLMs rely on.
This repository provides the code and evaluation pipeline for measuring
**Output Divergence Rate (ODR)**, the fraction of clips whose LLM-predicted
intent changes after enhancement relative to the clean baseline.

Key findings (5 conditions, 2,974 SLURP test clips, 77 intents):

- MetricGAN+ more than doubles ODR (0.318 vs 0.135) despite improving PESQ.
- Unmitigated echo reaches ODR 0.836 through wrong-speaker transcription.
- PESQ achieves the strongest pooled correlation with ODR (Spearman rho = -0.467),
  but within any single condition no metric exceeds |rho| = 0.34.
- Findings replicate across ASR architectures (Whisper large-v3, wav2vec2-large).

## Repository structure

```
src/
  enhancement/    # Simulate noise, reverb, echo; apply MetricGAN+, WPE, DTLN-AEC
  asr/            # Whisper and wav2vec2 transcription
  nlp/            # LLM intent classification (OpenAI-compatible API)
  metrics/        # PESQ, STOI, SNR, SI-SDR, SRMR, SQUIM-MOS
  analysis/       # ODR computation, correlations, error taxonomy
  visualization/  # Figures and tables for the paper
scripts/          # Entry-point scripts for each pipeline stage
configs/          # YAML configuration for conditions, metrics
tests/            # Sanity checks for labels, metrics, pipeline integrity
paper/            # LaTeX source, figures, tables
```

## Requirements

Python 3.10+ with CUDA. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the pipeline

Each stage reads from the previous stage's output and appends results
(idempotent; already-processed clips are skipped).

### 1. Simulate acoustic conditions

```bash
python -m src.enhancement.run_all_conditions
```

Generates `data/enhanced/{condition}/` directories from clean SLURP audio
using DNS Challenge noise and RIR corpora.

### 2. ASR transcription

```bash
python -m src.asr.transcribe_whisper          # Whisper large-v3
python scripts/run_wav2vec2_pipeline.py        # wav2vec2-large (secondary)
```

Outputs `results/transcripts_{condition}.csv`.

### 3. Intent classification

```bash
python -m src.nlp.classify_intent_gpt         # Primary LLM classifier
```

Requires `LLM_API_KEY` and `LLM_MODEL` environment variables (or `.env` file).
Uses any OpenAI-compatible API endpoint. Outputs `results/intents_{model}_{condition}.csv`.

### 4. Divergence and correlation analysis

```bash
python -m src.analysis.compute_divergence     # ODR per condition
python -m src.analysis.compute_correlations   # Metric-ODR correlations
python scripts/compute_extended_correlations.py
python scripts/compute_odr_wav2vec2.py        # wav2vec2 ODR + cross-model comparison
```

### 5. Audio quality metrics

```bash
python -m src.metrics.run_core_metrics        # 6 core metrics (PESQ, STOI, SNR, SI-SDR, SRMR, SQUIM)
```

### 6. Figures and tables

```bash
python -m src.visualization.plot_fig1_odr
python -m src.visualization.plot_fig2_wer_gap
python -m src.visualization.plot_fig3_correlations
python -m src.visualization.generate_tables
```

## Data

The SLURP test set (2,974 clips) and DNS Challenge noise/RIR corpora are
not included due to licensing. Download from:

- **SLURP**: https://github.com/pswietojanski/slurp
- **DNS Challenge**: https://github.com/microsoft/DNS-Challenge

Place files under `data/raw/slurp/test/` and `data/raw/dns/` respectively.

## Conditions

| Condition | Description |
|-----------|-------------|
| Clean | Original SLURP recording (reference) |
| Noisy | DNS noise at SNR = 10 dB |
| MetricGAN+ | GAN-based noise suppression |
| Echo (sim) | Simulated echo, no cancellation |
| Echo + DTLN-AEC | DTLN-AEC echo cancellation |
| Dereverb | WPE dereverberation (5 iterations) |

## Citation

```bibtex
% Citation to be added upon acceptance.
```

## License

Code released under the MIT License. See [LICENSE](LICENSE).
