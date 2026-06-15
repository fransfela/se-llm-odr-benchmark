"""Generate LaTeX tables for the EMNLP 2026 paper."""

from pathlib import Path

import pandas as pd

CONDITION_LABELS = {
    "clean":        "Clean",
    "noisy":        "Noisy",
    "ns_metricgan": "MetricGAN+",
    "aec_sim":      "Echo (sim)",
    "aec_full":     "Echo + DTLN-AEC",
    "dereverb":     "Dereverb",
}

METRIC_TYPE = {
    "pesq":        "Intrusive",
    "stoi":        "Intrusive",
    "warpq":       "Intrusive",
    "si_sdr":      "Intrusive",
    "snr":         "Intrusive",
    "srmr":        "Non-intrusive",
    "squim_mos":   "Non-intrusive",
    "dnsmos_ovrl": "Non-intrusive",
    "nisqa":       "Non-intrusive",
    "scoreq_nr":   "Non-intrusive",
}

METRIC_DISPLAY = {
    "pesq":        "PESQ",
    "stoi":        "STOI",
    "warpq":       "WARP-Q",
    "si_sdr":      "SI-SDR",
    "snr":         "SNR",
    "srmr":        "SRMR",
    "squim_mos":   "SQUIM-MOS",
    "dnsmos_ovrl": "DNSMOS",
    "nisqa":       "NISQA",
    "scoreq_nr":   "SCOREQ",
}


def _save(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def generate_table_conditions():
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{p{1.8cm}p{1.8cm}p{3.4cm}}
\toprule
\textbf{Condition} & \textbf{Category} & \textbf{Description} \\
\midrule
Clean      & Reference       & Original SLURP recording \\
Noisy      & Degraded        & DNS noise, SNR\,=\,10\,dB \\
MetricGAN+ & Discrim.\ SE    & GAN noise suppression \\
Echo (AEC) & Echo            & Simulated echo, no cancellation \\
\texttt{aec\_full} & AEC & DTLN-AEC on echo simulation \\
Dereverb   & Dereverberation & WPE (5 iterations) \\
\bottomrule
\end{tabular}
\caption{Enhancement conditions applied to 2{,}974 SLURP test
recordings. All degradations simulated using DNS Challenge corpora.}
\label{tab:conditions}
\end{table}"""
    _save("paper/tables/table1_conditions.tex", tex)


def generate_table_results(divergence_csv):
    df = pd.read_csv(divergence_csv)
    df = df[df.condition != "clean"].copy()
    df = df.sort_values("odr", ascending=False).reset_index(drop=True)

    rows = []
    max_odr_idx = df.odr.idxmax()
    for i, row in df.iterrows():
        label = CONDITION_LABELS.get(row.condition, row.condition)
        odr = f"{row.odr:.3f}"
        ci = f"[{row.odr_ci_low:.3f},\\,{row.odr_ci_high:.3f}]"
        wer_val = min(row.wer_mean, 1.0)
        wer = f"{wer_val:.3f}"
        gap = f"{row.odr - wer_val:+.3f}"
        wer_note = "$^*$" if row.wer_mean > 1.0 else ""
        line = (f"{label} & {odr} & "
                f"{ci} & {wer}{wer_note} & {gap}")
        if i == max_odr_idx:
            line = "\\textbf{" + line.replace(" & ", "} & \\textbf{") + "}"
        rows.append(line + r" \\")

    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
\textbf{{Condition}} & \textbf{{ODR}} & \textbf{{95\,\%\,CI}} &
\textbf{{WER}} & \textbf{{Gap}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Output Divergence Rate (ODR) per condition ($N=2{{,}}974$).
WER from Whisper large-v3, capped at 1.0.
$\text{{Gap}} = \text{{ODR}} - \min(\overline{{\text{{WER}}}},\,1.0)$.
95\,\% CI from bootstrap ($B=10{{,}}000$, seed\,=\,42).
Highest ODR in \textbf{{bold}}.
$^*$Raw mean $\text{{WER}} = 1.448$; see Appendix~\ref{{app:wer}}.}}
\label{{tab:results}}
\end{{table}}"""
    _save("paper/tables/table2_results.tex", tex)


def generate_table_correlations(correlations_csv):
    df = pd.read_csv(correlations_csv)
    df = df.sort_values("spearman_rho").reset_index(drop=True)
    n_metrics = len(df)

    def sig_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    rows = []
    max_abs_idx = df.spearman_rho.abs().idxmax()
    for i, row in df.iterrows():
        name = METRIC_DISPLAY.get(row.metric, row.metric)
        mtype = METRIC_TYPE.get(row.metric, "---")
        rho_s = f"{row.spearman_rho:+.3f}"
        rho_p = f"{row.pearson_r:+.3f}"
        ci = f"[{row.spearman_ci_low:+.3f},\\,{row.spearman_ci_high:+.3f}]"
        sig = sig_stars(row.spearman_p_corrected)
        line = (f"{name} & {mtype} & "
                f"{rho_s} & {rho_p} & {ci} & {sig}")
        if i == max_abs_idx:
            line = "\\textbf{" + line.replace(" & ", "} & \\textbf{") + "}"
        rows.append(line + r" \\")

    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[h]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llrrrr}}
\toprule
\textbf{{Metric}} & \textbf{{Type}} &
\textbf{{Spear.\ $\rho$}} & \textbf{{Pears.\ $r$}} &
\textbf{{95\,\%\,CI ($\rho$)}} & \textbf{{Sig.}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Spearman $\rho$ and Pearson $r$ between quality
metric scores and binary LLM output divergence (ODR)
across all clips and conditions. Bonferroni correction
applied ($n\!=\!{n_metrics}$). Highest $|\rho|$ in \textbf{{bold}}.
*** $p<0.001$, ** $p<0.01$, * $p<0.05$, n.s.
Correlations pool clips across all conditions; see
\\S\\ref{{sec:results}} for within-condition analysis showing
substantially weaker per-clip predictive power.}}
\label{{tab:correlations}}
\end{{table}}"""
    _save("paper/tables/table5_correlations.tex", tex)


def generate_table_metrics(metrics_csv='results/metrics_core.csv'):
    df = pd.read_csv(metrics_csv)
    conditions = ['clean', 'noisy', 'ns_metricgan', 'aec_sim', 'dereverb']
    df = df[df.condition.isin(conditions)]
    summary = df.groupby('condition')[
        ['pesq', 'stoi', 'snr', 'si_sdr', 'srmr', 'squim_mos']].mean()
    summary = summary.loc[conditions]

    noisy_pesq = summary.loc['noisy', 'pesq']
    ns_pesq = summary.loc['ns_metricgan', 'pesq']
    noisy_stoi = summary.loc['noisy', 'stoi']
    ns_stoi = summary.loc['ns_metricgan', 'stoi']
    pesq_delta = ns_pesq - noisy_pesq
    stoi_delta = ns_stoi - noisy_stoi

    rows = []
    for cond in conditions:
        row = summary.loc[cond]
        label = CONDITION_LABELS.get(cond, cond)
        rows.append(
            f'{label} & '
            f'{row.pesq:.2f} & {row.stoi:.2f} & '
            f'{row.snr:.2f} & {row.si_sdr:.2f} & '
            f'{row.srmr:.2f} & {row.squim_mos:.2f} \\\\'
        )

    body = '\n'.join(rows)
    caption = (
        "Mean audio quality metric scores per condition. "
        f"MetricGAN{{+}} improves PESQ slightly "
        f"(${noisy_pesq:.2f}\\!\\rightarrow\\!{ns_pesq:.2f}$, "
        f"$\\Delta\\!=\\!{pesq_delta:+.2f}$) "
        f"but \\emph{{reduces}} STOI "
        f"(${noisy_stoi:.2f}\\!\\rightarrow\\!{ns_stoi:.2f}$, "
        f"$\\Delta\\!=\\!{stoi_delta:+.2f}$) "
        "relative to Noisy, yet more than doubles ODR "
        "($0.135\\!\\rightarrow\\!0.318$). "
        "This triple decoupling\\,---\\,perceptual metrics moving in "
        "opposite directions while LLM damage increases\\,---\\,"
        "underscores that optimising for a single quality "
        "metric provides no guarantee of downstream task "
        "fidelity. Echo + DTLN-AEC metrics not computed "
        "(\\texttt{aec\\_full} excluded from "
        "\\texttt{metrics\\_core.csv})."
    )
    tex = (
        "\\begin{table}[t]\n"
        "\\centering\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "\\textbf{Condition} & \\textbf{PESQ} & \\textbf{STOI} &\n"
        "\\textbf{SNR} & \\textbf{SI-SDR} &\n"
        "\\textbf{SRMR} & \\textbf{SQUIM} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:metrics}\n"
        "\\end{table}"
    )
    _save('paper/tables/table2_metrics.tex', tex)


def generate_table_domain_odr(domain_csv):
    df = pd.read_csv(domain_csv)
    conditions = ["noisy", "ns_metricgan", "aec_sim", "aec_full", "dereverb"]
    present = [c for c in conditions if c in df.columns]
    df = df.sort_values("mean_odr", ascending=False).reset_index(drop=True)

    # Top 8 + bottom 3 (avoid duplicates if < 11 domains)
    n = len(df)
    if n <= 11:
        selected = df
    else:
        top = df.head(8)
        bottom = df.tail(3)
        selected = pd.concat([top, bottom]).drop_duplicates()

    DOMAIN_LABELS = {
        "iot_wemo": "IoT (WeMo)",
        "iot_hue": "IoT (Hue)",
        "audio_volume": "Volume",
        "qa": "QA",
    }

    rows = []
    for _, row in selected.iterrows():
        domain = str(row.domain)
        domain = DOMAIN_LABELS.get(domain, domain.replace("_", " ").capitalize())
        vals = " & ".join(f"{row[c]:.3f}" if c in row.index else "---"
                          for c in present)
        mean = f"{row.mean_odr:.3f}"
        rows.append(f"{domain} & {vals} & {mean} \\\\")
        # Add midrule separator between top and bottom groups
        if n > 11 and _ == top.index[-1]:
            rows.append("\\midrule")

    body = "\n".join(rows)
    cond_headers = " & ".join(
        f"\\textbf{{{CONDITION_LABELS.get(c, c)}}}"
        for c in present)
    n_cols = len(present) + 2
    col_spec = "l" + "r" * (len(present) + 1)
    tex = rf"""\begin{{table}}[h]
\centering\small
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{{col_spec}}}
\toprule
\textbf{{Domain}} & {cond_headers} & \textbf{{Mean}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}%
\caption{{Per-domain mean ODR across enhancement conditions.
Top 8 most vulnerable and bottom 3 most robust domains shown,
sorted by mean ODR descending.}}
\label{{tab:domain_odr}}
\end{{table}}"""
    _save("paper/tables/table5_domain_odr.tex", tex)


def generate_table_condspecific_metrics():
    from scipy import stats

    aecmos = pd.read_csv('results/metrics_aecmos.csv')
    clip = pd.read_csv('results/clip_level_divergence.csv')

    aec_div = clip[clip.condition == 'aec_sim'][['clip_id', 'diverged']]
    aec_div['clip_id'] = aec_div['clip_id'].astype(str)
    aecmos['clip_id'] = aecmos['clip_id'].astype(str)
    merged = aec_div.merge(aecmos, on='clip_id')
    rho_echo, p_echo = stats.spearmanr(merged.echo_mos, merged.diverged)
    rho_deg, p_deg = stats.spearmanr(merged.deg_mos, merged.diverged)

    print(f'AECMOS echo_mos vs ODR (aec_sim): rho={rho_echo:.3f} p={p_echo:.4f}')
    print(f'AECMOS deg_mos  vs ODR (aec_sim): rho={rho_deg:.3f}  p={p_deg:.4f}')
    return rho_echo, rho_deg


def generate_table_examples(clip_ids):
    slug = 'google_gemini-2_5-flash-lite'
    clean_int = pd.read_csv(f'results/intents_{slug}_clean.csv')
    ns_int    = pd.read_csv(f'results/intents_{slug}_ns_metricgan.csv')
    clean_trans = pd.read_csv('results/transcripts_clean.csv')
    ns_trans    = pd.read_csv('results/transcripts_ns_metricgan.csv')

    for df in [clean_int, ns_int, clean_trans, ns_trans]:
        df['clip_id'] = df['clip_id'].astype(str)

    rows = []
    for i, cid in enumerate(clip_ids):
        ci = clean_int[clean_int.clip_id==cid].iloc[0]
        ni = ns_int[ns_int.clip_id==cid].iloc[0]
        ct = clean_trans[clean_trans.clip_id==cid].iloc[0]
        nt = ns_trans[ns_trans.clip_id==cid].iloc[0]

        def esc(s):
            return (str(s).replace('&','\\&').replace('_','\\_')
                    .replace('%','\\%'))

        entry = (
            f'\\multirow{{2}}{{*}}{{{cid}}} & Clean & '
            f'``{esc(ct.whisper_text)}\'\' & '
            f'\\texttt{{{esc(ci.predicted_intent)}}} \\\\\n'
            f' & MetricGAN{{+}} & '
            f'``{esc(nt.whisper_text)}\'\' & '
            f'\\texttt{{{esc(ni.predicted_intent)}}} \\\\'
        )
        if i < len(clip_ids) - 1:
            entry += '\n\\midrule'
        rows.append(entry)

    body = '\n'.join(rows)
    tex = rf'''\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lp{{1.3cm}}p{{3.3cm}}l}}
\toprule
\textbf{{Clip}} & \textbf{{Cond.}} & \textbf{{Transcript}} &
\textbf{{Intent}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Example clips where MetricGAN{{+}} causes intent
collapse to catch-all categories. Transcription remains
fluent but loses domain-specific lexical cues, causing the
LLM to default to generic intents.}}
\label{{tab:examples}}
\end{{table}}'''
    _save('paper/tables/table3_examples.tex', tex)
    print('table_examples.tex written')


if __name__ == "__main__":
    generate_table_conditions()
    print("table1_conditions.tex written")
    print("Run generate_table_results + generate_table_correlations")
    print("after analysis scripts have produced results CSVs.")
