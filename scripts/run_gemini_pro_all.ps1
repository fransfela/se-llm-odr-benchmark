#!/usr/bin/env pwsh
# Run Gemini 2.5 Pro intent classification for all conditions sequentially.
# Usage: .\scripts\run_gemini_pro_all.ps1

$env:LLM_MODEL = "gemini-2.5-pro"
$env:LLM_BASE_URL = "https://apim-llm-gateway.azure-api.net/gemini"

$conditions = @("noisy", "ns_metricgan", "aec_sim", "dereverb", "aec_full")

foreach ($cond in $conditions) {
    Write-Host "`n========== Starting $cond at $(Get-Date -Format 'HH:mm:ss') ==========" -ForegroundColor Cyan
    python -m src.nlp.classify_intent_gpt $cond
    Write-Host "========== Finished $cond at $(Get-Date -Format 'HH:mm:ss') ==========" -ForegroundColor Green
}

Write-Host "`nAll conditions complete at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
