$ErrorActionPreference = "Continue"

$PaperDir = Join-Path $PSScriptRoot "paper"
$TexFile  = "emnlp2026_industry"
$TexPath  = Join-Path $PaperDir "$TexFile.tex"
$LogPath  = Join-Path $PaperDir "$TexFile.log"
$PdfPath  = Join-Path $PaperDir "$TexFile.pdf"

Write-Host "==> Pass 1: pdflatex" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -output-directory $PaperDir $TexPath | Out-Null

Write-Host "==> bibtex" -ForegroundColor Cyan
Push-Location $PaperDir
bibtex $TexFile | Out-Null
Pop-Location

Write-Host "==> Pass 2: pdflatex" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -output-directory $PaperDir $TexPath | Out-Null

Write-Host "==> Pass 3: pdflatex" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -output-directory $PaperDir $TexPath | Out-Null

$errors = Select-String -Path $LogPath -Pattern "^!" -ErrorAction SilentlyContinue
if ($errors) {
    Write-Host "`nERRORS:" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "  $($_.Line)" -ForegroundColor Red }
} else {
    $size = [math]::Round((Get-Item $PdfPath).Length / 1KB)
    Write-Host "`nDone - $TexFile.pdf ($size KB, no errors)" -ForegroundColor Green
}
