$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$texBin = Join-Path $projectDir '.tools\tinytex\TinyTeX\bin\windows'
$paperDir = Join-Path $projectDir 'paper'
$pdfOutput = Join-Path $projectDir 'output\pdf'
New-Item -ItemType Directory -Force $pdfOutput | Out-Null
Push-Location $paperDir
try {
    & "$texBin\pdflatex.exe" -interaction=batchmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'First LaTeX pass failed; inspect paper/main.log' }
    & "$texBin\bibtex.exe" main
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed; inspect paper/main.blg' }
    foreach ($pass in 1..2) {
        & "$texBin\pdflatex.exe" -interaction=batchmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw 'LaTeX reference pass failed' }
    }
    foreach ($pass in 1..2) {
        & "$texBin\pdflatex.exe" -interaction=batchmode -halt-on-error supplementary.tex
        if ($LASTEXITCODE -ne 0) { throw 'Supplement LaTeX pass failed' }
    }
    Copy-Item main.pdf (Join-Path $pdfOutput 'SEGAN_manuscript_Elsevier.pdf') -Force
    Copy-Item supplementary.pdf (Join-Path $pdfOutput 'SEGAN_appendices_Elsevier.pdf') -Force
} finally { Pop-Location }
