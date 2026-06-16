# run_e3_downstream.ps1 — E3: Downstream causal LM on RTX 5090
#
# Trains a causal Transformer LM (353M params) on FLUED v2 frozen encoder.
# Measures next-byte perplexity (bits-per-byte).
#
# Usage: powershell -File run_e3_downstream.ps1

param(
    [string]$DataPath = "./data/corpus_v3.txt",
    [string]$FluedCkpt = "./checkpoints/e1_v2_seed42/e1_step50000.pt",
    [int]$MaxLines = 50000,
    [int]$MaxSteps = 50000,
    [int]$BatchSize = 2
)

$Python = "python"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

$CkptDir = "checkpoints/e3_flued_v2"
New-Item -ItemType Directory $CkptDir -Force | Out-Null

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " E3: FLUED v2 Downstream LM on RTX 5090"
Write-Host " Config: 353M causal LM, frozen FLUED encoder"
Write-Host " Steps: $MaxSteps"
Write-Host "============================================================" -ForegroundColor Cyan

$Args = @(
    "-m", "flued.e3_train",
    "--model", "flued",
    "--flued-ckpt", $FluedCkpt,
    "--data-path", $DataPath,
    "--max-lines", "$MaxLines",
    "--max-steps", "$MaxSteps",
    "--batch-size", "$BatchSize",
    "--ckpt-dir", $CkptDir
)

& $Python $Args 2>&1 | ForEach-Object {
    $line = "$_"; Write-Host $line
    Add-Content -Path "$CkptDir/train.log" -Value $line
}

Write-Host "`nE3 Done. Exit: $LASTEXITCODE" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})
