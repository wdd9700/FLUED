# gpu_retrain_e1.ps1 — Full E1 retrain from scratch on GPU with tee logging
$Python = "python"
Set-Location "."

$DataPath = "data/corpus.txt"
$LogFile = "checkpoints/e1_retrain_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " GPU E1 RETRAIN — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host " Log: $LogFile"
Write-Host " Config: class300m_16gb, 100-step logging, overwrite ckpt"
Write-Host "============================================================" -ForegroundColor Cyan

$args = @(
    "-m", "flued.e1_stage_a",
    "--preset", "class300m_16gb",
    "--data-path", $DataPath,
    "--max-lines", "50000",
    "--max-eval-batches", "200",
    "--amp", "--amp-dtype", "fp16",
    "--target-accuracy", "1.0",
    "--max-steps", "50000",
    "--lambda-var", "0.5",
    "--lambda-entropy", "0.05",
    "--lambda-utf8", "0.02",
    "--lambda-type", "0.05",
    "--target-compression", "0.3",
    "--compression-weight", "0.1",
    "--entropy-warmup-steps", "10000",
    "--seed", "42",
    "--ckpt-every", "1000"
)

# Run with tee
& $Python $args 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Host ""
Write-Host "E1 retrain finished. Exit: $LASTEXITCODE" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})
