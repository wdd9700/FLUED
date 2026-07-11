# run_ablations.ps1 — v2 E1 ablation suite (compression sweep + denoise sweep)
# Runs on local RTX 5080. Each ablation = 50K steps at ~13h.
#
# Usage: .\tools\launcher\run_ablations.ps1

$Python = "python"
Set-Location "."
$DataPath = "data/corpus.txt"

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONUNBUFFERED = "1"

$BaseArgs = @(
    "-m", "flued.e1_stage_a",
    "--preset", "class300m_16gb",
    "--data-path", $DataPath,
    "--max-lines", "50000",
    "--max-eval-batches", "200",
    "--target-accuracy", "0.99",
    "--max-steps", "30000",
    "--lambda-var", "0.5",
    "--lambda-entropy", "0.05",
    "--lambda-utf8", "0.02",
    "--lambda-type", "0.05",
    "--compression-weight", "0.1",
    "--entropy-warmup-steps", "10000",
    "--ckpt-every", "1000",
    "--seed", "42"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " v2 E1 ABLATION SUITE — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================" -ForegroundColor Cyan

# ===========================================================================
# AB1: Compression Sweep
# ===========================================================================
$CompressionTargets = @(0.20, 0.30, 0.45, 0.60)
foreach ($tc in $CompressionTargets) {
    $Name = "ab1_tc$($tc.ToString('0.00').Replace('.',''))"
    $CkptDir = "checkpoints/$Name"
    Write-Host "`n--- AB1: target_compression=$tc → $CkptDir ---" -ForegroundColor Yellow

    & $Python $BaseArgs --target-compression $tc --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        Add-Content -Path "$CkptDir/run.log" -Value $line
    }
    Write-Host "AB1 tc=$tc DONE (exit=$LASTEXITCODE)" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})
}

# ===========================================================================
# AB2: Denoise Ratio Sweep
# ===========================================================================
$DenoiseProbs = @(0.3, 0.5, 0.9)
foreach ($dp in $DenoiseProbs) {
    $Name = "ab2_dp$($dp.ToString('0.0').Replace('.',''))"
    $CkptDir = "checkpoints/$Name"
    Write-Host "`n--- AB2: denoise_prob=$dp → $CkptDir ---" -ForegroundColor Yellow

    & $Python $BaseArgs --denoise-prob $dp --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        Add-Content -Path "$CkptDir/run.log" -Value $line
    }
    Write-Host "AB2 dp=$dp DONE (exit=$LASTEXITCODE)" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})
}

Write-Host "`n=== ALL ABLATIONS COMPLETE ===" -ForegroundColor Green
