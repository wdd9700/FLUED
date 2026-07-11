# run_ab2_when_free.ps1 — Wait for tc=0.60 to finish, then launch AB2 denoise sweep
# Background launcher; safe to run while tc=0.60 is training.

$Python = "python"
Set-Location "."
$DataPath = "data/corpus.txt"

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONUNBUFFERED = "1"

# Identify the running E1 training process (launched around 13:42 on 2026-06-11)
$TargetProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.StartTime -gt (Get-Date "2026-06-11 13:30:00")
}

if ($TargetProcs) {
    $Pids = $TargetProcs | ForEach-Object { $_.Id }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Waiting for python PIDs: $($Pids -join ', ') ..." -ForegroundColor Cyan
    while (Get-Process -Id $Pids -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Training process ended. Starting AB2 in 30s..." -ForegroundColor Green
    Start-Sleep -Seconds 30
} else {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] No active training process found. Starting AB2 immediately..." -ForegroundColor Yellow
}

# Base arguments (same as run_ablations.ps1)
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

# ===========================================================================
# AB2: Denoise Ratio Sweep
# ===========================================================================
$DenoiseProbs = @(0.3, 0.5, 0.9)
foreach ($dp in $DenoiseProbs) {
    $Name = "ab2_dp$($dp.ToString('0.0').Replace('.',''))"
    $CkptDir = "checkpoints/$Name"
    if (-not (Test-Path $CkptDir)) { New-Item -ItemType Directory -Path $CkptDir | Out-Null }

    Write-Host "`n--- AB2: denoise_prob=$dp -> $CkptDir ---" -ForegroundColor Yellow

    & $Python $BaseArgs --denoise-prob $dp --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        Add-Content -Path "$CkptDir/run.log" -Value $line
    }
    Write-Host "AB2 dp=$dp DONE (exit=$LASTEXITCODE)" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})
}

Write-Host "`n=== ALL AB2 ABLATIONS COMPLETE ===" -ForegroundColor Green
