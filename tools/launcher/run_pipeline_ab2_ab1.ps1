# run_ab2_then_ab1_reweight.ps1
# Phase 1: Wait for AB2 (dp=0.5 + dp=0.9) to finish
# Phase 2: Launch AB1 compression sweep with compression_weight=0.3
#
# Safe to run even if AB2 is already running — it will wait.

$Python = "C:\Python314\python.exe"
Set-Location "E:\projects\FLUED\FLUED"
$DataPath = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt"

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONUNBUFFERED = "1"

$LogFile = "pipeline_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
function Write-Log { param($msg) $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; "$ts  $msg" | Tee-Object -FilePath $LogFile -Append }

# =========================================================================
# Phase 0: Wait for any running E1 training to finish
# =========================================================================
Write-Log "=== PIPELINE: AB2 then AB1 (weight=0.3) ==="
Write-Log "Phase 0: Checking for running training..."

$running = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq 'C:\Python314\python.exe' }
if ($running) {
    $pids = $running | ForEach-Object { $_.Id }
    Write-Log "Found running python PIDs: $($pids -join ', '). Waiting..."
    while (Get-Process -Id $pids -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 120
    }
    Write-Log "All training processes ended. Cooling down 30s..."
    Start-Sleep -Seconds 30
} else {
    Write-Log "No running training detected."
}

# =========================================================================
# Phase 1: AB2 denoise (dp=0.5 + dp=0.9) if not already complete
# =========================================================================
Write-Log "Phase 1: AB2 denoise sweep check"

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

$DenoiseProbs = @(0.5, 0.9)
foreach ($dp in $DenoiseProbs) {
    $Name = "ab2_dp$($dp.ToString('0.0').Replace('.',''))"
    $CkptDir = "checkpoints/$Name"
    $CkptFile = "$CkptDir/e1_step30000.pt"

    if (Test-Path $CkptFile) {
        Write-Log "AB2 dp=$dp: already has step30000 checkpoint, skipping."
        continue
    }
    if (-not (Test-Path $CkptDir)) { New-Item -ItemType Directory -Path $CkptDir | Out-Null }

    Write-Log "AB2 dp=$dp: STARTING → $CkptDir"
    $free = [math]::Round((Get-PSDrive E).Free/1GB, 1)
    Write-Log "  E盘 free: ${free}GB"

    & $Python $BaseArgs --denoise-prob $dp --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        Add-Content -Path "$CkptDir/run.log" -Value $line
    }
    Write-Log "AB2 dp=$dp DONE (exit=$LASTEXITCODE)"
}

# =========================================================================
# Phase 2: AB1 compression sweep with weight=0.3
# =========================================================================
Write-Log "Phase 2: AB1 compression sweep (weight=0.3)"

$CompArgs = @(
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
    "--compression-weight", "0.3",
    "--entropy-warmup-steps", "10000",
    "--ckpt-every", "1000",
    "--seed", "42"
)

$CompressionTargets = @(0.20, 0.30, 0.45, 0.60)
foreach ($tc in $CompressionTargets) {
    $Name = "ab1_w03_tc$($tc.ToString('0.00').Replace('.',''))"
    $CkptDir = "checkpoints/$Name"
    $CkptFile = "$CkptDir/e1_step30000.pt"

    if (Test-Path $CkptFile) {
        Write-Log "AB1(w=0.3) tc=$tc: already has step30000 checkpoint, skipping."
        continue
    }
    if (-not (Test-Path $CkptDir)) { New-Item -ItemType Directory -Path $CkptDir | Out-Null }

    Write-Log "AB1(w=0.3) tc=$tc: STARTING → $CkptDir"
    $free = [math]::Round((Get-PSDrive E).Free/1GB, 1)
    Write-Log "  E盘 free: ${free}GB"

    & $Python $CompArgs --target-compression $tc --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        Add-Content -Path "$CkptDir/run.log" -Value $line
    }
    Write-Log "AB1(w=0.3) tc=$tc DONE (exit=$LASTEXITCODE)"
}

Write-Log "=== PIPELINE COMPLETE ==="
$free = [math]::Round((Get-PSDrive E).Free/1GB, 1)
Write-Log "E盘 free: ${free}GB"
