# fill_gaps.ps1 — Multi-segment retrain to fill E1 log data gaps
# Runs AFTER current E1 (PID 62944) finishes.
# Uses existing checkpoints to minimize retrain steps.

$Python = "python"
$ProjectDir = "."
Set-Location $ProjectDir

$DataPath = "data/corpus.txt"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

$BaseArgs = @(
    "--preset", "class300m_16gb",
    "--data-path", $DataPath,
    "--max-lines", "50000",
    "--max-eval-batches", "200",
    "--amp", "--amp-dtype", "fp16",
    "--target-accuracy", "1.0",
    "--lambda-var", "0.5",
    "--lambda-entropy", "0.05",
    "--lambda-utf8", "0.02",
    "--lambda-type", "0.05",
    "--target-compression", "0.3",
    "--compression-weight", "0.1",
    "--ckpt-every", "1000"
)

function Run-Training($tag, $extraArgs, $maxSteps, $resumeFrom) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " [$tag] Starting — $(Get-Date -Format 'HH:mm:ss')"
    Write-Host " Max steps: $maxSteps"
    if ($resumeFrom) { Write-Host " Resume: $resumeFrom" }
    Write-Host "============================================================" -ForegroundColor Cyan
    
    $logFile = "checkpoints/gapfill_${tag}.log"
    $args = @("-m", "flued.e1_stage_a") + $BaseArgs + $extraArgs
    
    if ($maxSteps) { $args += @("--max-steps", [string]$maxSteps) }
    if ($resumeFrom) { $args += @("--resume", $resumeFrom) }
    
    # Save command for reference
    "$Python $args 2>&1" | Out-File $logFile -Encoding utf8
    
    # Run with tee to both console and file
    & $Python $args 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Path $logFile -Value $line
    }
    
    $exitCode = $LASTEXITCODE
    Write-Host "[$tag] Exit code: $exitCode" -ForegroundColor $(if($exitCode -eq 0){'Green'}else{'Yellow'})
    return $exitCode
}

# ===========================================================================
# Wait for current E1 to finish
# ===========================================================================
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " GAP FILL PIPELINE — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Waiting for current E1 (PID 62944) to finish..."

while ($true) {
    $e1 = Get-Process -Id 62944 -ErrorAction SilentlyContinue
    if (-not $e1) { break }
    if (Test-Path checkpoints\e1_latest.pt) {
        $step = & $Python -c "import torch; c=torch.load('checkpoints/e1_latest.pt',map_location='cpu',weights_only=False); print(c.get('global_step',0))" 2>$null
        Write-Host "$(Get-Date -Format 'HH:mm:ss') E1 step=$step — waiting..."
    }
    Start-Sleep 60
}

Write-Host "$(Get-Date -Format 'HH:mm:ss') E1 finished. Saving final checkpoint..."
if (Test-Path checkpoints\e1_latest.pt) {
    $finalStep = & $Python -c "import torch; c=torch.load('checkpoints/e1_latest.pt',map_location='cpu',weights_only=False); print(c.get('global_step',0))" 2>$null
    Copy-Item checkpoints\e1_latest.pt "checkpoints\e1_step${finalStep}_final.pt" -Force
    Write-Host "Saved: e1_step${finalStep}_final.pt" -ForegroundColor Green
}

# ===========================================================================
# Gap Fill 1: From scratch → 25000 (covers early phase + overnight gap)
# ===========================================================================
Run-Training "gap1_0to25000" @() 25000 $null

# ===========================================================================
# Gap Fill 2: Resume from 37500 → 39200 (covers 37550-39050 gap)
# ===========================================================================
if (Test-Path checkpoints\e1_step37500.pt) {
    Copy-Item checkpoints\e1_step37500.pt checkpoints\e1_latest.pt -Force
    Run-Training "gap2_37500to39200" @() 1700 "checkpoints/e1_latest.pt"
}

# ===========================================================================
# Done — Merge logs
# ===========================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " GAP FILL COMPLETE — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================" -ForegroundColor Cyan

$logFiles = Get-ChildItem checkpoints\gapfill_*.log | Sort-Object Name
Write-Host "Generated logs:"
$logFiles | ForEach-Object { Write-Host "  $($_.Name)  $([math]::Round($_.Length/1KB,1))KB" }

Write-Host ""
Write-Host "To merge for paper:"
Write-Host "  1. e1_original_merged.log (original, 6050→50000)"
Write-Host "  2. gapfill_gap1_0to25000.log (retrain, 0→25000)"
Write-Host "  3. gapfill_gap2_37500to39200.log (retrain, 37500→39200)"
Write-Host ""
Write-Host "Merge command: python merge_logs.py"
