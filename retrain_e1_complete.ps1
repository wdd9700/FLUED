# retrain_e1_complete.ps1 — Wait for E1 finish, then retrain from scratch with FULL logging
# Captures the complete learning curve for paper.

$Python = "C:\Python314\python.exe"
$ProjectDir = "E:\projects\FLUED\FLUED"
Set-Location $ProjectDir

$DataPath = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " E1 RETRAIN — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host " Goal: Capture COMPLETE learning curve from step 0"
Write-Host "============================================================" -ForegroundColor Cyan

# ===========================================================================
# Phase 1: Wait for current E1 to reach step 50000
# ===========================================================================
Write-Host "Phase 1: Waiting for current E1 (PID 62944) to finish step 50000..."
while ($true) {
    $e1 = Get-Process -Id 62944 -ErrorAction SilentlyContinue
    if (-not $e1) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') E1 process exited — verifying checkpoint..."
        Start-Sleep 5
        $step = & $Python -c "import torch; c=torch.load('checkpoints/e1_latest.pt',map_location='cpu',weights_only=False); print(c.get('global_step',0))" 2>$null
        Write-Host "  e1_latest.pt step = $step"
        if ([int]$step -ge 50000) {
            Write-Host "  E1 COMPLETE (step $step)" -ForegroundColor Green
            break
        }
        Write-Host "  E1 stopped at step $step (< 50000). Waiting 60s then re-checking..."
        Start-Sleep 60
        continue
    }
    # Check checkpoint progress
    if (Test-Path checkpoints\e1_latest.pt) {
        $step = & $Python -c "import torch; c=torch.load('checkpoints/e1_latest.pt',map_location='cpu',weights_only=False); print(c.get('global_step',0))" 2>$null
        $gpu = (nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>$null) -replace ' %',''
        Write-Host "$(Get-Date -Format 'HH:mm:ss') E1 step=$step  GPU=${gpu}%  waiting..."
    }
    Start-Sleep 120
}

# Save final checkpoint
if (-not (Test-Path checkpoints\e1_step50000.pt)) {
    Copy-Item checkpoints\e1_latest.pt checkpoints\e1_step50000_final.pt -Force
    Write-Host "Saved final checkpoint: e1_step50000_final.pt" -ForegroundColor Green
}

# Save current log before starting new run
$existingLog = "checkpoints/e1_merged_full.log"
if (Test-Path $existingLog) {
    Copy-Item $existingLog "checkpoints/e1_original_merged.log" -Force
    Write-Host "Saved original merged log"
}

# ===========================================================================
# Phase 2: Retrain from scratch with FULL logging
# ===========================================================================
Write-Host ""
Write-Host "Phase 2: Retraining FLUED E1 from step 0 (complete log capture)" -ForegroundColor Cyan

$LogFile = "checkpoints/e1_retrain_full.log"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Same hyperparameters as original class300m_16gb run
$E1Args = @(
    "-m", "flued.e1_stage_a",
    "--preset", "class300m_16gb",
    "--data-path", $DataPath,
    "--max-lines", "50000",
    "--max-eval-batches", "200",
    "--amp", "--amp-dtype", "fp16",
    "--target-accuracy", "1.0",
    "--max-steps", "25000",
    "--lambda-var", "0.5",
    "--lambda-entropy", "0.05",
    "--lambda-utf8", "0.02",
    "--lambda-type", "0.05",
    "--target-compression", "0.3",
    "--compression-weight", "0.1",
    "--ckpt-every", "1000"
)

Write-Host "Command: $Python $E1Args"
Write-Host "Log file: $LogFile"
Write-Host "Target: 25000 steps (will capture 0→25000 complete curve)"
Write-Host ""

# Set environment
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

# Launch with Tee to capture all output
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Python
$psi.Arguments = [string]::Join(" ", $E1Args)
$psi.WorkingDirectory = $ProjectDir
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $false
$psi.EnvironmentVariables["OMP_NUM_THREADS"] = "4"
$psi.EnvironmentVariables["MKL_NUM_THREADS"] = "4"

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi

# Register output handlers to write to both console and log file
$logWriter = [System.IO.StreamWriter]::new($LogFile, $true)
$proc.add_OutputDataReceived({
    if ($EventArgs.Data -ne $null) {
        Write-Host $EventArgs.Data
        $logWriter.WriteLine($EventArgs.Data)
        $logWriter.Flush()
    }
})
$proc.add_ErrorDataReceived({
    if ($EventArgs.Data -ne $null) {
        Write-Host $EventArgs.Data -ForegroundColor Red
        $logWriter.WriteLine("[STDERR] " + $EventArgs.Data)
        $logWriter.Flush()
    }
})

$proc.Start() | Out-Null
$proc.ProcessorAffinity = [IntPtr]0xFF000000
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

Write-Host "E1 Retrain PID: $($proc.Id)  Affinity: cores 24-31"
Write-Host "Capturing COMPLETE log to $LogFile"
Write-Host ""

$sw = [System.Diagnostics.Stopwatch]::StartNew()

while (-not $proc.HasExited) {
    Start-Sleep 30
}
$sw.Stop()
$proc.WaitForExit()
$logWriter.Close()

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " RETRAIN COMPLETE — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host " Exit code: $($proc.ExitCode)  Elapsed: $([math]::Round($sw.Elapsed.TotalHours,1))h"
Write-Host " Log: $LogFile"
Write-Host "============================================================" -ForegroundColor Cyan

# Quick log stats
if (Test-Path $LogFile) {
    $lines = (Get-Content $LogFile).Count
    $dataLines = (Get-Content $LogFile | Select-String "step=").Count
    Write-Host "Log: $lines total lines, $dataLines data points"
}
