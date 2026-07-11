# auto_e3_launcher.ps1 — Autonomous E1→E3 pipeline supervisor
# Usage: powershell -File auto_e3_launcher.ps1

$ErrorActionPreference = "Continue"
$Python = "python"
$ProjectDir = "."
Set-Location $ProjectDir

$DataPath = "data/corpus.txt"
$MaxLines = 50000
$E1MaxSteps = 50000
$E3MaxSteps = 20000

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " AUTO E3 LAUNCHER — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Phase 1: Waiting for FLUED E1 to reach step $E1MaxSteps..."

$e1Checkpoint = "checkpoints/e1_step${E1MaxSteps}.pt"
$e1Latest = "checkpoints/e1_latest.pt"

while ($true) {
    if (Test-Path $e1Checkpoint) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') E1 checkpoint FOUND: $e1Checkpoint" -ForegroundColor Green
        break
    }
    $e1Procs = Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'e1_stage_a' }
    if (-not $e1Procs) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') E1 process gone — checking checkpoint..." -ForegroundColor Yellow
        Start-Sleep 10
        if (Test-Path $e1Checkpoint) { Write-Host "  Found! Proceeding." -ForegroundColor Green; break }
        if (Test-Path $e1Latest) {
            $stepInfo = & $Python -c "import torch; c=torch.load('$e1Latest',map_location='cpu',weights_only=False); print(c.get('global_step',0))" 2>$null
            Write-Host "  e1_latest.pt step=$stepInfo"
            if ([int]$stepInfo -ge $E1MaxSteps) {
                Copy-Item $e1Latest $e1Checkpoint -Force
                Write-Host "  Copied to $e1Checkpoint" -ForegroundColor Green
                break
            }
        }
        Write-Host "  E1 incomplete (step=$stepInfo). Aborting." -ForegroundColor Red
        exit 1
    }
    $gpuUtil = (nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>$null) -replace ' %',''
    Write-Host "$(Get-Date -Format 'HH:mm:ss') E1 running (GPU: ${gpuUtil}%) — next check in 60s"
    Start-Sleep 60
}

# Phase 2: Launch E3
Write-Host ""
Write-Host "Phase 2: Launching FLUED E3 (frozen DSC + 350M CausalTransformerLM)" -ForegroundColor Cyan

$E3LogDir = "checkpoints/e3_logs"
New-Item -ItemType Directory -Force -Path $E3LogDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$E3LogFile = "$E3LogDir/e3_run_${ts}.log"

$E3Args = @("-m", "flued.e3_train", "--model", "flued", "--preset", "500m",
    "--flued-ckpt", $e1Checkpoint, "--data-path", $DataPath,
    "--max-lines", $MaxLines, "--max-steps", $E3MaxSteps)

Write-Host "Command: $Python $E3Args"
Write-Host "Log: $E3LogFile"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Python
$psi.Arguments = [string]::Join(" ", $E3Args)
$psi.WorkingDirectory = $ProjectDir
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.EnvironmentVariables["OMP_NUM_THREADS"] = "4"
$psi.EnvironmentVariables["MKL_NUM_THREADS"] = "4"

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$proc.Start() | Out-Null
$proc.ProcessorAffinity = [IntPtr]0xFF000000  # cores 24-31

Write-Host "E3 PID: $($proc.Id)  Affinity: cores 24-31"
Write-Host ""

# Phase 3: Log streaming + health monitoring
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$lastStep = 0
$stallCount = 0

while (-not $proc.HasExited) {
    Start-Sleep 30
    
    # Read new output
    $newOutput = $proc.StandardOutput.ReadToEnd()
    if ($newOutput) {
        Add-Content -Path $E3LogFile -Value $newOutput
        $lines = $newOutput -split "`n"
        foreach ($line in $lines) {
            if ($line -match "step=\s*(\d+)") {
                $lastStep = [int]$Matches[1]
                $elapsed = [math]::Round($sw.Elapsed.TotalMinutes, 1)
                Write-Host "$(Get-Date -Format 'HH:mm:ss') E3 step=$lastStep  elapsed=${elapsed}min"
                $stallCount = 0
            }
            if ($line -match "CUDA out of memory|RuntimeError|FloatingPointError|NaN") {
                Write-Host "  ERROR: $line" -ForegroundColor Red
                if ($line -match "out of memory") {
                    Write-Host "  OOM — killing process" -ForegroundColor Red
                    $proc.Kill()
                    break
                }
            }
            if ($line -match "bpb=") {
                $stallCount = 0  # Training progressing
            }
        }
    } else {
        $stallCount++
        if ($stallCount -gt 20) {
            Write-Host "$(Get-Date -Format 'HH:mm:ss') STALL DETECTED — no output for 10min" -ForegroundColor Yellow
            $stallCount = 0
        }
    }
    
    # GPU health
    $gpuTemp = (nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>$null) -replace ' ',''
    if ($gpuTemp -and [int]$gpuTemp -gt 85) {
        Write-Host "  GPU temp ${gpuTemp}C — WARNING" -ForegroundColor Yellow
    }
}

# Phase 4: Completion
$sw.Stop()
$exitCode = $proc.ExitCode
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " E3 FINISHED — exit=$exitCode  elapsed=$([math]::Round($sw.Elapsed.TotalHours,1))h"
Write-Host " Final step: $lastStep"
Write-Host "============================================================" -ForegroundColor Cyan

# Quick final check
if (Test-Path $E3LogFile) {
    $bpbLine = Get-Content $E3LogFile | Select-String "bpb=" | Select-Object -Last 1
    if ($bpbLine) { Write-Host "Last BPB: $bpbLine" }
}

nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>$null | Write-Host
