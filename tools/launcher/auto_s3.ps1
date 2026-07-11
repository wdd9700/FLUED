# auto_s3.ps1 — Auto-launch S3 E1 after S2 finishes
$ErrorActionPreference = "Continue"

$Python = "python"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

$DataPath = "data/corpus.txt"
$ResumePath = "F:\FLUED\checkpoints\e1_seed999\e1_step24000.pt"
$CkptDir = "F:\FLUED\checkpoints\e1_seed999"
$LogFile = "$CkptDir\e1_class300m_48gb.log"

Write-Host "=== S3 Auto-Launcher ===" -ForegroundColor Cyan
Write-Host "Waiting for S2 E1 to finish..." -ForegroundColor Yellow
Write-Host "Polling every 60s for S2 process exit..."

# Wait until no e1_stage_a python process exists (S2 finished)
do {
    Start-Sleep 60
    $s2 = Get-Process python* -ErrorAction SilentlyContinue | 
           Where-Object { $_.CommandLine -match 'e1_stage_a' }
    if (-not $s2) { break }
    Write-Host "$(Get-Date -Format 'HH:mm:ss') S2 still running (PID=$($s2.Id))..."
} while ($true)

Write-Host "$(Get-Date -Format 'HH:mm:ss') S2 finished. Waiting 10s for GPU cleanup..." -ForegroundColor Green
Start-Sleep 10

# Additional wait for CUDA context cleanup (AGENTS.md pitfall #4)
Write-Host "Launching S3 E1 (seed=999, 24000 → 40000)..." -ForegroundColor Cyan

$args = @(
    "-m", "flued.e1_stage_a",
    "--preset", "class300m_48gb",
    "--batch-size", "4",
    "--grad-accum-steps", "8",
    "--seed", "999",
    "--resume", $ResumePath,
    "--ckpt-dir", $CkptDir,
    "--ckpt-every", "2000",
    "--data-path", $DataPath
)

& $Python $args 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
}

Write-Host ""
Write-Host "S3 E1 finished. Exit: $LASTEXITCODE" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})
