# GPU Pipeline Watchdog
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = 4
$env:MKL_NUM_THREADS = 4

Write-Host "=== GPU Pipeline Watchdog ===" -ForegroundColor Cyan
Write-Host "Started: $(Get-Date -Format 'HH:mm:ss')"

$python = "C:\Python314\python.exe"
$data = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt"

# Step 1: Wait for FLUED
Write-Host "[1/3] Waiting for FLUED to finish..." -ForegroundColor Yellow
do {
    Start-Sleep -Seconds 30
    $flued = Get-Process python* -ErrorAction SilentlyContinue | Where-Object {
        (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine -match "flued.e1_stage_a" 2>$null
    }
} while ($flued)
Write-Host "FLUED done: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green

# Step 2: BPE GPU
Write-Host "[2/3] BPE GPU training..." -ForegroundColor Yellow
$bpeArgs = @(
    "bpe_baseline_standalone/train.py",
    "--preset", "300m_8gb",
    "--data-path", $data,
    "--max-lines", "50000", "--max-steps", "40000",
    "--device", "cuda",
    "--resume", "bpe_baseline_standalone/checkpoints/bpe_latest.pt",
    "--batch-size", "4", "--grad-accum-steps", "4",
    "--ckpt-dir", "bpe_baseline_standalone/checkpoints"
)
& $python $bpeArgs
Write-Host "BPE done: $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE" -ForegroundColor Green

# Step 3: BLT GPU
Write-Host "[3/3] BLT GPU training..." -ForegroundColor Yellow
$bltArgs = @(
    "train_blt.py",
    "--preset", "300m_frozen", "--global-layers", "11",
    "--local-lm-ckpt", "checkpoints/bytel m_latest.pt",
    "--data-path", $data,
    "--max-lines", "50000", "--max-steps", "40000",
    "--device", "cuda", "--ckpt-every", "5000"
)
& $python $bltArgs
Write-Host "BLT done: $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE" -ForegroundColor Green

Write-Host "=== ALL DONE: $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
