# B_experiment: target_compression sweep
# Launches E1 training with target=0.45 and target=0.20
# GPU required. Run when BLT ByteLM finishes.
param(
    [ValidateSet('0.45','0.20')]
    [string]$Target = '0.45'
)

$ErrorActionPreference = 'Stop'

# ---- Validate ----
$DataPath = 'data/corpus.txt'
if (-not (Test-Path $DataPath)) {
    Write-Error "Data not found: $DataPath"
    exit 1
}

# ---- Check no other GPU processes ----
$existing = Get-Process python* -ErrorAction SilentlyContinue
if ($existing) {
    Write-Warning "Existing Python processes:"
    $existing | Format-Table Id, ProcessName, StartTime
    $confirm = Read-Host "Kill them and continue? (y/n)"
    if ($confirm -ne 'y') { exit 0 }
    $existing | Stop-Process -Force
    Start-Sleep 10
}

# ---- Config ----
$CkptDir = "checkpoints/e1_tc$($Target.Replace('.',''))"
$LogFile = "$CkptDir/train_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host "=== B Experiment: target_compression=$Target ===" -ForegroundColor Cyan
Write-Host "Checkpoint dir: $CkptDir"
Write-Host "Log: $LogFile"

# ---- CPU thread limit ----
$env:OMP_NUM_THREADS = 4
$env:MKL_NUM_THREADS = 4

# ---- Launch ----
$cmd = @(
    'python', '-m', 'flued.e1_stage_a',
    '--preset', 'class300m_16gb',
    '--data-path', $DataPath,
    '--max-lines', '50000',
    '--max-eval-batches', '200',
    '--amp', '--amp-dtype', 'fp16',
    '--target-accuracy', '1.0',
    '--max-steps', '40000',
    '--target-compression', $Target,
    '--compression-weight', '0.1',
    '--lambda-var', '0.5',
    '--lambda-entropy', '0.05',
    '--lambda-utf8', '0.02',
    '--lambda-type', '0.05',
    '--ckpt-every', '5000',
    '--ckpt-dir', $CkptDir,
    '--seed', '42'
)

Write-Host "Command:" -ForegroundColor DarkGray
Write-Host ($cmd -join ' ') -ForegroundColor DarkGray

New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
& $cmd[0] $cmd[1..$cmd.Length] 2>&1 | Tee-Object -FilePath $LogFile
