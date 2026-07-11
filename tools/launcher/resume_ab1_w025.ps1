# resume_ab1_w025.ps1 — Resume tc=0.30 from step 28000, then tc=0.60
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
    "--compression-weight", "0.25",
    "--entropy-warmup-steps", "10000",
    "--ckpt-every", "1000",
    "--seed", "42"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " RESUME AB1 w=0.25 — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host " E盘: $([math]::Round((Get-PSDrive E).Free/1GB,1)) GB" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Step 1: Resume tc=0.30 from checkpoint
$Name = "ab1_w025_tc030"
$CkptDir = "checkpoints/$Name"
$ResumeFrom = "$CkptDir/e1_latest.pt"

Write-Host "`n--- Resume tc=0.30 from $ResumeFrom (step 28000) ---" -ForegroundColor Yellow

& $Python $BaseArgs --target-compression 0.30 --resume $ResumeFrom --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
    $line = "$_"; Write-Host $line
    Add-Content -Path "$CkptDir/run.log" -Value $line
}
Write-Host "tc=0.30 DONE (exit=$LASTEXITCODE)" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})

# Step 2: tc=0.60 fresh
$Name = "ab1_w025_tc060"
$CkptDir = "checkpoints/$Name"
if (-not (Test-Path $CkptDir)) { New-Item -ItemType Directory -Path $CkptDir | Out-Null }

Write-Host "`n--- AB1(w=0.25) tc=0.60 -> $CkptDir ---" -ForegroundColor Yellow

& $Python $BaseArgs --target-compression 0.60 --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
    $line = "$_"; Write-Host $line
    Add-Content -Path "$CkptDir/run.log" -Value $line
}
Write-Host "tc=0.60 DONE (exit=$LASTEXITCODE)" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})

Write-Host "`n=== ALL DONE ===" -ForegroundColor Green
