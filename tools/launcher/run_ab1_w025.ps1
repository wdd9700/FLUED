# run_ab1_w025.ps1 — AB1 compression sweep with weight=0.25 (retry after w=0.3 NaN)
$Python = "C:\Python314\python.exe"
Set-Location "E:\projects\FLUED\FLUED"
$DataPath = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt"

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
Write-Host " AB1 COMPRESSION w=0.25 — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host " E盘: $([math]::Round((Get-PSDrive E).Free/1GB,1)) GB" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Run tc=0.30 first (failed at w=0.3)
# If it survives, proceed to tc=0.60
$Targets = @(0.30, 0.60)
foreach ($tc in $Targets) {
    $Name = "ab1_w025_tc$($tc.ToString('0.00').Replace('.',''))"
    $CkptDir = "checkpoints/$Name"
    if (-not (Test-Path $CkptDir)) { New-Item -ItemType Directory -Path $CkptDir | Out-Null }

    Write-Host "`n--- AB1(w=0.25) tc=$tc -> $CkptDir ---" -ForegroundColor Yellow

    & $Python $BaseArgs --target-compression $tc --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        Add-Content -Path "$CkptDir/run.log" -Value $line
    }
    Write-Host "AB1(w=0.25) tc=$tc DONE (exit=$LASTEXITCODE)" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})
}

Write-Host "`n=== AB1 w=0.25 COMPLETE ===" -ForegroundColor Green
