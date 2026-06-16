# run_ab2_remaining.ps1 — AB2 denoise dp=0.5 → dp=0.9 (dp=0.3 already done)
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
    "--compression-weight", "0.1",
    "--entropy-warmup-steps", "10000",
    "--ckpt-every", "1000",
    "--seed", "42"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " AB2 DENOISE dp=0.5 + dp=0.9 — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host " E盘 free: $([math]::Round((Get-PSDrive E).Free/1GB,1)) GB" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$DenoiseProbs = @(0.5, 0.9)
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

Write-Host "`n=== AB2 dp=0.5 + dp=0.9 COMPLETE ===" -ForegroundColor Green
