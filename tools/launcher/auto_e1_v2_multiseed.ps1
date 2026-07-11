# auto_e1_v2_multiseed.ps1 — Chain-launch v2 E1 with seeds 42, 123, 999
# Pure denoising (latent_consistency_weight=0 confirmed working)
#
# Usage: .\tools\launcher\auto_e1_v2_multiseed.ps1

$Python = "python"
Set-Location "."

$DataPath = "data/corpus.txt"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONUNBUFFERED = "1"

$Seeds = @(42, 123, 999)
$MaxSteps = 50000
$BaseDir = "checkpoints"

# Common args (same as current seed=42 run)
$CommonArgs = @(
    "-m", "flued.e1_stage_a",
    "--preset", "class300m_16gb",
    "--data-path", $DataPath,
    "--max-lines", "50000",
    "--max-eval-batches", "200",
    "--target-accuracy", "1.0",
    "--max-steps", "$MaxSteps",
    "--lambda-var", "0.5",
    "--lambda-entropy", "0.05",
    "--lambda-utf8", "0.02",
    "--lambda-type", "0.05",
    "--target-compression", "0.3",
    "--compression-weight", "0.1",
    "--entropy-warmup-steps", "10000",
    "--ckpt-every", "1000"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " FLUED v2 Multi-Seed E1 — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host " Seeds: $($Seeds -join ', ')"
Write-Host " Steps per seed: $MaxSteps"
Write-Host " Config: pure denoising (latent_consistency_weight=0)"
Write-Host "============================================================" -ForegroundColor Cyan

foreach ($seed in $Seeds) {
    $CkptDir = "$BaseDir/e1_v2_seed$seed"
    $LogFile = "$BaseDir/e1_v2_seed${seed}_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host " SEED $seed — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
    Write-Host " Checkpoint dir: $CkptDir"
    Write-Host " Log: $LogFile"
    Write-Host "============================================================" -ForegroundColor Yellow

    $SeedArgs = $CommonArgs + @(
        "--seed", "$seed",
        "--ckpt-dir", $CkptDir
    )

    & $Python $SeedArgs 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Path $LogFile -Value $line
    }

    $exitCode = $LASTEXITCODE
    Write-Host ""
    Write-Host "SEED $seed FINISHED — Exit: $exitCode @ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor $(if($exitCode -eq 0){'Green'}else{'Red'})

    if ($exitCode -ne 0) {
        Write-Host "Seed $seed failed! Stopping chain." -ForegroundColor Red
        break
    }

    # Verify final checkpoint exists
    $finalCkpt = "$CkptDir/e1_latest.pt"
    if (Test-Path $finalCkpt) {
        $step = & $Python -c "import torch; c=torch.load('$finalCkpt',map_location='cpu',weights_only=False); print(c.get('global_step',0))" 2>$null
        Write-Host "Final checkpoint: step=$step" -ForegroundColor Green
    } else {
        Write-Host "WARNING: No final checkpoint found!" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ALL SEEDS COMPLETE — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
