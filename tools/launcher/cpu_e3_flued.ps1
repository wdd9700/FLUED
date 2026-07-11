# cpu_e3_flued.ps1 — E3 downstream LM on CPU (migrate to GPU later)
$Python = "python"
Set-Location "."

$DataPath = "data/corpus.txt"
$LogFile = "checkpoints/e3_cpu_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " CPU E3 FLUED Downstream — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host " Preset: small (8L d=512), CPU, checkpoint every 500 steps"
Write-Host " Log: $LogFile"
Write-Host " NOTE: CPU training is for pipeline validation + early progress."
Write-Host "       Will migrate to GPU when GPU retrain completes (~15h)."
Write-Host "============================================================" -ForegroundColor Cyan

# Use "small" preset for CPU feasibility (~50M LM, 8 layers)
# Can migrate to 500m preset on GPU later by changing --preset
$args = @(
    "-m", "flued.e3_train",
    "--model", "flued",
    "--preset", "small",
    "--flued-ckpt", "checkpoints/e1_step40000.pt",
    "--data-path", $DataPath,
    "--max-lines", "50000",
    "--device", "cpu"
)

& $Python $args 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Host ""
Write-Host "E3 CPU finished. Exit: $LASTEXITCODE" -ForegroundColor $(if($LASTEXITCODE -eq 0){'Green'}else{'Yellow'})
Write-Host "To migrate to GPU: change --preset small to --preset 500m, remove --device cpu"
