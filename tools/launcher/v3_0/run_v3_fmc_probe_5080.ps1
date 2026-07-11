param(
    [string]$Python = "python",
    [string]$DataPath = "data/corpus.txt",
    [string]$RunRoot = "checkpoints\v3_fmc_small_5080"
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Runs = @(
    "A_baseline_dp07",
    "B_low_denoise_dp03",
    "C_high_denoise_dp09",
    "D_budget_stress_w025"
)

foreach ($name in $Runs) {
    $Ckpt = Join-Path $RunRoot "$name\e1_latest.pt"
    if (!(Test-Path $Ckpt)) {
        Write-Warning "Missing checkpoint: $Ckpt"
        continue
    }
    $Out = Join-Path $RunRoot "$name\fmc_probe.json"
    $ActJson = Join-Path $RunRoot "$name\activation_probe.json"
    $ActPng = Join-Path $RunRoot "$name\activation_probe.png"
    Write-Host ""
    Write-Host "=== FMC probe: $name ==="
    & $Python -u "tools\analysis\v3_0\fmc_boundary_probe.py" `
        --ckpt $Ckpt `
        --data-path $DataPath `
        --seq-len 128 `
        --max-lines 20000 `
        --max-batches 16 `
        --batch-size 8 `
        --device cuda `
        --output-json $Out `
        --activation-json $ActJson `
        --activation-plot $ActPng
    if ($LASTEXITCODE -ne 0) {
        throw "Probe failed: $name, exit=$LASTEXITCODE"
    }
}
