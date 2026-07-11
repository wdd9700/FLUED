param(
    [string]$Python = "python",
    [string]$DataPath = "data/corpus.txt",
    [string]$OutRoot = "checkpoints\v3_fmc_small_5080",
    [int]$MaxSteps = 2000,
    [int]$MaxEvalBatches = 16
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$BaseArgs = @(
    "-u", "-m", "flued.e1_stage_a",
    "--preset", "small_gpu",
    "--data-path", $DataPath,
    "--max-steps", "$MaxSteps",
    "--max-eval-batches", "$MaxEvalBatches",
    "--ckpt-every", "1000",
    "--batch-size", "16",
    "--seq-len", "128",
    "--stride", "64",
    "--amp",
    "--amp-dtype", "bf16",
    "--latent-consistency-weight", "0",
    "--target-accuracy", "0.0",
    "--min-compression", "0.0",
    "--max-compression", "1.0"
)

$Runs = @(
    @{
        Name = "A_baseline_dp07";
        Extra = @("--denoise-prob", "0.7", "--compression-weight", "0.1", "--target-compression", "0.3")
    },
    @{
        Name = "B_low_denoise_dp03";
        Extra = @("--denoise-prob", "0.3", "--compression-weight", "0.1", "--target-compression", "0.3")
    },
    @{
        Name = "C_high_denoise_dp09";
        Extra = @("--denoise-prob", "0.9", "--compression-weight", "0.1", "--target-compression", "0.3")
    },
    @{
        Name = "D_budget_stress_w025";
        Extra = @("--denoise-prob", "0.7", "--compression-weight", "0.25", "--target-compression", "0.3")
    }
)

Write-Host "FLUED v3 FMC small 5080 probe"
Write-Host "Data: $DataPath"
Write-Host "Out:  $OutRoot"
Write-Host "Steps per run: $MaxSteps"

foreach ($run in $Runs) {
    $CkptDir = Join-Path $OutRoot $run.Name
    New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
    $Log = Join-Path $CkptDir "run.log"

    Write-Host ""
    Write-Host "=== $($run.Name) ==="
    $Args = $BaseArgs + @("--ckpt-dir", $CkptDir) + $run.Extra
    Write-Host "$Python $($Args -join ' ')"

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python $Args 2>&1 | ForEach-Object {
        $text = $_.ToString()
        Write-Host $text
        Add-Content -LiteralPath $Log -Value $text
    }
    $ErrorActionPreference = $oldEap
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $($run.Name), exit=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "All small runs complete. Next: run FMC boundary probe on latest checkpoints."
