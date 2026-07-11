# E1 Ablation Experiments — Auto-launcher
# Each experiment: 10K steps, seed=42, varying lambda settings
# Usage: .\run_ablation.ps1 -Experiment A1

param([string]$Experiment = "A1")

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = 4
$env:MKL_NUM_THREADS = 4

$base = @(
    "--preset", "class300m_48gb",
    "--data-path", "data/corpus.txt",
    "--max-lines", "50000",
    "--max-steps", "10000",
    "--batch-size", "4",
    "--grad-accum-steps", "4",
    "--target-compression", "0.225",
    "--compression-weight", "0.02",
    "--entropy-warmup-steps", "2000",
    "--seed", "42",
    "--ckpt-every", "1000"
)

$ablation = @{
    "A1" = @{
        desc = "lambda_type=0: can pure reconstruction learn semantics?"
        lambda_var = 0.1; lambda_entropy = 0.05; lambda_utf8 = 0.02
        lambda_type = 0.0; lambda_cjk = 0.0; cjk_target = 0.15
        dir = "checkpoints/e1_ablation_no_type"
    }
    "A2" = @{
        desc = "lambda_var=0: is variance loss necessary?"
        lambda_var = 0.0; lambda_entropy = 0.05; lambda_utf8 = 0.02
        lambda_type = 0.15; lambda_cjk = 0.0; cjk_target = 0.15
        dir = "checkpoints/e1_ablation_no_var"
    }
    "A3" = @{
        desc = "lambda_entropy=0: is entropy loss necessary?"
        lambda_var = 0.1; lambda_entropy = 0.0; lambda_utf8 = 0.02
        lambda_type = 0.15; lambda_cjk = 0.0; cjk_target = 0.15
        dir = "checkpoints/e1_ablation_no_entropy"
    }
    "A4" = @{
        desc = "lambda_utf8=0: is UTF-8 constraint necessary?"
        lambda_var = 0.1; lambda_entropy = 0.05; lambda_utf8 = 0.0
        lambda_type = 0.15; lambda_cjk = 0.0; cjk_target = 0.15
        dir = "checkpoints/e1_ablation_no_utf8"
    }
    "A5" = @{
        desc = "all_lambda=0: pure reconstruction only"
        lambda_var = 0.0; lambda_entropy = 0.0; lambda_utf8 = 0.0
        lambda_type = 0.0; lambda_cjk = 0.0; cjk_target = 0.15
        dir = "checkpoints/e1_ablation_pure_recon"
    }
}

$cfg = $ablation[$Experiment]
if (-not $cfg) { Write-Host "Unknown experiment: $Experiment"; exit 1 }

Write-Host "=== $Experiment : $($cfg.desc) ===" -ForegroundColor Cyan
Write-Host "  var=$($cfg.lambda_var) entropy=$($cfg.lambda_entropy) utf8=$($cfg.lambda_utf8) type=$($cfg.lambda_type) cjk=$($cfg.lambda_cjk)"
Write-Host "  Output: $($cfg.dir)"
Write-Host ""

$args = $base + @(
    "--lambda-var", $cfg.lambda_var,
    "--lambda-entropy", $cfg.lambda_entropy,
    "--lambda-utf8", $cfg.lambda_utf8,
    "--lambda-type", $cfg.lambda_type,
    "--lambda-cjk", $cfg.lambda_cjk,
    "--cjk-target", $cfg.cjk_target,
    "--ckpt-dir", $cfg.dir
)

$logFile = "checkpoints/e1_ablation_$Experiment.log"
$cmd = "python -m flued.e1_stage_a $args 2>&1 | Tee-Object -FilePath `"$logFile`""
Write-Host "Launching: python -m flued.e1_stage_a ..."
Invoke-Expression $cmd
