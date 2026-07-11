param(
  [string]$DataPath = "data/corpus.txt",
  [string]$OutRoot = "archive\v31_segmental_diffusion_20260629\ablation_1500",
  [int]$MaxSteps = 1500,
  [int]$SeqLen = 128,
  [int]$Seed = 1234
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Run-One {
  param(
    [string]$Name,
    [string[]]$ExtraArgs
  )
  $outDir = Join-Path $OutRoot $Name
  $stride = [Math]::Max(1, [int]($SeqLen / 2))
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  Write-Host "=== v3.1 diffusion ablation: $Name ==="
  python tools\analysis\v3_0\train_v3_segmental_diffusion_2m.py `
    --data-path $DataPath `
    --out-dir $outDir `
    --streaming-train `
    --seq-len $SeqLen `
    --stride $stride `
    --batch-size 64 `
    --max-steps $MaxSteps `
    --max-eval-batches 16 `
    --eval-max-lines 12000 `
    --num-workers 2 `
    --seed $Seed `
    --device cuda `
    --amp `
    --prediction-target current `
    --future-target current `
    --recon-loss-weight 1.0 `
    --future-loss-weight 1.0 `
    --boundary-value-loss-weight 0.0 `
    --stage-a-ratio 0.50 `
    --stage-b-ratio 0.85 `
    --noise-scale 0.03 `
    --rate-target 0.35 `
    --log-every 100 `
    --metrics-every 100 `
    --ckpt-every 500 `
    @ExtraArgs
}

Run-One -Name "full_anneal_memory_ar" -ExtraArgs @()
Run-One -Name "abl_no_memory" -ExtraArgs @("--no-memory")
Run-One -Name "abl_no_ar" -ExtraArgs @("--no-ar")
Run-One -Name "abl_fixed_target_1step" -ExtraArgs @("--step-schedule", "fixed_target")
Run-One -Name "abl_fixed_max_multistep" -ExtraArgs @("--step-schedule", "fixed_max")

python tools\analysis\v3_1\summarize_v31_diffusion_sweep.py --root $OutRoot
