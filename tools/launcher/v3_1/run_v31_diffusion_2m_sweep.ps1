param(
  [string]$DataPath = "data/corpus.txt",
  [string]$OutRoot = "archive\v31_segmental_diffusion_20260629\sweep_500",
  [int]$MaxSteps = 500,
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
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  Write-Host "=== v3.1 diffusion sweep: $Name ==="
  python tools\analysis\v3_0\train_v3_segmental_diffusion_2m.py `
    --data-path $DataPath `
    --out-dir $outDir `
    --streaming-train `
    --seq-len $SeqLen `
    --stride ([Math]::Max(1, [int]($SeqLen / 2))) `
    --batch-size 64 `
    --max-steps $MaxSteps `
    --max-eval-batches 8 `
    --eval-max-lines 6000 `
    --num-workers 2 `
    --seed $Seed `
    --device cuda `
    --amp `
    --log-every 100 `
    --metrics-every 100 `
    --ckpt-every 500 `
    @ExtraArgs
}

Run-One -Name "base_anneal_lr3e4_noise04" -ExtraArgs @(
  "--lr", "3e-4",
  "--noise-scale", "0.04",
  "--stage-a-ratio", "0.40",
  "--stage-b-ratio", "0.75"
)

Run-One -Name "slow_anneal_lr3e4_noise04" -ExtraArgs @(
  "--lr", "3e-4",
  "--noise-scale", "0.04",
  "--stage-a-ratio", "0.60",
  "--stage-b-ratio", "0.90"
)

Run-One -Name "slow_anneal_lr2e4_noise04" -ExtraArgs @(
  "--lr", "2e-4",
  "--noise-scale", "0.04",
  "--stage-a-ratio", "0.60",
  "--stage-b-ratio", "0.90"
)

Run-One -Name "slow_anneal_lr3e4_noise02" -ExtraArgs @(
  "--lr", "3e-4",
  "--noise-scale", "0.02",
  "--stage-a-ratio", "0.60",
  "--stage-b-ratio", "0.90"
)

Run-One -Name "fixed_target_lr3e4" -ExtraArgs @(
  "--step-schedule", "fixed_target",
  "--lr", "3e-4",
  "--noise-scale", "0.02"
)

Run-One -Name "fixed_max_lr3e4" -ExtraArgs @(
  "--step-schedule", "fixed_max",
  "--lr", "3e-4",
  "--noise-scale", "0.04"
)

python tools\analysis\v3_1\summarize_v31_diffusion_sweep.py --root $OutRoot
