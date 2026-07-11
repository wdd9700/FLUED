param(
  [string]$DataPath = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt",
  [string]$OutRoot = "K:\FLUED_archive\v31_segmental_diffusion_20260629\sweep_boundary_value_500",
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
  $stride = [Math]::Max(1, [int]($SeqLen / 2))
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  Write-Host "=== v3.1 boundary-value sweep: $Name ==="
  python tools\analysis\train_v3_segmental_diffusion_2m.py `
    --data-path $DataPath `
    --out-dir $outDir `
    --streaming-train `
    --seq-len $SeqLen `
    --stride $stride `
    --batch-size 64 `
    --max-steps $MaxSteps `
    --max-eval-batches 8 `
    --eval-max-lines 6000 `
    --num-workers 2 `
    --seed $Seed `
    --device cuda `
    --amp `
    --prediction-target current `
    --future-target current `
    --recon-loss-weight 1.0 `
    --future-loss-weight 1.0 `
    --stage-a-ratio 0.50 `
    --stage-b-ratio 0.85 `
    --noise-scale 0.03 `
    --log-every 100 `
    --metrics-every 100 `
    --ckpt-every 500 `
    @ExtraArgs
}

Run-One -Name "bv000_base" -ExtraArgs @("--boundary-value-loss-weight", "0.0")
Run-One -Name "bv002" -ExtraArgs @("--boundary-value-loss-weight", "0.02")
Run-One -Name "bv005" -ExtraArgs @("--boundary-value-loss-weight", "0.05")
Run-One -Name "bv010" -ExtraArgs @("--boundary-value-loss-weight", "0.10")
Run-One -Name "bv005_rate030" -ExtraArgs @("--boundary-value-loss-weight", "0.05", "--rate-target", "0.30")

python tools\analysis\summarize_v31_diffusion_sweep.py --root $OutRoot
