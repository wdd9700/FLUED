param(
  [string]$DataPath = "data/corpus.txt",
  [string]$OutRoot = "archive\v3_segmental_workspace_20260629",
  [int]$MaxSteps = 15000,
  [int]$BatchSize = 16,
  [int]$SeqLen = 128,
  [int]$EvalBatches = 24,
  [int]$EvalMaxLines = 30000,
  [int]$NumWorkers = 2,
  [int]$Seed = 1234
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Run-Variant {
  param(
    [string]$Name,
    [string[]]$ExtraArgs
  )
  $outDir = Join-Path $OutRoot $Name
  $stride = [Math]::Max(1, [int]($SeqLen / 2))
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  Write-Host "=== FLUED v3.1 2M matrix: $Name ==="
  python tools\analysis\v3_0\train_v3_segmental_workspace_2m.py `
    --data-path $DataPath `
    --out-dir $outDir `
    --streaming-train `
    --seq-len $SeqLen `
    --stride $stride `
    --batch-size $BatchSize `
    --max-steps $MaxSteps `
    --max-eval-batches $EvalBatches `
    --eval-max-lines $EvalMaxLines `
    --num-workers $NumWorkers `
    --seed $Seed `
    --device cuda `
    --amp `
    --d-model 192 `
    --hidden 192 `
    --controller-hidden 256 `
    --log-every 100 `
    --metrics-every 100 `
    --ckpt-every 3000 `
    @ExtraArgs

  $ckpt = Join-Path $outDir "latest.pt"
  if (Test-Path $ckpt) {
    python tools\eval\v3_0\eval_v3_segmental_workspace_roi.py `
      --ckpt $ckpt `
      --out-dir (Join-Path $outDir "roi") `
      --seq-len $SeqLen `
      --device cuda
  }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Run-Variant -Name "full_refine4_ar2_attenres_memory" -ExtraArgs @(
  "--refine-steps", "4",
  "--student-refine-steps", "1",
  "--ar-correction-passes", "2",
  "--residual-mixer", "attn",
  "--denoise-prob", "0.5",
  "--denoise-steps", ([string][Math]::Floor($MaxSteps / 2))
)

Run-Variant -Name "abl_no_refine_ar2_memory" -ExtraArgs @(
  "--refine-steps", "0",
  "--student-refine-steps", "0",
  "--ar-correction-passes", "2",
  "--residual-mixer", "last",
  "--distill-loss-weight", "0.0",
  "--denoise-prob", "0.5",
  "--denoise-steps", ([string][Math]::Floor($MaxSteps / 2))
)

Run-Variant -Name "abl_refine4_no_ar_memory" -ExtraArgs @(
  "--refine-steps", "4",
  "--student-refine-steps", "1",
  "--ar-correction-passes", "0",
  "--residual-mixer", "attn",
  "--denoise-prob", "0.5",
  "--denoise-steps", ([string][Math]::Floor($MaxSteps / 2))
)

Run-Variant -Name "abl_refine4_ar2_no_memory" -ExtraArgs @(
  "--refine-steps", "4",
  "--student-refine-steps", "1",
  "--ar-correction-passes", "2",
  "--residual-mixer", "attn",
  "--no-memory",
  "--denoise-prob", "0.5",
  "--denoise-steps", ([string][Math]::Floor($MaxSteps / 2))
)

Run-Variant -Name "abl_refine4_ar2_last_residual" -ExtraArgs @(
  "--refine-steps", "4",
  "--student-refine-steps", "1",
  "--ar-correction-passes", "2",
  "--residual-mixer", "last",
  "--denoise-prob", "0.5",
  "--denoise-steps", ([string][Math]::Floor($MaxSteps / 2))
)

Run-Variant -Name "abl_refine4_ar2_no_value_probe" -ExtraArgs @(
  "--refine-steps", "4",
  "--student-refine-steps", "1",
  "--ar-correction-passes", "2",
  "--residual-mixer", "attn",
  "--value-loss-weight", "0.0",
  "--confidence-loss-weight", "0.0",
  "--denoise-prob", "0.5",
  "--denoise-steps", ([string][Math]::Floor($MaxSteps / 2))
)

python tools\analysis\v3_0\summarize_v3_segmental_workspace_matrix.py --root $OutRoot

Write-Host "All v3.1 2M matrix runs completed and summarized. Root: $OutRoot"
