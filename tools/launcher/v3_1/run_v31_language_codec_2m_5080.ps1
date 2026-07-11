param(
    [string]$DataPath = "data/corpus.txt",
    [string]$OutRoot = "archive\v31_language_codec_2m_20260702",
    [string]$RunName = "codec_10k_utf8clean_repro",
    [int]$MaxSteps = 10000,
    [int]$BatchSize = 128,
    [int]$NumWorkers = 12,
    [int]$StreamSamplesPerWorker = 500000,
    [int]$PrefetchFactor = 4,
    [int]$LogEvery = 500,
    [int]$SeqLen = 128,
    [int]$MaxSpan = 16,
    [ValidateSet("mean", "mean_first_last")]
    [string]$PoolMode = "mean"
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$runDir = Join-Path $OutRoot $RunName
$stride = [Math]::Max(1, [int]($SeqLen / 2))
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

python (Join-Path $repo "tools\analysis\v3_1\train_v31_language_codec_2m.py") `
    --data-path $DataPath `
    --out-dir $runDir `
    --device cuda --amp `
    --streaming-train --streaming-eval `
    --stream-samples-per-worker $StreamSamplesPerWorker `
    --seq-len $SeqLen --stride $stride `
    --batch-size $BatchSize `
    --num-workers $NumWorkers `
    --prefetch-factor $PrefetchFactor `
    --max-steps $MaxSteps `
    --warmup-steps 300 `
    --max-eval-batches 32 `
    --d-model 192 --hidden 192 --nhead 4 --encoder-layers 2 --ffn-dim 768 `
    --pool-mode $PoolMode `
    --min-span 2 --max-span $MaxSpan --max-units $SeqLen `
    --eval-max-lines 5000 `
    --log-every $LogEvery `
    --ckpt-every 3000

python (Join-Path $repo "tools\analysis\v3_1\summarize_v31_language_codec.py") `
    $OutRoot `
    --out-path (Join-Path $OutRoot "sweep_summary.md")

python (Join-Path $repo "tools\eval\v3_1\eval_v31_language_codec_roi.py") `
    --ckpt $runDir `
    --out-path (Join-Path $runDir "roi_constrained.md") `
    --device cpu

python (Join-Path $repo "tools\eval\v3_1\eval_v31_language_codec_decoder.py") `
    --ckpt $runDir `
    --data-path $DataPath `
    --streaming-eval `
    --max-batches 32 `
    --batch-size $BatchSize `
    --out-path (Join-Path $runDir "decoder_streaming.md") `
    --device cpu

python (Join-Path $repo "tools\eval\v3_1\eval_v31_language_codec_memory_ablation.py") `
    --checkpoint $runDir `
    --data-path $DataPath `
    --streaming-eval `
    --max-eval-batches 16 `
    --batch-size $BatchSize `
    --out-path (Join-Path $runDir "memory_ablation_streaming.md") `
    --device cpu

python (Join-Path $repo "tools\eval\v3_1\eval_v31_language_codec_memory_cases.py") `
    --checkpoint $runDir `
    --out-path (Join-Path $runDir "memory_cases.md") `
    --device cpu

Write-Host "v3.1 codec run complete: $runDir"
