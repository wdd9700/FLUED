param(
    [string]$DataPath = "data/corpus.txt",
    [string]$CodecCkpt = "archive\v31_language_codec_2m_20260702\codec_10k_utf8clean",
    [string]$OutRoot = "archive\v31_backbone_20260702",
    [int]$MaxSteps = 3000,
    [int]$BatchSize = 128,
    [int]$NumWorkers = 8
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

function Run-Backbone {
    param(
        [string]$Mode,
        [string]$RunName,
        [double]$LatentByteLossWeight = 0.0
    )

    $runDir = Join-Path $OutRoot $RunName
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $common = @(
        (Join-Path $repo "tools\analysis\v3_1\train_v31_min_backbone.py"),
        "--mode", $Mode,
        "--data-path", $DataPath,
        "--out-dir", $runDir,
        "--device", "cuda",
        "--amp",
        "--streaming-train",
        "--streaming-eval",
        "--stream-samples-per-worker", "200000",
        "--seq-len", "128",
        "--stride", "64",
        "--batch-size", "$BatchSize",
        "--num-workers", "$NumWorkers",
        "--prefetch-factor", "4",
        "--max-steps", "$MaxSteps",
        "--warmup-steps", "200",
        "--max-eval-batches", "32",
        "--hidden", "192",
        "--layers", "2",
        "--nhead", "4",
        "--ffn-dim", "768",
        "--min-span", "2",
        "--max-span", "16",
        "--max-units", "64",
        "--mask-prob", "0.15",
        "--log-every", "250",
        "--ckpt-every", "1500"
    )

    if ($Mode -eq "byte") {
        $common += @("--byte-mask-mode", "segment")
    } else {
        $common += @("--codec-ckpt", $CodecCkpt, "--latent-byte-loss-weight", "$LatentByteLossWeight")
    }

    python @common
}

Run-Backbone -Mode "byte" -RunName "byte_3k_segmentmask_repro"
Run-Backbone -Mode "latent" -RunName "latent_3k_mse_repro" -LatentByteLossWeight 0.0
Run-Backbone -Mode "latent" -RunName "latent_3k_byteaux1_repro" -LatentByteLossWeight 1.0

python (Join-Path $repo "tools\analysis\v3_1\summarize_v31_min_backbone.py") `
    $OutRoot `
    --out-path (Join-Path $OutRoot "backbone_summary.md")

Write-Host "v3.1 minimal-backbone comparison complete: $OutRoot"
