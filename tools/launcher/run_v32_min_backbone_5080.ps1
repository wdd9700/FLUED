param(
    [string]$DataPath = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt",
    [string]$NoMemoryCodecCkpt = "K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_nomemory_10k",
    [string]$MemoryCodecCkpt = "K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_memory_10k",
    [string]$OutRoot = "K:\FLUED_archive\v32_backbone_20260703",
    [int]$MaxSteps = 3000,
    [int]$BatchSize = 128,
    [int]$NumWorkers = 8
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Run-Backbone {
    param(
        [string]$Mode,
        [string]$RunName,
        [string]$CodecCkpt = "",
        [double]$LatentByteLossWeight = 0.0
    )

    $runDir = Join-Path $OutRoot $RunName
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $common = @(
        (Join-Path $repo "tools\analysis\train_v32_min_backbone.py"),
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
Run-Backbone -Mode "latent" -RunName "latent_v32_mfl_nomemory_3k_mse" -CodecCkpt $NoMemoryCodecCkpt -LatentByteLossWeight 0.0
Run-Backbone -Mode "latent" -RunName "latent_v32_mfl_memory_3k_mse" -CodecCkpt $MemoryCodecCkpt -LatentByteLossWeight 0.0
Run-Backbone -Mode "latent" -RunName "latent_v32_mfl_memory_3k_byteaux1" -CodecCkpt $MemoryCodecCkpt -LatentByteLossWeight 1.0

python (Join-Path $repo "tools\analysis\summarize_v32_min_backbone.py") `
    $OutRoot `
    --out-path (Join-Path $OutRoot "backbone_summary.md")

Write-Host "v3.2 minimal-backbone comparison complete: $OutRoot"
