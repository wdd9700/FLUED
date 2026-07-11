param(
    [string]$DataPath = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt",
    [string]$NoMemoryCodecCkpt = "K:\FLUED_archive\v32_masked_codec_2m_20260703\v321_mfl_nomemory_masked_15k",
    [string]$MemoryCodecCkpt = "K:\FLUED_archive\v32_masked_codec_2m_20260703\v321_mfl_memory_masked_15k",
    [string]$OutRoot = "K:\FLUED_archive\v32_strict_backbone_20260703_masked_codec_15k",
    [int]$MaxSteps = 3000,
    [int]$BatchSize = 128,
    [int]$NumWorkers = 8,
    [switch]$SkipByteBaseline
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Run-StrictBackbone {
    param(
        [string]$Mode,
        [string]$RunName,
        [string]$CodecCkpt = ""
    )

    $runDir = Join-Path $OutRoot $RunName
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $common = @(
        (Join-Path $repo "tools\analysis\train_v32_strict_masked_backbone.py"),
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
        "--mask-span-min", "1",
        "--mask-span-max", "8",
        "--length-loss-weight", "0.05",
        "--log-every", "250",
        "--ckpt-every", "1500"
    )

    if ($Mode -eq "latent") {
        $common += @("--codec-ckpt", $CodecCkpt)
    }

    python @common
}

if (-not $SkipByteBaseline) {
    Run-StrictBackbone -Mode "byte" -RunName "byte_3k_strict_mask_recheck"
}
Run-StrictBackbone -Mode "latent" -RunName "latent_v321_mfl_nomemory_maskedcodec15k_3k" -CodecCkpt $NoMemoryCodecCkpt
Run-StrictBackbone -Mode "latent" -RunName "latent_v321_mfl_memory_maskedcodec15k_3k" -CodecCkpt $MemoryCodecCkpt

python (Join-Path $repo "tools\analysis\summarize_v32_min_backbone.py") `
    $OutRoot `
    --out-path (Join-Path $OutRoot "strict_backbone_summary.md")

Write-Host "v3.2.1 strict masked-source backbone comparison complete: $OutRoot"
