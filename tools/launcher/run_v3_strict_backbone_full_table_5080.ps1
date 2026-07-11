param(
    [string]$DataPath = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt",
    [string]$OutRoot = "K:\FLUED_archive\v3_strict_backbone_full_table_20260703",
    [int]$MaxSteps = 3000,
    [int]$BatchSize = 128,
    [int]$NumWorkers = 8,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = "C:\Users\74090\Miniconda3\envs\soulvlm\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

function Run-StrictBackbone {
    param(
        [string]$Mode,
        [string]$RunName,
        [string]$CodecCkpt = ""
    )

    $runDir = Join-Path $OutRoot $RunName
    $summaryPath = Join-Path $runDir "summary.json"
    if ((Test-Path $summaryPath) -and (-not $Force)) {
        Write-Host "skip existing $RunName"
        return
    }
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $common = @(
        (Join-Path $repo "tools\analysis\train_v3_strict_masked_backbone.py"),
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

    & $python @common
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Run-StrictBackbone -Mode "byte" -RunName "byte_3k_strict_mask"

$runs = @(
    @{ Name = "latent_v31_codec40k_utf8clean_3k"; Path = "K:\FLUED_archive\v31_language_codec_2m_20260702\codec_40k_utf8clean" },
    @{ Name = "latent_v31_codec10k_pool_mfl_3k"; Path = "K:\FLUED_archive\v31_language_codec_2m_20260702\codec_10k_pool_mfl" },
    @{ Name = "latent_v32_stage3_mfl_nomemory_10k_3k"; Path = "K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_nomemory_10k" },
    @{ Name = "latent_v32_stage3_mfl_memory_10k_3k"; Path = "K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_memory_10k" },
    @{ Name = "latent_v321_mfl_nomemory_masked15k_3k"; Path = "K:\FLUED_archive\v32_masked_codec_2m_20260703\v321_mfl_nomemory_masked_15k" },
    @{ Name = "latent_v321_mfl_memory_masked15k_3k"; Path = "K:\FLUED_archive\v32_masked_codec_2m_20260703\v321_mfl_memory_masked_15k" }
)

foreach ($run in $runs) {
    Run-StrictBackbone -Mode "latent" -RunName $run.Name -CodecCkpt $run.Path
}

& $python (Join-Path $repo "tools\analysis\summarize_v3_strict_backbone_sweep.py") `
    $OutRoot `
    --out-dir $OutRoot `
    --out-md (Join-Path $OutRoot "strict_backbone_full_table.md")

Write-Host "v3 strict masked-source backbone full table complete: $OutRoot"
