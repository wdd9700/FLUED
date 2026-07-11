param(
    [string]$MemoryCodecCkpt = "K:\FLUED_archive\v32_masked_codec_2m_20260703\v321_mfl_memory_masked_15k",
    [string]$NoMemoryCodecCkpt = "K:\FLUED_archive\v32_masked_codec_2m_20260703\v321_mfl_nomemory_masked_15k",
    [string]$OutRoot = "K:\FLUED_archive\v32_masked_codec_2m_20260703\memory_stress_15k",
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

python (Join-Path $repo "tools\eval\eval_v32_masked_memory_stress.py") `
    --checkpoint $MemoryCodecCkpt `
    --device $Device `
    --amp `
    --out-json (Join-Path $OutRoot "memory_topk.json") `
    --out-md (Join-Path $OutRoot "memory_topk.md")

python (Join-Path $repo "tools\eval\eval_v32_masked_memory_stress.py") `
    --checkpoint $NoMemoryCodecCkpt `
    --device $Device `
    --amp `
    --out-json (Join-Path $OutRoot "nomemory.json") `
    --out-md (Join-Path $OutRoot "nomemory.md")

Write-Host "v3.2.1 strict masked memory stress complete: $OutRoot"
