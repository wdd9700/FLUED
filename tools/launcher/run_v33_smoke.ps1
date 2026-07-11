param(
    [string]$Config = "configs\v33_no_memory_smoke.json",
    [string]$DataPath = "",
    [string]$Device = "cuda",
    [switch]$Amp
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$python = "C:\Users\74090\Miniconda3\envs\soulvlm\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$args = @(
    "tools\train\train_v33.py",
    "--config", $Config,
    "--device", $Device
)

if ($DataPath) {
    $args += @("--data-path", $DataPath)
}

if ($Amp) {
    $args += @("--amp")
}

& $python @args
