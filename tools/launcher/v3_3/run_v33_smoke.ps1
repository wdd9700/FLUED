param(
    [string]$Config = "configs\v3_3\v33_no_memory_smoke.json",
    [string]$DataPath = "",
    [string]$Device = "cuda",
    [string]$Python = "python",
    [switch]$Amp
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$args = @(
    "tools\train\v3_3\train_v33.py",
    "--config", $Config,
    "--device", $Device
)

if ($DataPath) {
    $args += @("--data-path", $DataPath)
}

if ($Amp) {
    $args += @("--amp")
}

& $Python @args
