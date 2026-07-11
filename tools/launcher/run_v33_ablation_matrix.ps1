param(
    [string]$Matrix = "configs\v33_ablation_2m.json",
    [string]$DataPath = "",
    [string]$Device = "cuda",
    [string]$Only = "",
    [int]$BatchSize = 0,
    [int]$MaxSteps = 0,
    [int]$NumWorkers = -1,
    [switch]$NoAmp,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$python = "C:\Users\74090\Miniconda3\envs\soulvlm\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$args = @(
    "tools\launcher\run_v33_ablation_matrix.py",
    "--matrix", $Matrix,
    "--device", $Device
)

if ($DataPath) { $args += @("--data-path", $DataPath) }
if ($Only) { $args += @("--only", $Only) }
if ($BatchSize -gt 0) { $args += @("--batch-size", "$BatchSize") }
if ($MaxSteps -gt 0) { $args += @("--max-steps", "$MaxSteps") }
if ($NumWorkers -ge 0) { $args += @("--num-workers", "$NumWorkers") }
if ($NoAmp) { $args += @("--no-amp") }
if ($DryRun) { $args += @("--dry-run") }

& $python @args
