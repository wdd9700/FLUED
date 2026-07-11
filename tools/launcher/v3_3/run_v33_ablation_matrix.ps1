param(
    [string]$Matrix = "configs\v3_3\v33_ablation_2m.json",
    [string]$DataPath = "",
    [string]$Device = "cuda",
    [string]$Only = "",
    [int]$BatchSize = 0,
    [int]$MaxSteps = 0,
    [int]$NumWorkers = -1,
    [string]$Python = "python",
    [switch]$NoAmp,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$args = @(
    "tools\launcher\v3_3\run_v33_ablation_matrix.py",
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

& $Python @args
