param(
    [string]$Config = "configs\v3_3\v33_full_300m_100m_corpus_v4.json",
    [string]$Device = "cuda",
    [int]$MaxSteps = 0,
    [int]$BatchSize = 0,
    [int]$GradAccumSteps = 0,
    [string]$Python = "python",
    [switch]$DryRun,
    [switch]$NoAmp,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$args = @(
    "tools\train\v3_3\train_v33.py",
    "--config", $Config,
    "--device", $Device
)

if ($MaxSteps -gt 0) { $args += @("--max-steps", "$MaxSteps") }
if ($BatchSize -gt 0) { $args += @("--batch-size", "$BatchSize") }
if ($GradAccumSteps -gt 0) { $args += @("--grad-accum-steps", "$GradAccumSteps") }
if ($DryRun) { $args += @("--dry-run") }
if ($NoAmp) { $args += @("--no-amp") }
if ($NoResume) { $args += @("--no-resume") }

& $Python @args
