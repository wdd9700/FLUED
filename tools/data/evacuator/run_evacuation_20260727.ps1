param(
    [ValidateSet('O', 'N')]
    [string]$Lane,
    [int]$Workers = 1
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'evacuator\target\release\flued-corpus-evacuator.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing release binary: $exe" }

$layout = @{
    O = @(
        @{ Input = 'W:\incoming'; Output = 'O:\FLUED_evacuation_20260727'; Source = 'fineweb_w' },
        @{ Input = 'Z:\v11';      Output = 'O:\FLUED_evacuation_20260727'; Source = 'v11_z_v3' },
        @{ Input = 'Y:\v11';      Output = 'O:\FLUED_evacuation_20260727'; Source = 'v11_y_v3' }
    )
    N = @(
        @{ Input = 'X:\incoming'; Output = 'N:\FLUED_evacuation_20260727'; Source = 'github_code_x' }
    )
}

$destination = "$Lane`:\FLUED_evacuation_20260727"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$log = Join-Path $destination "lane_$Lane.log"

foreach ($job in $layout[$Lane]) {
    "[$(Get-Date -Format o)] start $($job.Source)" | Tee-Object -FilePath $log -Append
    # Concatenated gzip exports can have large members. One worker keeps their
    # bounded streaming footprint below the host-memory limit during evacuation.
    & $exe --input $job.Input --output $job.Output --source $job.Source --workers $Workers --zstd-level 3 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "Conversion failed: $($job.Source), exit=$LASTEXITCODE" }
    "[$(Get-Date -Format o)] complete $($job.Source)" | Tee-Object -FilePath $log -Append
}

"[$(Get-Date -Format o)] lane complete" | Tee-Object -FilePath $log -Append
