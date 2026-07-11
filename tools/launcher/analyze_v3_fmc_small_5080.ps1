param(
    [string]$Python = "python",
    [string]$RunRoot = "checkpoints\v3_fmc_small_5080"
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

& $Python -u "tools\analysis\plot_v3_fmc_small.py" --run-root $RunRoot
if ($LASTEXITCODE -ne 0) {
    throw "analysis failed: exit=$LASTEXITCODE"
}

Write-Host "Analysis artifacts:"
Get-ChildItem -LiteralPath (Join-Path $RunRoot "analysis") -File | Select-Object Name, Length, LastWriteTime

