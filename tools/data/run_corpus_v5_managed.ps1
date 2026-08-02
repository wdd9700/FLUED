param(
    [string]$Python = 'C:\Users\74090\Miniconda3\envs\soulvlm\python.exe',
    [string]$ConfigPath = '',
    [string]$OutputDir = 'N:\FLUED_corpus_v5_increment_20260725',
    [string]$ReferenceDb = 'N:\FLUED_corpus_v4\state\dedupe_hashes.sqlite3'
)

$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

$PipelineDir = $PSScriptRoot
if (-not $ConfigPath) {
    $LocalConfig = Join-Path $PipelineDir 'corpus_v5_sources_20260725.json'
    $RepoConfig = Join-Path (Split-Path (Split-Path $PipelineDir -Parent) -Parent) 'configs\data\corpus_v5_sources_20260725.json'
    $ConfigPath = if (Test-Path -LiteralPath $LocalConfig) { $LocalConfig } else { $RepoConfig }
}
$LogDir = Join-Path $OutputDir 'logs'
$ReportDir = Join-Path $OutputDir 'reports'
$StatusPath = Join-Path $ReportDir 'managed_status.json'
$BuildLog = Join-Path $LogDir 'managed_build.log'
$VerifyLog = Join-Path $LogDir 'managed_verify.log'

New-Item -ItemType Directory -Force -Path $LogDir, $ReportDir | Out-Null
@{
    state = 'running'
    pid = $PID
    started_at = (Get-Date).ToString('o')
    build_exit_code = $null
    verify_exit_code = $null
} | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8

Push-Location $PipelineDir
try {
    & $Python '.\build_corpus_v5.py' `
        --config $ConfigPath `
        --force-unlock 2>&1 | Tee-Object -FilePath $BuildLog -Append
    $buildExit = $LASTEXITCODE

    $verifyExit = $null
    if ($buildExit -in @(0, 2)) {
        & $Python '.\verify_corpus_v5.py' `
            --output-dir $OutputDir `
            --reference-db $ReferenceDb 2>&1 | Tee-Object -FilePath $VerifyLog -Append
        $verifyExit = $LASTEXITCODE
    }

    @{
        state = if ($buildExit -in @(0, 2) -and $verifyExit -eq 0) { 'complete' } else { 'failed' }
        pid = $PID
        started_at = (Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json).started_at
        finished_at = (Get-Date).ToString('o')
        build_exit_code = $buildExit
        verify_exit_code = $verifyExit
        build_log = $BuildLog
        verify_log = $VerifyLog
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
    exit $(if ($buildExit -in @(0, 2) -and $verifyExit -eq 0) { 0 } else { 1 })
}
catch {
    @{
        state = 'failed'
        pid = $PID
        finished_at = (Get-Date).ToString('o')
        error = $_.Exception.ToString()
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
    throw
}
finally {
    Pop-Location
}
