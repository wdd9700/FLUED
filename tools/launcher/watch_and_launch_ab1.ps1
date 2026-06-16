# watch_and_launch_ab1.ps1 — Wait for AB2 to finish, then launch AB1 compression (weight=0.3)
$Python = "C:\Python314\python.exe"
Set-Location "E:\projects\FLUED\FLUED"
$DataPath = "E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONUNBUFFERED = "1"

$LogFile = "ab1_w03_launcher.log"
$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
"$ts  Watcher started. Waiting for AB2 training to complete..." | Out-File $LogFile -Encoding utf8

# Detect if E1 training is running
$sleepMinutes = 10
while ($true) {
    $running = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq 'C:\Python314\python.exe' }
    if (-not $running) {
        $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        "$ts  No training detected. Proceeding in 30s..." | Out-File $LogFile -Append -Encoding utf8
        Start-Sleep -Seconds 30
        break
    }
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $count = ($running | Measure-Object).Count
    "$ts  Still running ($count python procs). Sleeping ${sleepMinutes}min..." | Out-File $LogFile -Append -Encoding utf8
    Start-Sleep -Seconds ($sleepMinutes * 60)
}

# AB1 compression sweep with weight=0.3
$CompArgs = @(
    "-m", "flued.e1_stage_a",
    "--preset", "class300m_16gb",
    "--data-path", $DataPath,
    "--max-lines", "50000",
    "--max-eval-batches", "200",
    "--target-accuracy", "0.99",
    "--max-steps", "30000",
    "--lambda-var", "0.5",
    "--lambda-entropy", "0.05",
    "--lambda-utf8", "0.02",
    "--lambda-type", "0.05",
    "--compression-weight", "0.3",
    "--entropy-warmup-steps", "10000",
    "--ckpt-every", "1000",
    "--seed", "42"
)

$CompressionTargets = @(0.20, 0.30, 0.45, 0.60)
foreach ($tc in $CompressionTargets) {
    $Name = "ab1_w03_tc$($tc.ToString('0.00').Replace('.',''))"
    $CkptDir = "checkpoints/$Name"
    if (-not (Test-Path $CkptDir)) { New-Item -ItemType Directory -Path $CkptDir | Out-Null }

    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $free = [math]::Round((Get-PSDrive E).Free/1GB, 1)
    "$ts  AB1(w=0.3) tc=$tc STARTING (E盘: ${free}GB)" | Out-File $LogFile -Append -Encoding utf8

    & $Python $CompArgs --target-compression $tc --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        Add-Content -Path "$CkptDir/run.log" -Value $line
    }

    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "$ts  AB1(w=0.3) tc=$tc DONE (exit=$LASTEXITCODE)" | Out-File $LogFile -Append -Encoding utf8
}

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$free = [math]::Round((Get-PSDrive E).Free/1GB, 1)
"$ts  ALL DONE. E盘: ${free}GB" | Out-File $LogFile -Append -Encoding utf8
