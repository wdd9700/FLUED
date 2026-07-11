# archive_and_clean.ps1 — Complete archive + verify + clean
# Runs everything in background, logs to archive_cleanup.log

$ErrorActionPreference = "Continue"
Set-Location "."
$LogFile = "archive_cleanup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log { param($msg) $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; "$ts  $msg" | Tee-Object -FilePath $LogFile -Append }

Write-Log "=== ARCHIVE & CLEANUP STARTED ==="

# Step 1: Copy remaining seed999 (robocopy with retry)
Write-Log "Step 1: Copying e1_v2_seed999 to K盘..."
robocopy "checkpoints\e1_v2_seed999" "archive\v2_final_seeds\e1_v2_seed999" /E /R:3 /W:10
Write-Log "seed999 robocopy exit: $LASTEXITCODE"

# Step 2: Copy root log files
Write-Log "Step 2: Copying root log files..."
Copy-Item "checkpoints\a_class_v2_summary.json" "archive\v2_final_seeds\" -Force
Copy-Item "checkpoints\ablation_run.log" "archive\ablation_20260611\" -Force
Copy-Item "checkpoints\e1_smoke_cpu.log" "archive\ablation_20260611\" -Force
Copy-Item "checkpoints\e1_v2_seed42_resume_20260608_041527.log" "archive\v2_final_seeds\" -Force
Write-Log "Root files copied."

# Step 3: Verify K盘 copies (dir count only)
Write-Log "Step 3: Verifying K盘 copies..."

$checks = @(
    @{src="checkpoints/ab1_tc020"; dst="archive\ablation_20260611\ab1_tc020"; name="AB1 tc=0.20"},
    @{src="checkpoints/ab1_tc030"; dst="archive\ablation_20260611\ab1_tc030"; name="AB1 tc=0.30"},
    @{src="checkpoints/ab1_tc045"; dst="archive\ablation_20260611\ab1_tc045"; name="AB1 tc=0.45"},
    @{src="checkpoints/ab1_tc060"; dst="archive\ablation_20260611\ab1_tc060"; name="AB1 tc=0.60"},
    @{src="checkpoints/ab2_dp03"; dst="archive\ab2_denoise_20260612\ab2_dp03"; name="AB2 dp=0.3"},
    @{src="checkpoints/ab2_dp05"; dst="archive\ab2_denoise_20260612\ab2_dp05"; name="AB2 dp=0.5"},
    @{src="checkpoints/ab2_dp09"; dst="archive\ab2_denoise_20260612\ab2_dp09"; name="AB2 dp=0.9"},
    @{src="checkpoints/e1_v2_seed42"; dst="archive\v2_final_seeds\e1_v2_seed42"; name="v2 seed42"},
    @{src="checkpoints/e1_v2_seed123"; dst="archive\v2_final_seeds\e1_v2_seed123"; name="v2 seed123"},
    @{src="checkpoints/e1_v2_seed999"; dst="archive\v2_final_seeds\e1_v2_seed999"; name="v2 seed999"}
)

$allOk = $true
foreach ($c in $checks) {
    $srcCount = (Get-ChildItem $c.src -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $dstCount = (Get-ChildItem $c.dst -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $srcSize = [math]::Round(((Get-ChildItem $c.src -Recurse -File | Measure-Object Length -Sum).Sum)/1GB, 2)
    $dstSize = [math]::Round(((Get-ChildItem $c.dst -Recurse -File | Measure-Object Length -Sum).Sum)/1GB, 2)
    if ($srcCount -eq $dstCount -and $srcSize -eq $dstSize) {
        Write-Log "  OK: $($c.name) — $srcCount files / ${srcSize}GB"
    } else {
        Write-Log "  MISMATCH: $($c.name) — src=$srcCount/$srcSize GB  dst=$dstCount/$dstSize GB"
        $allOk = $false
    }
}

if (-not $allOk) {
    Write-Log "ERROR: Verification failed! Skipping deletions. Check $LogFile"
    exit 1
}

# Step 4: Delete from E盘 (verified copies only)
Write-Log "Step 4: Deleting archived directories from E盘..."

$deleteDirs = @(
    'checkpoints/ab1_tc020',
    'checkpoints/ab1_tc030',
    'checkpoints/ab1_tc045',
    'checkpoints/ab1_tc060',
    'checkpoints/ab2_dp03',
    'checkpoints/ab2_dp05',
    'checkpoints/ab2_dp09',
    'checkpoints/e1_v2_seed42',
    'checkpoints/e1_v2_seed123',
    'checkpoints/e1_v2_seed999'
)

foreach ($d in $deleteDirs) {
    if (Test-Path $d) {
        Write-Log "  Deleting $d..."
        Remove-Item $d -Recurse -Force
        Write-Log "  Deleted."
    }
}

# Step 5: Delete smoke test dir
Write-Log "Step 5: Deleting audit_smoke..."
if (Test-Path "checkpoints/audit_smoke") {
    Remove-Item "checkpoints/audit_smoke" -Recurse -Force
    Write-Log "  Deleted."
}

# Step 6: Clean root log files (already copied)
Write-Log "Step 6: Cleaning root log files..."
Remove-Item "checkpoints/ablation_run.log" -Force -ErrorAction SilentlyContinue
Remove-Item "checkpoints/e1_smoke_cpu.log" -Force -ErrorAction SilentlyContinue
Remove-Item "checkpoints/e1_v2_seed42_resume_20260608_041527.log" -Force -ErrorAction SilentlyContinue

$free = [math]::Round((Get-PSDrive E).Free/1GB, 1)
Write-Log "=== CLEANUP COMPLETE === E盘 free: ${free}GB"

# Step 7: List remaining
Write-Log "Remaining checkpoints:"
Get-ChildItem checkpoints -Directory | ForEach-Object { Write-Log "  $($_.Name)" }
Get-ChildItem checkpoints -File | ForEach-Object { Write-Log "  $($_.Name)" }
