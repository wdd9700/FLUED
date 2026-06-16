# run_d0_calibrate.ps1 — D0: Short calibration runs before formal D1
#
# Verifies for each baseline:
#   1. BLT: actual entropy_theta in use (0.3 vs 3.5 preset mismatch)
#   2. BPE: 20L vs 24L config, tokenizer loading
#   3. Byte: no-decode baseline works
#   4. FLUED: checkpoint loads, segmenter frozen
#   5. All: log format, eval set, BPB calculation consistent
#
# Each run: 2000 steps, ~30-40 min per model on 5090
#
# Usage: pwsh run_d0_calibrate.ps1

param(
    [string]$DataPath = "./data/corpus_v3.txt",
    [string]$FluedCkpt = "./checkpoints/e1_v2_seed42/e1_step50000.pt",
    [string]$BytelmCkpt = "./checkpoints/bytel_m_latest.pt",
    [string]$BltCkpt = "./checkpoints/blt_latest.pt",
    [int]$MaxLines = 50000,
    [int]$CalibSteps = 2000
)

$Python = "python"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " D0: CALIBRATION — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host " Each model: $CalibSteps steps"
Write-Host "============================================================" -ForegroundColor Cyan

# ===========================================================================
# Calibration 1: BLT entropy_theta
# ===========================================================================
Write-Host "`n--- D0-1: BLT entropy_theta calibration ---" -ForegroundColor Yellow

# Quick test: run BLT with explicit theta values and check avg_patches
$Thetas = @(0.3, 1.0, 3.5)
foreach ($theta in $Thetas) {
    $CkptDir = "checkpoints/d0_blt_theta$($theta.ToString().Replace('.','p'))"
    New-Item -ItemType Directory $CkptDir -Force | Out-Null

    Write-Host "  Testing theta=$theta ..."
    & $Python -m tools.train.train_blt `
        --preset 300m_frozen `
        --data-path $DataPath --max-lines $MaxLines `
        --local-lm-ckpt $BytelmCkpt `
        --max-steps $CalibSteps --ckpt-every 500 `
        --ckpt-dir $CkptDir `
        --entropy-theta $theta `
        2>&1 | Select-String "avg_patches|theta|patches_per" | ForEach-Object { "    $_" }

    # Extract avg_patches from log
    $log = Get-Content "$CkptDir/blt_*.log" -Raw -ErrorAction SilentlyContinue
    if ($log -match "avg_patches[=:\s]+([\d.]+)") {
        Write-Host "  theta=$theta → avg_patches=$($Matches[1])" -ForegroundColor Green
    }
}

# ===========================================================================
# Calibration 2: BPE layer config (20L vs 24L)
# ===========================================================================
Write-Host "`n--- D0-2: BPE layer calibration ---" -ForegroundColor Yellow

$BpeConfigs = @(
    @{Name="bpe_20L_8k";  Enc=10; Dec=10; Tokenizer="checkpoints/bpe_tokenizer_8k/tokenizer.json"},
    @{Name="bpe_24L_8k";  Enc=12; Dec=12; Tokenizer="checkpoints/bpe_tokenizer_8k/tokenizer.json"},
    @{Name="bpe_20L_16k"; Enc=10; Dec=10; Tokenizer="checkpoints/bpe_tokenizer_16k/tokenizer.json"},
    @{Name="bpe_24L_16k"; Enc=12; Dec=12; Tokenizer="checkpoints/bpe_tokenizer_16k/tokenizer.json"},
    @{Name="bpe_20L_32k"; Enc=10; Dec=10; Tokenizer="checkpoints/bpe_tokenizer_32k/tokenizer.json"},
    @{Name="bpe_24L_32k"; Enc=12; Dec=12; Tokenizer="checkpoints/bpe_tokenizer_32k/tokenizer.json"}
)

$BpeResults = @()
foreach ($cfg in $BpeConfigs) {
    $CkptDir = "checkpoints/d0_$($cfg.Name)"
    New-Item -ItemType Directory $CkptDir -Force | Out-Null

    Write-Host "  Testing $($cfg.Name): enc=$($cfg.Enc) dec=$($cfg.Dec) ..."
    & $Python -m flued.e3_train `
        --model public --tokenizer-name hf --tokenizer-path $cfg.Tokenizer `
        --data-path $DataPath --max-lines $MaxLines `
        --max-steps $CalibSteps --batch-size 1 `
        --ckpt-dir $CkptDir `
        --d-model 1024 --nhead 16 --dim-feedforward 4096 `
        --num-encoder-layers $cfg.Enc --num-decoder-layers $cfg.Dec `
        2>&1 | Out-Null  # Just verify it doesn't crash

    $params = & $Python -c "
import torch; c=torch.load('$CkptDir/e3_latest.pt',map_location='cpu',weights_only=False)
total = sum(p.numel() for p in c['model'].values())
print(f'{total/1e6:.1f}M')
" 2>$null
    $BpeResults += "$($cfg.Name): $params params"
    Write-Host "  $($cfg.Name): $params params" -ForegroundColor Green
}

Write-Host "`n=== D0 BPE Summary ==="
$BpeResults

# ===========================================================================
# Calibration 3: Byte baseline
# ===========================================================================
Write-Host "`n--- D0-3: Byte baseline (no compression) ---" -ForegroundColor Yellow
$CkptDir = "checkpoints/d0_byte"
New-Item -ItemType Directory $CkptDir -Force | Out-Null
& $Python -m flued.e3_train `
    --model byte --data-path $DataPath --max-lines $MaxLines `
    --max-steps $CalibSteps --batch-size 1 --ckpt-dir $CkptDir `
    2>&1 | Select-String "bpb|loss|step" | Select-Object -Last 5

# ===========================================================================
# Calibration 4: FLUED v2 frozen segmenter
# ===========================================================================
Write-Host "`n--- D0-4: FLUED v2 frozen segmenter ---" -ForegroundColor Yellow
$CkptDir = "checkpoints/d0_flued"
New-Item -ItemType Directory $CkptDir -Force | Out-Null
& $Python -m flued.e3_train `
    --model flued --flued-ckpt $FluedCkpt `
    --data-path $DataPath --max-lines $MaxLines `
    --max-steps $CalibSteps --batch-size 1 --ckpt-dir $CkptDir `
    2>&1 | Select-String "bpb|loss|step|segmenter" | Select-Object -Last 5

Write-Host "`n=== D0 CALIBRATION COMPLETE ===" -ForegroundColor Green
Write-Host "Check checkpoints/d0_*/ for results"
