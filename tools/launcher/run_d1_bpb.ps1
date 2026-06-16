# run_d1_bpb.ps1 — D1: BPB comparison (D0-calibrated configs)
#
# Models: byte | bpe_8k_20L | bpe_8k_24L | bpe_16k_20L | bpe_16k_24L
#         bpe_32k_20L | bpe_32k_24L | blt | flued_v2
#
# Split across 2x 5090: GPU0=BLT+BPE, GPU1=Byte+FLUED
# Usage: pwsh run_d1_bpb.ps1 -Gpu 0   (on each GPU)

param(
    [string]$DataPath = "./data/corpus_v3.txt",
    [string]$FluedCkpt = "./checkpoints/e1_v2_seed42/e1_step50000.pt",
    [string]$BytelmCkpt = "./checkpoints/bytel_m_latest.pt",
    [string]$BltCkpt = "./checkpoints/blt_latest.pt",
    [int]$MaxLines = 50000,
    [int]$TrainSteps = 50000,
    [int]$Gpu = 0
)

$Python = "python"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:CUDA_VISIBLE_DEVICES = "$Gpu"

$Experiments = @(
    @{Id="D1_byte"; Model="byte"; Params="353M"; Cmd=@("--model","byte")},
    @{Id="D1_bpe_8k_20L"; Model="bpe_8k_20L"; Params="~311M"; Cmd=@("--model","public","--tokenizer-name","hf","--tokenizer-path","checkpoints/bpe_tokenizer_8k/tokenizer.json","--num-encoder-layers","10","--num-decoder-layers","10")},
    @{Id="D1_bpe_8k_24L"; Model="bpe_8k_24L"; Params="~369M"; Cmd=@("--model","public","--tokenizer-name","hf","--tokenizer-path","checkpoints/bpe_tokenizer_8k/tokenizer.json","--num-encoder-layers","12","--num-decoder-layers","12")},
    @{Id="D1_bpe_16k_20L"; Model="bpe_16k_20L"; Params="~317M"; Cmd=@("--model","public","--tokenizer-name","hf","--tokenizer-path","checkpoints/bpe_tokenizer_16k/tokenizer.json","--num-encoder-layers","10","--num-decoder-layers","10")},
    @{Id="D1_bpe_16k_24L"; Model="bpe_16k_24L"; Params="~375M"; Cmd=@("--model","public","--tokenizer-name","hf","--tokenizer-path","checkpoints/bpe_tokenizer_16k/tokenizer.json","--num-encoder-layers","12","--num-decoder-layers","12")},
    @{Id="D1_bpe_32k_20L"; Model="bpe_32k_20L"; Params="~328M"; Cmd=@("--model","public","--tokenizer-name","hf","--tokenizer-path","checkpoints/bpe_tokenizer_32k/tokenizer.json","--num-encoder-layers","10","--num-decoder-layers","10")},
    @{Id="D1_bpe_32k_24L"; Model="bpe_32k_24L"; Params="~386M"; Cmd=@("--model","public","--tokenizer-name","hf","--tokenizer-path","checkpoints/bpe_tokenizer_32k/tokenizer.json","--num-encoder-layers","12","--num-decoder-layers","12")},
    @{Id="D1_blt"; Model="blt"; Params="303M"; Cmd=@("--model","blt","--blt-ckpt",$BltCkpt,"--bytelm-ckpt",$BytelmCkpt)},
    @{Id="D1_flued_v2"; Model="flued_v2"; Params="328M"; Cmd=@("--model","flued","--flued-ckpt",$FluedCkpt)}
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " D1 BPB — GPU=$Gpu — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================" -ForegroundColor Cyan

$BaseArgs = @("-m","flued.e3_train","--data-path",$DataPath,"--max-lines","$MaxLines","--max-steps","$TrainSteps","--batch-size","2","--d-model","1024","--nhead","16","--dim-feedforward","4096")

foreach ($exp in $Experiments) {
    $CkptDir = "checkpoints/$($exp.Id.ToLower())"
    New-Item -ItemType Directory $CkptDir -Force | Out-Null
    Write-Host "`n>>> $($exp.Id) ($($exp.Model), $($exp.Params))" -ForegroundColor Yellow
    $t0 = Get-Date

    & $Python $BaseArgs @($exp.Cmd) --ckpt-dir $CkptDir 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        Add-Content -Path "$CkptDir/train.log" -Value $line
    }

    $wall = [math]::Round(((Get-Date) - $t0).TotalHours, 1)
    $log = Get-Content "$CkptDir/train.log" -Raw -ErrorAction SilentlyContinue
    $bpb = if ($log -match "bpb[=:\s]+([\d.]+)") { $Matches[1] } else { "N/A" }

    Write-Host "$($exp.Id) DONE: BPB=$bpb, wall=$($wall)h" -ForegroundColor Green
    & $Python tools/eval/experiment_tracker.py record --exp $exp.Id --phase D1 --model $exp.Model --train-steps $TrainSteps --params $exp.Params --wall-time $wall --bpb $bpb --notes "GPU=$Gpu"
}

Write-Host "`nGPU=$Gpu complete." -ForegroundColor Green
& $Python tools/eval/experiment_tracker.py table
