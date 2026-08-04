#!/bin/bash
# S0.6 summarizer capacity full-factorial matrix (spec section 15).
# Each arm: 20K steps, S0 four prefixes from s05_baseline_3k snapshot, all else fresh.
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE PYTHONUTF8=1
PY=C:/Users/74090/Miniconda3/envs/soulvlm/python.exe
CKPT="L:\\FLUED_archive\\s05_baseline_3k_20260802\\latest.pt"
DATA="E:\\projects\\SoulMamba\\soulvlm_project\\temp\\corpus_v3.txt"
PFX="byte_lookup,encoder_blocks,segmentor_blocks,segmentor_head"
cd /e/projects/FLUED/FLUED

run_arm() {
  name=$1; shift
  echo "=== arm $name start $(date -u +%H:%M:%S) ==="
  "$PY" tools/train/v3_6/train_v36.py --config configs/canonical_v36.json \
    --init-checkpoint "$CKPT" --init-prefixes "$PFX" \
    --data-path "$DATA" --run-id "$name" --out-dir "checkpoints/$name" \
    --max-steps 20000 "$@" > "checkpoints/${name}_stdout.log" 2>&1
  echo "=== arm $name done rc=$? $(date -u +%H:%M:%S) ==="
}

run_arm s05_sum_a0_20k_20260802
run_arm s05_sum_s_20k_20260802  --summarizer-slots 16
run_arm s05_sum_h_20k_20260802  --summarizer-hidden 2048
run_arm s05_sum_m_20k_20260802  --d-mem 1024
run_arm s05_sum_sh_20k_20260802 --summarizer-slots 16 --summarizer-hidden 2048
run_arm s05_sum_sm_20k_20260802 --summarizer-slots 16 --d-mem 1024
run_arm s05_sum_hm_20k_20260802 --summarizer-hidden 2048 --d-mem 1024
run_arm s05_sum_shm_20k_20260802 --summarizer-slots 16 --summarizer-hidden 2048 --d-mem 1024
echo "=== matrix complete ==="
