#!/bin/bash
# scripts/run_exp3.sh — Run Experiment 3 (V3 CrossAttn@C2+C4)
# Usage: bash scripts/run_exp3.sh <GPU_ID> <BATCH_SIZE> <EPOCHS>
# Example: bash scripts/run_exp3.sh 4 24 100

GPU=${1:-4}
BATCH=${2:-24}
EPOCHS=${3:-100}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=============================================="
echo " Experiment 3 — V3 CrossAttn@C2+C4"
echo " GPU=$GPU  BATCH=$BATCH  EPOCHS=$EPOCHS"
echo "=============================================="

python tools/train.py \
    --config      config/voc_v3.yaml \
    --version     v3 \
    --gpu         "$GPU" \
    --epochs      "$EPOCHS" \
    --batch_size  "$BATCH" \
    --seed        42 \
    --amp \
    --checkpoint_interval 10 \
    --output_dir  outputs \
    --vis_images  12 \
    --vis_interval 5 \
    --score_thresh 0.35

echo "Experiment 3 done."
