#!/bin/bash
# scripts/run_exp1.sh — Run Experiment 1 (V1 Baseline)
GPU=${1:-4}; BATCH=${2:-24}; EPOCHS=${3:-100}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
echo "=== Experiment 1 — V1 Baseline | GPU=$GPU BATCH=$BATCH EPOCHS=$EPOCHS ==="
python tools/train.py --config config/voc_v1.yaml --version v1 \
    --gpu "$GPU" --epochs "$EPOCHS" --batch_size "$BATCH" \
    --seed 42 --amp --checkpoint_interval 10 --output_dir outputs \
    --vis_images 12 --vis_interval 5 --score_thresh 0.35
echo "Experiment 1 done."
