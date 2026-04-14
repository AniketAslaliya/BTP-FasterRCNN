# README_EXPERIMENTS.md — Multi-Experiment Faster R-CNN Guide

## Overview

This project trains three multimodal Faster R-CNN variants on the **LLVIP** dataset (RGB + IR pedestrian detection) and compares their performance.

| Exp | Model | Fusion Strategy |
|-----|-------|----------------|
| 1   | **V1 Baseline**       | Avg@C1/C2/C3 + CrossAttn@C4 |
| 2   | **V2 CBAM@C4**        | Avg@C1/C2/C3 + CBAM + CrossAttn@C4 |
| 3   | **V3 CrossAttn@C2+C4** | Avg@C1/C3 + CrossAttn@C2 + CrossAttn@C4 ← **NEW** |

---

## Project Structure

```
FasterRCNN-PyTorch/
├── model/
│   ├── faster_rcnn_v1.py        Exp 1 — baseline
│   ├── faster_rcnn_v2.py        Exp 2 — CBAM@C4
│   └── faster_rcnn_v3.py        Exp 3 — CrossAttn@C2+C4 (NEW)
├── dataset/
│   └── voc.py                   LLVIP multimodal dataset loader
├── utils/
│   ├── metrics.py               AP50/75/90, mAP, Prec, Rec + CSV logging
│   ├── visualizer.py            Detection visualization (GT vs Pred)
│   ├── logger.py                File + console logger
│   └── seed_utils.py            Reproducibility seed setter
├── tools/
│   ├── train.py                 Unified training script (all 3 experiments)
│   └── compare_results.py       Final comparison table + plots
├── config/
│   ├── voc_v1.yaml              Exp 1 config
│   ├── voc_v2.yaml              Exp 2 config
│   └── voc_v3.yaml              Exp 3 config
└── scripts/
    ├── run_exp1.sh              Run Experiment 1
    ├── run_exp2.sh              Run Experiment 2
    ├── run_exp3.sh              Run Experiment 3
    └── run_all.sh               Run all 3 + comparison
```

---

## Setup on Remote GPU Server

### Step 1 — Upload folder
```bash
# From your local machine:
scp -r FasterRCNN-PyTorch/  user@server:/home/user/
```

### Step 2 — Install dependencies
```bash
cd /home/user/FasterRCNN-PyTorch
pip install -r requirements.txt
```

### Step 3 — Update data paths in config files
Edit the LLVIP paths in `config/voc_v1.yaml`, `voc_v2.yaml`, `voc_v3.yaml`:
```yaml
dataset_params:
  rgb_train_path: "/path/to/LLVIP/visible/train"
  rgb_test_path:  "/path/to/LLVIP/visible/test"
  ir_train_path:  "/path/to/LLVIP/infrared/train"
  ir_test_path:   "/path/to/LLVIP/infrared/test"
  ann_train_path: "/path/to/LLVIP/Annotations"
  ann_test_path:  "/path/to/LLVIP/Annotations"
```

---

## Terminal Commands

### ▶ Run one experiment (GPU, batch size, epochs all specified)

```bash
# Experiment 1 — V1 Baseline
bash scripts/run_exp1.sh  0  24  100
#                         ^   ^   ^
#                        GPU Batch Epochs

# Experiment 2 — V2 CBAM
bash scripts/run_exp2.sh  0  24  100

# Experiment 3 — V3 CrossAttn@C2+C4
bash scripts/run_exp3.sh  0  24  100
```

### ▶ Run all 3 experiments sequentially + auto comparison
```bash
bash scripts/run_all.sh  0  24  100
```

### ▶ Run in background (recommended for long jobs)
```bash
nohup bash scripts/run_all.sh 0 24 100 > run_all.log 2>&1 &

# Monitor progress
tail -f run_all.log
```

### ▶ Run on multiple GPUs in parallel (3 different GPUs)
```bash
bash scripts/run_exp1.sh 0 24 100 &   # GPU 0
bash scripts/run_exp2.sh 1 24 100 &   # GPU 1
bash scripts/run_exp3.sh 2 24 100 &   # GPU 2
wait
python tools/compare_results.py \
    --exp1_dir outputs/exp1_v1_baseline \
    --exp2_dir outputs/exp2_v2_cbam \
    --exp3_dir outputs/exp3_v3_c2c4 \
    --output_dir outputs/comparison
```

### ▶ Low GPU memory? Reduce batch size
```bash
# If batch=24 causes OOM, use batch=8 with AMP still enabled
bash scripts/run_all.sh 0 8 100
```

### ▶ Direct python command (full control)
```bash
python tools/train.py \
    --config   config/voc_v3.yaml \
    --version  v3 \
    --gpu      0 \
    --epochs   100 \
    --batch_size 24 \
    --seed     42 \
    --amp \
    --checkpoint_interval 10 \
    --output_dir outputs/exp3_v3_c2c4 \
    --vis_images 12 \
    --vis_interval 5 \
    --score_thresh 0.35
```

---

## Outputs

After training, each experiment folder contains:

```
outputs/exp1_v1_baseline/
├── training.log           # Full training log (console + file)
├── checkpoints/
│   ├── best_model.pth     # Best AP50 checkpoint
│   ├── epoch_010.pth      # Interval checkpoints
│   ├── epoch_020.pth
│   └── ...
├── metrics/
│   └── metrics.csv        # Per-epoch: AP50, AP75, AP90, mAP, Prec, Rec
└── visualizations/
    ├── epoch_000/         # Detection images: GT (green) vs Pred (red)
    │   ├── img001.png
    │   └── ...
    ├── epoch_005/
    └── ...
```

After running `compare_results.py`:
```
outputs/comparison/
├── comparison_table.csv   # Best epoch per model
├── loss_curve.png
├── ap50_curve.png
├── ap90_curve.png
└── map_coco_curve.png
```

---

## Metrics Definitions

| Metric     | Description |
|------------|-------------|
| **AP50**   | Average Precision at IoU=0.50 (PASCAL VOC 11-pt) |
| **AP75**   | Average Precision at IoU=0.75 |
| **AP90**   | Average Precision at IoU=0.90 (strictest localization) |
| **mAP**    | COCO-style mAP averaged over IoU=[0.50:0.05:0.95] |
| **Precision** | TP / (TP+FP) at IoU=0.50 |
| **Recall** | TP / (TP+FN) at IoU=0.50 |

---

## V3 Model Architecture (NEW)

```
RGB Image → ResNet-50 → C1(256) / C2(512) / C3(1024) / C4(2048)
IR  Image → ResNet-50 → C1(256) / C2(512) / C3(1024) / C4(2048)

Fusion:
  C1_fused = avg(RGB_C1, IR_C1)
  C2_fused = C2FusionModule(RGB_C2, IR_C2)    ← CrossAttn(8-head) + weighted avg
  C3_fused = avg(RGB_C3, IR_C3)
  C4_fused = CrossAttentionFusion(RGB_C4, IR_C4)  ← 8-head, Q=RGB, K/V=IR

→ FPN(C1..C4) → 5 levels × 256ch
→ RPN → proposals
→ MultiScaleRoIAlign (7×7)
→ FC Head → class + bbox
→ NMS → final detections
```

---

## Assumptions

1. LLVIP dataset is already downloaded and structured (visible/, infrared/, Annotations/)
2. PyTorch 1.13.1 + torchvision 0.14.1 installed (CUDA must be available)
3. At least one GPU is available; `--amp` flag is strongly recommended for batch=24
4. `--gpu 0` sets `CUDA_VISIBLE_DEVICES=0` inside the training script
5. All 3 experiments use identical seeds (42), LR (5e-5), split strategy for fair comparison
