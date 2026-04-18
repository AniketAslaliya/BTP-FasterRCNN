# BTP Project Context

## What This Project Is

A Bachelor's Thesis Project building a dual-stream multimodal Faster R-CNN for RGB+Thermal object detection. The model runs two ResNet-50 backbones in parallel (one for RGB, one for Infrared), fuses them using attention mechanisms, and runs a standard Faster R-CNN detection head on top.

## Two Experiments

| Experiment | Dataset | Classes | Best AP50 |
|---|---|---|---|
| V4 | LLVIP | 1 (person) | 0.609 |
| V5 | FLIR Aligned ADAS v2 | 3 (person, car, bicycle) | 0.613 at epoch 14, training ongoing |

## Critical Fact About V4 and V5

They are the **SAME architecture**. V5 is V4 with dataset-specific hyperparameters changed. The backbone, fusion, FPN are all identical. This is the same approach used in **CSSA (Cao et al., CVPR 2023)** which was our baseline paper.

---

## Codebase Structure

```
/home/23uec571/BTP/
├── model/
│   ├── faster_rcnn_v1.py     # Baseline single-stream
│   ├── faster_rcnn_v2.py     # Early fusion
│   ├── faster_rcnn_v3.py     # Mid fusion
│   ├── faster_rcnn_v4.py     # CBAM + CrossAttention — LLVIP SOTA
│   ├── faster_rcnn_v5.py     # FLIR 3-class + Focal Loss — current experiment
│   └── unimodal.py           # Unimodal RGB/IR baselines
├── dataset/
│   ├── voc.py                # LLVIP loader (VOC XML format)
│   └── flir.py               # FLIR Aligned loader (VOC XML format)
├── tools/
│   └── train.py              # Unified training script (supports v1-v5, unimodal_rgb, unimodal_ir)
├── config/
│   ├── voc_v4.yaml           # LLVIP V4 training config
│   └── flir_v5.yaml          # FLIR V5 training config (task_name: exp5_v5_optimized)
├── utils/
│   ├── metrics.py            # COCO-standard AP metrics
│   ├── logger.py             # Training logger
│   ├── seed_utils.py         # Reproducibility
│   └── visualizer.py        # Detection visualizations
└── experiments/
    └── unimodal/
        ├── results_rgb/      # Unimodal RGB baseline (50 epochs, LLVIP)
        └── results_ir/       # Unimodal IR baseline (50 epochs, LLVIP)
```

---

## Architecture Details

### V4 Model (`faster_rcnn_v4.py`)

- Dual ResNet-50 (one per modality, both ImageNet pretrained)
- `LearnableCBAMFusion` at C1 (256ch), C2 (512ch), C3 (1024ch)
- `CrossAttentionFusion` at C4 (2048ch, attn_dim=256, num_heads=8)
- FPN: 5 levels, out_channels=256
- Standard Faster R-CNN head, `num_classes=2` (background + person)
- `min_size=512`, `max_size=640`

### V5 Model (`faster_rcnn_v5.py`)

Everything from V4, plus:

- **Focal Loss** injected via monkey-patch on torchvision roi_heads: `_rh.fastrcnn_loss = _patched` (alpha=0.25, gamma=2.0)
- **Anchors:** `FLIR_ANCHOR_SIZES = ((16,32),(32,64),(64,128),(128,256),(256,512))`, `FLIR_ANCHOR_RATIOS = ((0.33,0.5,1.0,2.0,3.0),)*5`
- **NMS:** `box_score_thresh=0.30`, `box_nms_thresh=0.45`
- **RPN:** `rpn_pre_nms_top_n_train=3000`, `rpn_post_nms_top_n_train=2000`
- `num_classes=4` (background=0, person=1, bicycle=2, car=3)
- `load_llvip_checkpoint_for_flir()` function at bottom — loads V4 weights, filters shape mismatches, freezes C1-C3

---

## FLIR Label Mapping (`dataset/flir.py`)

```python
TARGET_CLASSES = ['person', 'bicycle', 'car']
CLASS_TO_LABEL = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}
# person=1, bicycle=2, car=3, background=0
```

---

## `train.py` argparse Arguments

```
--config, --version, --gpu, --epochs, --batch_size, --lr, --seed, --amp,
--checkpoint_interval, --output_dir, --vis_images, --vis_interval,
--resume_llvip, --resume, --start_epoch, --score_thresh
```

> **The argparse object is named `p` (not `parser`).** Output dir structure: `args.output_dir / task_name_from_yaml /`

---

## Dataset Paths on Cluster

```
LLVIP:  /home/23uec571/LLVIP/LLVIP/
        visible/train, visible/test, infrared/train, infrared/test, Annotations/

FLIR:   /home/23uec571/FLIR_aligned/
        JPEGImages/FLIR_XXXXX_RGB.jpg              (RGB)
        JPEGImages/FLIR_XXXXX_PreviewData.jpeg     (Thermal)
        Annotations/FLIR_XXXXX_PreviewData.xml     (VOC XML)
        align_train.txt, align_validation.txt
```

### `MultimodalVOCDataset` signature (`voc.py`)

```python
MultimodalVOCDataset(split, rgb_dir, ir_dir, ann_dir)
# split is FIRST argument
```

---

## Current Training Status

| Experiment | Output Dir | Status | Best AP50 |
|---|---|---|---|
| V4 LLVIP | `exp4_v4_improved/exp4_v4_improved/` | Complete 100 epochs | 0.609 |
| V5 FLIR finetuned | `exp5_v5_finetuned/exp5_v5_optimized/` | In Progress (ep 21+) | 0.6131 at ep14 |
| Unimodal RGB | `experiments/unimodal/results_rgb/` | Complete 50 epochs | TBD |
| Unimodal IR | `experiments/unimodal/results_ir/` | Complete 50 epochs | TBD |

### Important Checkpoint Paths

```
V4 best:      exp4_v4_improved/exp4_v4_improved/checkpoints/best_model.pth
V5 epoch 020: exp5_v5_finetuned/exp5_v5_optimized/checkpoints/epoch_020.pth
V5 best:      exp5_v5_finetuned/exp5_v5_optimized/checkpoints/best_model.pth
```

### Resume Command (current active experiment)

```bash
python tools/train.py \
    --config              config/flir_v5.yaml \
    --version             v5 \
    --gpu                 1 \
    --epochs              100 \
    --batch_size          16 \
    --amp \
    --output_dir          exp5_v5_finetuned \
    --resume              exp5_v5_finetuned/exp5_v5_optimized/checkpoints/epoch_020.pth \
    --start_epoch         21 \
    --checkpoint_interval 20 \
    --vis_interval        20 \
    2>&1 | tee exp5_finetuned_resume.log
```

---

## Cluster Information

- **Hardware:** 8x Tesla V100-SXM2 32GB
- **Personal disk quota:** 48 GB (fills up fast — each V5 checkpoint ≈ 280 MB)
- **Conda env:** `pytorch` (Python 3.10)
- **Key packages:** PyTorch 1.x/2.0, torchvision 0.13+, CUDA 12.4

### Memory Management

```bash
quota -s                                                               # Check personal quota
nvidia-smi -i <GPU> --query-compute-apps=pid,used_memory --format=csv  # Find zombie PIDs
kill -9 <PID>                                                          # Kill zombie
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tools/train.py   # Memory frag fix
```

### GPU Selection Guide

- If GPU shows 0% utilization but high memory → zombie process, kill and reuse
- V5 batch=24 needs ~18-20 GB free; batch=16 needs ~13-15 GB free
- Always prefer a GPU with **20+ GB free**

---

## Known Bugs (All Fixed in Current Codebase)

| Bug | Description | Fix |
|---|---|---|
| Duplicate kwargs | `box_score_thresh` appeared twice in `TorchFasterRCNN` init | Removed duplicate |
| Focal loss scope | `import yaml` inside `train()` created local var shadowing global | Moved import to top |
| argparse choices | Rigid `choices=[v1,v2,v3,v4,v5]` blocked unimodal versions | Removed choices restriction |
| Dataset 0 samples | `split` argument passed in wrong position to `MultimodalVOCDataset` | Fixed argument order |
| State dict crash | `strict=False` skips missing keys but crashes on shape mismatch | Filter by shape before loading |
| Disk quota crash | 48 GB quota filled by checkpoints | Use `--checkpoint_interval 20` |
| cuDNN OOM | Zombie GPU processes holding memory | Kill with `nvidia-smi` + `kill -9` |
| argparse object name | argparse object is `p` not `parser` in `parse_args()` | Use `p` consistently |

---

## Next Steps

1. Wait for V5 FLIR finetuned experiment to complete (currently at epoch 21+)
2. Collect final metrics from all experiments (V4 LLVIP, V5 FLIR, Unimodal RGB, Unimodal IR)
3. Update comparison table in README with final numbers
4. Evaluate using COCO-standard mAP (already implemented in `utils/metrics.py`)
5. Generate final visualizations for thesis report

---

## Instructions for Claude Code

- **DO NOT modify** any existing `.py` files in `model/`, `dataset/`, `tools/`, `utils/`
- **DO NOT modify** `config/` yaml files
- You may create new files only if explicitly asked
- When suggesting fixes, always show the **exact command to run**, not the code to edit
- Always check GPU memory with `nvidia-smi` before recommending a GPU
- Always check `quota -s` before suggesting long training runs
- The argparse object in `train.py` is named `p` not `parser`
- The output directory structure is `args.output_dir / task_name_from_yaml /`
- V5 uses `num_classes=4`, NOT 3 (background counts as class 0)
