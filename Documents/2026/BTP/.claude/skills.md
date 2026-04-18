# Claude Code Skills — BTP Project

## HARD RULE: Do Not Modify Code

**DO NOT modify** any existing `.py` files in `model/`, `dataset/`, `tools/`, `utils/`.
**DO NOT modify** `config/` yaml files.
Only create new files when explicitly asked.
When suggesting fixes, show the **exact command to run**, never the code to edit.

---

## argparse Object Name

The argparse object in `tools/train.py` is named **`p`**, not `parser`.

```python
# Correct
p = argparse.ArgumentParser(...)
args = p.parse_args()

# Wrong — do NOT assume
parser = argparse.ArgumentParser(...)
```

---

## Output Directory Structure

```
args.output_dir / task_name_from_yaml /
```

Example: `--output_dir exp5_v5_finetuned` + `task_name: exp5_v5_optimized` in YAML
→ actual output at `exp5_v5_finetuned/exp5_v5_optimized/`

Current paths:
- V4: `exp4_v4_improved/exp4_v4_improved/`
- V5: `exp5_v5_finetuned/exp5_v5_optimized/`

---

## Dataset Signatures

### LLVIP — `MultimodalVOCDataset` (`dataset/voc.py`)

```python
MultimodalVOCDataset(split, rgb_dir, ir_dir, ann_dir)
# split is the FIRST argument — not keyword, positional
```

### FLIR — `FlirAlignedDataset` (`dataset/flir.py`)

Label mapping:
```python
TARGET_CLASSES = ['person', 'bicycle', 'car']
CLASS_TO_LABEL = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}
# person=1, bicycle=2, car=3, background=0
```

---

## num_classes

| Model | num_classes | Classes |
|---|---|---|
| V4 (LLVIP) | 2 | background=0, person=1 |
| V5 (FLIR) | **4** | background=0, person=1, bicycle=2, car=3 |

V5 uses `num_classes=4`, **NOT 3**. Background always counts.

---

## GPU Memory Requirements

| Config | GPU Memory Needed |
|---|---|
| V5 batch=24 | ~18-20 GB free |
| V5 batch=16 | ~13-15 GB free |

**Always run `nvidia-smi` before recommending a GPU.**

```bash
# Check all GPUs
nvidia-smi

# Find zombie processes on a specific GPU
nvidia-smi -i <GPU_ID> --query-compute-apps=pid,used_memory --format=csv

# Kill a zombie
kill -9 <PID>
```

GPU with 0% utilization but high memory = zombie → kill and reuse.
Always prefer a GPU with **20+ GB free**.

---

## Disk Quota

- Personal quota: **48 GB** (hard limit)
- Each V5 checkpoint ≈ **280 MB**
- **Always run `quota -s` before suggesting long training runs**
- Use `--checkpoint_interval 20` to avoid filling quota

```bash
quota -s   # Check current disk usage vs quota
```

---

## Dataset Paths on Cluster

```
LLVIP:  /home/23uec571/LLVIP/LLVIP/
        visible/train, visible/test
        infrared/train, infrared/test
        Annotations/

FLIR:   /home/23uec571/FLIR_aligned/
        JPEGImages/FLIR_XXXXX_RGB.jpg              (RGB)
        JPEGImages/FLIR_XXXXX_PreviewData.jpeg     (Thermal)
        Annotations/FLIR_XXXXX_PreviewData.xml     (VOC XML)
        align_train.txt, align_validation.txt
```

---

## Checkpoint Paths

```
V4 best:      exp4_v4_improved/exp4_v4_improved/checkpoints/best_model.pth
V5 epoch 020: exp5_v5_finetuned/exp5_v5_optimized/checkpoints/epoch_020.pth
V5 best:      exp5_v5_finetuned/exp5_v5_optimized/checkpoints/best_model.pth
```

---

## Conda Environment

```bash
conda activate pytorch   # Python 3.10, PyTorch 2.0, torchvision 0.13+, CUDA 12.4
```

---

## OOM Fix

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tools/train.py ...
```

---

## Known Fixed Bugs (do not re-introduce)

| Bug | Fix Applied |
|---|---|
| `box_score_thresh` duplicate kwarg | Removed duplicate in V5 init |
| `import yaml` inside `train()` shadowing global | Moved to top-level imports |
| `choices=[v1,...,v5]` blocking unimodal | Removed choices restriction |
| `split` in wrong position for `MultimodalVOCDataset` | Fixed to positional first arg |
| `strict=False` crashing on shape mismatch | Filter state dict by shape before loading |
| Checkpoints filling 48 GB quota | `--checkpoint_interval 20` |
| Zombie GPU processes causing OOM | `nvidia-smi` + `kill -9` |
| argparse object named `parser` not `p` | Use `p` throughout `train.py` |
