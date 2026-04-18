# Multimodal Faster R-CNN for Thermal-RGB Object Detection
A dual-stream multimodal Faster R-CNN architecture for object detection using aligned RGB and Infrared (Thermal) image pairs. Evaluated on two benchmark datasets: **LLVIP** (pedestrian detection) and **FLIR Aligned ADAS v2** (multi-class: person, car, bicycle).
---
## Architecture Overview
### V4 — LLVIP Pedestrian Detection
- **Backbone:** Dual ResNet-50 (one stream per modality)
- **Fusion:** LearnableCBAMFusion at C1/C2/C3 + CrossAttentionFusion at C4
- **FPN:** 5-level Feature Pyramid Network
- **Dataset:** LLVIP (12,025 train / 3,463 val)
- **Classes:** 1 (person)
- **Best AP50:** 0.609 mAP
### V5 — FLIR Multi-Class Detection
- **Backbone:** Same dual ResNet-50 as V4 (architecture unchanged)
- **Fusion:** Same CBAM + CrossAttention fusion (identical to V4)
- **Loss:** Focal Loss (α=0.25, γ=2.0) via monkey-patch on RoI head
- **Anchors:** Multi-scale FLIR-tuned `((16,32),(32,64),(64,128),(128,256),(256,512))` with ratios `(0.33, 0.5, 1.0, 2.0, 3.0)`
- **NMS:** `box_score_thresh=0.30`, `box_nms_thresh=0.45`
- **Dataset:** FLIR Aligned ADAS v2 (4,129 train / 1,013 val)
- **Classes:** 3 (person=1, bicycle=2, car=3) + background=0
- **Status:** Training in progress (Epoch 21+, AP50=0.6131 at Epoch 14)
---
## Project Structure
BTP/ ├── model/ │ ├── faster_rcnn_v1.py # Baseline single-stream │ ├── faster_rcnn_v2.py # Early fusion │ ├── faster_rcnn_v3.py # Mid fusion │ ├── faster_rcnn_v4.py # CBAM + CrossAttention (LLVIP SOTA) │ ├── faster_rcnn_v5.py # FLIR 3-class + Focal Loss (current) │ └── unimodal.py # Unimodal RGB/IR baselines ├── dataset/ │ ├── voc.py # LLVIP VOC-format loader │ └── flir.py # FLIR Aligned VOC-format loader ├── tools/ │ └── train.py # Unified training script ├── config/ │ ├── voc_v4.yaml # LLVIP V4 config │ └── flir_v5.yaml # FLIR V5 config ├── utils/ │ ├── metrics.py # COCO-standard AP metrics │ ├── logger.py # Training logger │ ├── seed_utils.py # Reproducibility │ └── visualizer.py # Detection visualizations └── experiments/ └── unimodal/ # Unimodal RGB and IR baselines

---
## Dataset Setup
### LLVIP Dataset
/home/23uec571/LLVIP/LLVIP/ ├── visible/ │ ├── train/ # 12,025 RGB images │ └── test/ # 3,463 RGB images ├── infrared/ │ ├── train/ # 12,025 IR images │ └── test/ # 3,463 IR images └── Annotations/ # VOC XML annotations (person only)

### FLIR Aligned ADAS v2
/home/23uec571/FLIR_aligned/ ├── JPEGImages/ │ ├── FLIR_XXXXX_RGB.jpg # RGB images │ └── FLIR_XXXXX_PreviewData.jpeg # Thermal images ├── Annotations/ │ └── FLIR_XXXXX_PreviewData.xml # VOC XML (person/car/bicycle) ├── align_train.txt └── align_validation.txt

---
## Training Commands
### Train V4 (LLVIP Pedestrian Detection)
```bash
python tools/train.py \
    --config    config/voc_v4.yaml \
    --version   v4 \
    --gpu       4 \
    --epochs    100 \
    --batch_size 24 \
    --amp \
    --output_dir exp4_v4_improved \
    2>&1 | tee v4_train.log
Train V5 (FLIR Multi-Class, fine-tuned from V4)
bash
python tools/train.py \
    --config              config/flir_v5.yaml \
    --version             v5 \
    --gpu                 6 \
    --epochs              100 \
    --batch_size          24 \
    --amp \
    --output_dir          exp5_v5_finetuned \
    --resume_llvip        exp4_v4_improved/exp4_v4_improved/checkpoints/best_model.pth \
    2>&1 | tee exp5_finetuned.log
Resume V5 Training
bash
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
Train Unimodal Baselines (LLVIP)
bash
# RGB Baseline
python tools/train.py --config config/voc_v4.yaml --version unimodal_rgb \
    --gpu 6 --epochs 50 --batch_size 24 --amp \
    --output_dir experiments/unimodal/results_rgb 2>&1 | tee unimodal_rgb.log
# IR Baseline
python tools/train.py --config config/voc_v4.yaml --version unimodal_ir \
    --gpu 7 --epochs 50 --batch_size 24 --amp \
    --output_dir experiments/unimodal/results_ir 2>&1 | tee unimodal_ir.log
Results Summary
ModelDatasetClassesAP50mAPNotes
Faster R-CNN (RGB only)LLVIP1~0.52~0.25Unimodal baseline
Faster R-CNN (IR only)LLVIP1~0.55~0.27Unimodal baseline
V4 (Ours)LLVIP10.6090.31Dual-stream CBAM+CrossAttn
V5 (Ours, ep.14)FLIR30.6130.295Fine-tuned from V4
Key Design Decisions
Why Focal Loss for FLIR?
FLIR has severe class imbalance (background >> objects). Standard Cross-Entropy floods gradients with easy background examples. Focal Loss (α=0.25, γ=2.0) down-weights easy negatives and forces the model to focus on hard positives like occluded cars and distant bicycles.

Why re-cluster anchors for FLIR?
LLVIP pedestrians: tall bounding boxes, ratio ~1:3
FLIR cars: wide bounding boxes, ratio ~3:1
Using LLVIP anchors on FLIR would miss most car proposals entirely.

Why freeze C1-C3 during FLIR fine-tuning?
Low-level features (edges, textures, thermal gradients) are domain-general and already perfectly learned from LLVIP. Freezing them prevents catastrophic forgetting while allowing C4+ to specialize for multi-class FLIR objects.

Memory Management Tips (HPC Cluster)
bash
# Check your personal disk quota
quota -s
# Find zombie processes on a GPU
nvidia-smi -i <GPU_ID> --query-compute-apps=pid,used_memory --format=csv
# Kill a zombie process
kill -9 <PID>
# Run training inside screen to survive SSH disconnects
screen -S training_session
# Detach: Ctrl+A then D
# Reattach: screen -r training_session
# Launch with memory fragmentation fix
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tools/train.py ...
Citation / References
Faster R-CNN: Ren et al., NeurIPS 2015
CBAM: Woo et al., ECCV 2018
Focal Loss: Lin et al. (RetinaNet), ICCV 2017
LLVIP Dataset: Jia et al., ICCV 2021
FLIR ADAS Dataset: FLIR Systems, 2020
CSSA (Baseline): Cao et al., CVPR 2023 EOF echo "✅ README.md created!"
### Step 2: Create context.md
```bash
cat > /home/23uec571/BTP/context.md << 'EOF'
# Project Context & Session Notes
## Objective
Build a multimodal Faster R-CNN that achieves SOTA detection performance on:
1. **LLVIP** — Pedestrian detection using RGB+IR pairs (V4)
2. **FLIR Aligned ADAS v2** — Multi-class detection: person, car, bicycle (V5)
---
## Architecture Philosophy
V4 and V5 are the **same core architecture**. V5 is V4 adapted for a different dataset:
- Same dual ResNet-50 backbone
- Same CBAM + CrossAttention fusion modules
- Only dataset-specific parameters changed (anchors, NMS thresholds, num_classes)
This is consistent with how Detectron2-based papers (e.g., CSSA, CVPR 2023) handle multi-dataset evaluation — dataset-specific hyperparameters are treated as config, not architecture changes.
---
## V5 Key Changes vs V4
| Parameter | V4 (LLVIP) | V5 (FLIR) | Reason |
|---|---|---|---|
| num_classes | 2 | 4 | +car, +bicycle |
| box_score_thresh | 0.05 | 0.30 | Reduce false positives |
| box_nms_thresh | 0.50 | 0.45 | Tighter car suppression |
| Anchor sizes | Single per level | Multi per level | Cover wider size range |
| Anchor ratios | (0.5,1.0,1.5,2.0,3.0) | (0.33,0.5,1.0,2.0,3.0) | 0.33 for tall persons |
| Loss function | Cross Entropy | Focal Loss α=0.25 γ=2.0 | Class imbalance |
| RPN proposals | Default | 3000/2000 train, 1500/1000 test | Better recall |
---
## FLIR Dataset Label Mapping
```python
TARGET_CLASSES = ['person', 'bicycle', 'car']
CLASS_TO_LABEL = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}
# person=1, bicycle=2, car=3, background=0 (implicit)
num_classes=4 in model (background + 3 object classes)

Fine-Tuning Strategy (V4 → V5)
Load V4 LLVIP best checkpoint (strict=False with shape-mismatch filtering)
Skip mismatched layers: RPN head (anchor count changed) + RoI classifier (num_classes changed)
Freeze C1-C3 backbone layers (23.4% of params)
Train C4, fusion modules, FPN, RPN head, classifier head
LR = 5e-6 (10x lower than LLVIP training LR of 5e-5)
Current Experiment Status
ExperimentDirStatusBest AP50
V4 LLVIP/exp4_v4_improved✅ Complete (100 epochs)0.609
V5 FLIR alignedexp5_v5_aligned✅ Complete (100 epochs)TBD
V5 FLIR optimizedexp5_v5_optimized✅ CompleteTBD
V5 FLIR finetunedexp5_v5_finetuned🔄 In Progress (ep 21+)0.6131 (ep14)
Unimodal RGB_experiments/unimodal/results_rgb✅ Complete (50 epochs)TBD
Unimodal IR_flir_yolov5.ipynb experiments/unimodal/results_ir✅ Complete (50 epochs)TBD
Common Issues & Fixes
cuDNN Algorithm Error
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
Cause: GPU OOM — not enough free memory for cuDNN workspace Fix: Kill zombie processes on GPU, or switch to a cleaner GPU

Disk Quota Crash During torch.save
RuntimeError: PytorchStreamWriter failed writing file data/522: file write failed
Cause: Personal quota hit 100% (quota is ~48GB on this cluster) Fix: Delete intermediate epoch_*.pth from completed experiments. Keep only best_model.pth and model_final.pth

NameError: train_ds referenced before assignment
Cause: Dataset initialization block missing for the given version Fix: Ensure both v5 (FLIR) and else (LLVIP via MultimodalVOCDataset) branches exist in train.py

shape mismatch in load_state_dict
Cause: V4 checkpoint has different RPN/head shapes than V5 Fix: Filter state_dict by shape compatibility before loading (implemented in load_llvip_checkpoint_for_flir)

GPU Cluster Notes
8x Tesla V100-SXM2 32GB GPUs (GPU 0-7)
Personal quota: 48 GB per user
Each V5 checkpoint: ~280 MB
V5 model memory usage: ~15-19 GB at batch 24, ~12-14 GB at batch 16
Always check quota -s before long training runs
Always run inside screen to survive SSH disconnects
Files Modified in This Project
FilePurpose
model/faster_rcnn_v5.pyV5 architecture with Focal Loss, tuned anchors, fine-tune loader
dataset/flir.pyFLIR Aligned dataset loader (VOC XML format)
tools/train.pyUnified training script with v5 support and resume_llvip flag
config/flir_v5.yamlFLIR V5 training configuration
model/unimodal.pyUnimodal RGB/IR baseline wrapper
