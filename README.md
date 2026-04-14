# CSSA Multimodal Faster R-CNN

Implementation of multimodal RGB-IR object detection on LLVIP dataset.

## Results
- **AP@0.50**: 80.86%
- **mAP@0.50:0.95**: 47.57%
- **Precision**: 86.48%
- **Recall**: 86.02%

## Files
- `model/faster_rcnn.py` - Model architecture
- `dataset/voc.py` - LLVIP dataset loader
- `tools/train_with_eval.py` - Training script with evaluation
- `tools/full_evaluation.py` - Final evaluation
- `tools/visualize.py` - Visualization generator
- `config/voc.yaml` - Configuration

## Usage
```bash
# Train with per-epoch evaluation
python -m tools.train_with_eval --config config/voc.yaml

# View results
python -m tools.view_results

# Evaluate final model
python -m tools.full_evaluation

# Generate visualizations
python -m tools.visualize
```
