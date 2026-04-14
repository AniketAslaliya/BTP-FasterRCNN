"""
utils/visualizer.py — Detection Visualization

Saves side-by-side GT (green) vs Prediction (red) bounding boxes on RGB images.
"""

import os
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
import cv2


def denormalize_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised RGB tensor [3,H,W] → uint8 numpy [H,W,3] BGR for cv2."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_boxes(img: np.ndarray, boxes: torch.Tensor,
               color: tuple, thickness: int = 2) -> np.ndarray:
    for box in boxes.cpu().tolist():
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def save_visualizations(
    rgb_batch: torch.Tensor,
    predictions: List[Dict],
    targets: List[Dict],
    output_dir: str,
    epoch: int,
    n_images: int = 12,
    score_thresh: float = 0.35,
):
    """
    Save up to n_images detection visualizations for an epoch.

    GT boxes = green, Predicted boxes = red.
    """
    vis_dir = Path(output_dir) / "visualizations" / f"epoch_{epoch:03d}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_images, len(rgb_batch))
    for i in range(n):
        img = denormalize_rgb(rgb_batch[i])

        # Draw GT in green
        if len(targets[i]["boxes"]) > 0:
            img = draw_boxes(img, targets[i]["boxes"], color=(0, 255, 0))

        # Draw predictions above score_thresh in red
        pred_boxes = predictions[i]["boxes"]
        pred_scores = predictions[i]["scores"]
        keep = pred_scores >= score_thresh
        if keep.any():
            img = draw_boxes(img, pred_boxes[keep], color=(0, 0, 255))

        cv2.imwrite(str(vis_dir / f"img{i:03d}.png"), img)
