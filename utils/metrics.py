"""
utils/metrics.py — Detection Metrics

Computes AP50, AP75, AP90, COCO mAP, Precision, Recall.
Logs per-epoch results to a CSV file.
"""

import csv
import os
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np


def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes [N,4] and [M,4]."""
    ax1, ay1, ax2, ay2 = box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3]
    bx1, by1, bx2, by2 = box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3]

    ix1 = torch.max(ax1[:, None], bx1[None, :])
    iy1 = torch.max(ay1[:, None], by1[None, :])
    ix2 = torch.min(ax2[:, None], bx2[None, :])
    iy2 = torch.min(ay2[:, None], by2[None, :])

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter

    return inter / union.clamp(min=1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """11-point interpolated AP (PASCAL VOC style)."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_rec = precisions[recalls >= thr]
        ap += prec_at_rec.max() if len(prec_at_rec) > 0 else 0.0
    return ap / 11.0


def evaluate_detections(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a batch of detections at a single IoU threshold.

    Args:
        predictions : list of dicts with 'boxes' [N,4], 'scores' [N], 'labels' [N]
        targets     : list of dicts with 'boxes' [M,4], 'labels' [M]
        iou_threshold: IoU threshold for TP/FP matching

    Returns dict with keys: ap, precision, recall
    """
    all_scores, all_tp, all_fp = [], [], []
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        pred_boxes  = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes    = gt["boxes"]

        total_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue
        if len(gt_boxes) == 0:
            all_scores.extend(pred_scores.cpu().tolist())
            all_tp.extend([0] * len(pred_boxes))
            all_fp.extend([1] * len(pred_boxes))
            continue

        iou = compute_iou(pred_boxes.cpu(), gt_boxes.cpu())  # [N, M]
        matched_gt = set()

        # Sort by descending score
        order = pred_scores.argsort(descending=True)
        for i in order.cpu().tolist():
            score = pred_scores[i].item()
            best_iou, best_j = iou[i].max(0)
            best_j = best_j.item()

            if best_iou.item() >= iou_threshold and best_j not in matched_gt:
                all_tp.append(1)
                all_fp.append(0)
                matched_gt.add(best_j)
            else:
                all_tp.append(0)
                all_fp.append(1)
            all_scores.append(score)

    if total_gt == 0 or len(all_scores) == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0}

    # Sort by score descending
    order    = np.argsort(all_scores)[::-1]
    tp_cum   = np.cumsum(np.array(all_tp)[order])
    fp_cum   = np.cumsum(np.array(all_fp)[order])

    recalls    = tp_cum / (total_gt + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    ap        = compute_ap(recalls, precisions)
    precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
    recall    = float(recalls[-1])    if len(recalls)    > 0 else 0.0

    return {"ap": ap, "precision": precision, "recall": recall}


def compute_all_metrics(
    predictions: List[Dict],
    targets: List[Dict],
) -> Dict[str, float]:
    """
    Compute AP50, AP75, AP90, COCO mAP, Precision@50, Recall@50.
    """
    thresholds = np.arange(0.50, 0.96, 0.05)
    aps = []
    for thr in thresholds:
        result = evaluate_detections(predictions, targets, iou_threshold=float(thr))
        aps.append(result["ap"])

    r50 = evaluate_detections(predictions, targets, iou_threshold=0.50)

    return {
        "AP50":      aps[0],
        "AP75":      aps[5],
        "AP90":      aps[8],
        "mAP":       float(np.mean(aps)),
        "Precision": r50["precision"],
        "Recall":    r50["recall"],
    }


class MetricsLogger:
    """Writes per-epoch metrics to a CSV file."""

    FIELDS = ["epoch", "AP50", "AP75", "AP90", "mAP", "Precision", "Recall",
              "train_loss", "lr"]

    def __init__(self, output_dir: str):
        self.csv_path = Path(output_dir) / "metrics" / "metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, epoch: int, metrics: Dict, train_loss: float = 0.0, lr: float = 0.0):
        row = {"epoch": epoch, "train_loss": f"{train_loss:.4f}", "lr": f"{lr:.6f}"}
        row.update({k: f"{v:.4f}" for k, v in metrics.items()})
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)
