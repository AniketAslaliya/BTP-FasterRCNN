import torch
"""
tools/train.py — Unified Training Script for V1 / V2 / V3 / V4 / V5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.metrics    import compute_all_metrics, MetricsLogger
from utils.logger     import get_logger
from utils.seed_utils import set_seed
from utils.visualizer import save_visualizations


def load_model(version: str, num_classes: int):
    if version == "v1":
        from model.faster_rcnn_v1 import FasterRCNN
    elif version == "v2":
        from model.faster_rcnn_v2 import FasterRCNN
    elif version == "v3":
        from model.faster_rcnn_v3 import FasterRCNN
    elif version == "v4":
        from model.faster_rcnn_v4 import FasterRCNN
    elif version == "v5":
        from model.faster_rcnn_v5 import FasterRCNN
    else:
        raise ValueError(f"Unknown version: {version}. Choose v1/v2/v3/v4/v5")
    if version == "v5":
        return FasterRCNN(num_classes=num_classes, box_score_thresh=0.30, box_nms_thresh=0.45)
    return FasterRCNN(num_classes=num_classes)


@torch.no_grad()
def evaluate(model, loader, device, score_thresh=0.35):
    model.eval()
    all_preds, all_targets = [], []
    for rgb, ir, targets in tqdm(loader, desc="  Eval", leave=False):
        rgb = rgb.to(device)
        ir  = ir.to(device)
        preds = model(rgb, ir)
        for pred in preds:
            keep = pred["scores"] >= score_thresh
            all_preds.append({
                "boxes":  pred["boxes"][keep].cpu(),
                "scores": pred["scores"][keep].cpu(),
                "labels": pred["labels"][keep].cpu(),
            })
        for t in targets:
            all_targets.append({k: v.cpu() for k, v in t.items()})
    model.train()
    return compute_all_metrics(all_preds, all_targets), all_preds


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dp = cfg["dataset_params"]
    tp = cfg["train_params"]

    epochs      = args.epochs     or tp.get("num_epochs", 100)
    batch_size  = args.batch_size or tp.get("batch_size", 8)
    lr          = args.lr         or tp.get("lr", 5e-5)
    seed        = args.seed       or tp.get("seed", 42)
    task_name   = tp.get("task_name", f"exp_{args.version}")
    num_classes = dp.get("num_classes", 2)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    out_dir  = Path(args.output_dir) / task_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train", str(out_dir))
    logger.info(f"Version: {args.version}  |  Device: {device}  |  GPU: {args.gpu}")
    logger.info(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    logger.info(f"AMP: {args.amp}")

    # ── Dataset loading ────────────────────────────────────────────────────
    if args.version == "v5":
        from dataset.flir import FLIRDataset, collate_fn
        root = getattr(args, "root_path", "/home/23uec571/FLIR_aligned")
        train_ds = FLIRDataset(root, "train")
        val_ds   = FLIRDataset(root, "val")
    else:
        from dataset.voc import MultimodalVOCDataset, collate_fn
        train_ds = MultimodalVOCDataset(
            "train", dp["rgb_train_path"], dp["ir_train_path"], dp["ann_train_path"])
        val_ds = MultimodalVOCDataset(
            "val", dp["rgb_test_path"], dp["ir_test_path"], dp["ann_test_path"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    logger.info(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # ── Model ──────────────────────────────────────────────────────────────
    model = load_model(args.version, num_classes).to(device)

    # Fine-tune from LLVIP V4 checkpoint if provided
    if args.version == "v5" and args.resume_llvip:
        from model.faster_rcnn_v5 import load_llvip_checkpoint_for_flir
        model = load_llvip_checkpoint_for_flir(model, args.resume_llvip, device=str(device))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4)

    warmup_epochs = 5
    _warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    _cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-7)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[_warmup, _cosine], milestones=[warmup_epochs])

    if args.start_epoch > 1:
        for _ in range(args.start_epoch - 1):
            scheduler.step()
        logger.info(f"Scheduler fast-forwarded to epoch {args.start_epoch}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    metrics_logger = MetricsLogger(str(out_dir))
    best_ap50 = 0.0

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(args.start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}", ncols=100)
        for rgb, ir, targets in pbar:
            rgb     = rgb.to(device)
            ir      = ir.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss_dict = model(rgb, ir, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        cur_lr   = scheduler.get_last_lr()[0]
        elapsed  = time.time() - t0
        logger.info(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | lr={cur_lr:.6f} | {elapsed:.0f}s")

        if epoch % args.checkpoint_interval == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path.name}")

        metrics, preds = evaluate(model, val_loader, device, args.score_thresh)
        metrics_logger.log(epoch, metrics, avg_loss, cur_lr)

        ap50 = metrics["AP50"]
        logger.info(
            f"  AP50={ap50:.4f} | AP75={metrics['AP75']:.4f} | "
            f"mAP={metrics['mAP']:.4f} | "
            f"Prec={metrics['Precision']:.4f} | Rec={metrics['Recall']:.4f}")

        if ap50 > best_ap50:
            best_ap50 = ap50
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            logger.info(f"  ★ New best AP50: {best_ap50:.4f}")

        if epoch % args.vis_interval == 0 or epoch == 1:
            rgb_v, ir_v, tgt_v = next(iter(val_loader))
            model.eval()
            with torch.no_grad():
                vis_preds = model(rgb_v.to(device), ir_v.to(device))
            save_visualizations(
                rgb_v, vis_preds, tgt_v,
                str(out_dir), epoch,
                n_images=args.vis_images,
                score_thresh=args.score_thresh)
            model.train()

    torch.save(model.state_dict(), out_dir / "model_final.pth")
    logger.info(f"Training complete. Best AP50: {best_ap50:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal Faster R-CNN")
    p.add_argument("--config",       required=True,  help="Path to YAML config")
    p.add_argument("--version",      required=True)
    p.add_argument("--gpu",          type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--checkpoint_interval", type=int, default=10)
    p.add_argument("--output_dir",   default="outputs")
    p.add_argument("--vis_images",   type=int,   default=12)
    p.add_argument("--vis_interval", type=int,   default=5)
    p.add_argument("--resume_llvip", type=str,   default=None,
                   help="Path to LLVIP V4 checkpoint for FLIR warm-start fine-tuning")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume training")
    p.add_argument("--start_epoch",  type=int,   default=1)
    p.add_argument("--score_thresh", type=float, default=0.35)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
