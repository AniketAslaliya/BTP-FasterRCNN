import os
import math
import random
import xml.etree.ElementTree as ET

import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


# =====================================================
# LOAD ANNOTATIONS
# =====================================================

def load_images_and_anns(ann_dir, rgb_dir, label2idx):
    infos = []
    valid_ids = set(
        f.replace(".jpg", "")
        for f in os.listdir(rgb_dir)
        if f.endswith(".jpg")
    )
    print(f"Found {len(valid_ids)} images in {rgb_dir}")
    for img_id in tqdm(valid_ids):
        ann_file = os.path.join(ann_dir, img_id + ".xml")
        if not os.path.exists(ann_file):
            continue
        tree = ET.parse(ann_file)
        root = tree.getroot()
        detections = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name != "person":
                continue
            bbox = obj.find("bndbox")
            x1 = int(float(bbox.find("xmin").text))
            y1 = int(float(bbox.find("ymin").text))
            x2 = int(float(bbox.find("xmax").text))
            y2 = int(float(bbox.find("ymax").text))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"bbox": [x1, y1, x2, y2], "label": label2idx["person"]})
        if len(detections) == 0:
            continue
        infos.append({"img_id": img_id, "detections": detections})
    print(f"Loaded {len(infos)} valid samples")
    return infos


# =====================================================
# DATASET
# =====================================================

class MultimodalVOCDataset(Dataset):

    def __init__(self, split, rgb_dir, ir_dir, ann_dir):
        self.split   = split
        self.rgb_dir = rgb_dir
        self.ir_dir  = ir_dir
        self.ann_dir = ann_dir
        self.classes   = ["background", "person"]
        self.label2idx = {c: i for i, c in enumerate(self.classes)}
        self.idx2label = {i: c for i, c in enumerate(self.classes)}
        self.infos = load_images_and_anns(ann_dir, rgb_dir, self.label2idx)
        self.new_h = 512
        self.new_w = 640
        self.resize    = torchvision.transforms.Resize((self.new_h, self.new_w))
        self.to_tensor = torchvision.transforms.ToTensor()
        self.rgb_norm  = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.ir_norm   = torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.infos)

    # ─────────────────────────────────────────────────────────
    # AUGMENTATION METHODS (train only)
    # ─────────────────────────────────────────────────────────

    def color_jitter_aug(self, rgb, ir):
        """Brightness + contrast jitter on RGB and IR separately."""
        if self.split != "train":
            return rgb, ir
        if random.random() < 0.5:
            rgb = TF.adjust_brightness(rgb, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            ir  = TF.adjust_brightness(ir,  random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            rgb = TF.adjust_contrast(rgb, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            ir  = TF.adjust_contrast(ir,  random.uniform(0.7, 1.3))
        return rgb, ir

    def random_scale_aug(self, rgb, ir, boxes):
        """Multi-scale resize [0.75–1.25x] then crop/pad back to target."""
        if self.split != "train":
            return rgb, ir, boxes
        scale = random.uniform(0.75, 1.25)
        h = int(self.new_h * scale)
        w = int(self.new_w * scale)
        rgb = TF.resize(rgb, (h, w))
        ir  = TF.resize(ir,  (h, w))
        sx, sy = w / self.new_w, h / self.new_h
        scaled = [[x1*sx, y1*sy, x2*sx, y2*sy] for x1,y1,x2,y2 in boxes]

        if scale >= 1.0:   # crop back to target
            cx = random.randint(0, max(w - self.new_w, 0))
            cy = random.randint(0, max(h - self.new_h, 0))
            rgb = TF.crop(rgb, cy, cx, self.new_h, self.new_w)
            ir  = TF.crop(ir,  cy, cx, self.new_h, self.new_w)
            out = []
            for x1, y1, x2, y2 in scaled:
                x1, x2 = max(0.0, x1-cx), min(float(self.new_w), x2-cx)
                y1, y2 = max(0.0, y1-cy), min(float(self.new_h), y2-cy)
                if x2 > x1 + 2 and y2 > y1 + 2:
                    out.append([x1, y1, x2, y2])
        else:              # pad to target
            pad_h = self.new_h - h;  pad_w = self.new_w - w
            pt = random.randint(0, pad_h); pb = pad_h - pt
            pl = random.randint(0, pad_w); pr = pad_w - pl
            rgb = TF.pad(rgb, [pl, pt, pr, pb], fill=0)
            ir  = TF.pad(ir,  [pl, pt, pr, pb], fill=0)
            out = [[x1+pl, y1+pt, x2+pl, y2+pt] for x1,y1,x2,y2 in scaled]

        return rgb, ir, out

    def random_erase_aug(self, tensor):
        """Randomly zero out 1–3 rectangular patches."""
        if self.split != "train" or random.random() > 0.3:
            return tensor
        C, H, W = tensor.shape
        for _ in range(random.randint(1, 3)):
            area   = random.uniform(0.02, 0.12) * H * W
            aspect = random.uniform(0.3, 3.3)
            eh = max(1, min(int(math.sqrt(area / aspect)), H - 1))
            ew = max(1, min(int(math.sqrt(area * aspect)), W - 1))
            ey = random.randint(0, H - eh)
            ex = random.randint(0, W - ew)
            tensor[:, ey:ey+eh, ex:ex+ew] = 0.0
        return tensor

    # ─────────────────────────────────────────────────────────
    # HORIZONTAL FLIP
    # ─────────────────────────────────────────────────────────

    def random_flip(self, rgb, ir, boxes):
        if self.split != "train":
            return rgb, ir, boxes
        if random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            ir  = ir.transpose(Image.FLIP_LEFT_RIGHT)
            new_boxes = []
            for x1, y1, x2, y2 in boxes:
                nx1 = self.new_w - x2
                nx2 = self.new_w - x1
                if nx2 > nx1 and y2 > y1:
                    new_boxes.append([nx1, y1, nx2, y2])
            boxes = new_boxes
        return rgb, ir, boxes

    # ─────────────────────────────────────────────────────────
    # MAIN LOADER
    # ─────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        info   = self.infos[idx]
        img_id = info["img_id"]

        rgb = Image.open(os.path.join(self.rgb_dir, img_id + ".jpg")).convert("RGB")
        ir  = Image.open(os.path.join(self.ir_dir,  img_id + ".jpg")).convert("RGB")

        ow, oh = rgb.size
        rgb = self.resize(rgb)
        ir  = self.resize(ir)
        sx, sy = self.new_w / ow, self.new_h / oh

        boxes = []
        for d in info["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        # ── Augmentation pipeline ──────────────────────────
        rgb, ir = self.color_jitter_aug(rgb, ir)

        rgb_s, ir_s, boxes_s = self.random_scale_aug(rgb, ir, list(boxes))
        if len(boxes_s) > 0:          # accept only if boxes survive
            rgb, ir, boxes = rgb_s, ir_s, boxes_s

        rgb, ir, boxes = self.random_flip(rgb, ir, boxes)

        # ── Tensor conversion ──────────────────────────────
        boxes  = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        rgb = self.rgb_norm(self.to_tensor(rgb))
        ir  = self.ir_norm(self.to_tensor(ir))

        rgb = self.random_erase_aug(rgb)
        ir  = self.random_erase_aug(ir)

        target = {"boxes": boxes, "labels": labels}
        return rgb, ir, target, img_id


# =====================================================
# COLLATE FUNCTION
# =====================================================

def collate_fn(batch):
    rgbs, irs, targets, _ = zip(*batch)
    return torch.stack(rgbs), torch.stack(irs), list(targets)


# =====================================================
# LLVIP DATASET (keyword-arg interface for train.py)
# =====================================================

class LLVIPDataset(MultimodalVOCDataset):
    def __init__(self, rgb_path, ir_path, ann_path, split):
        super().__init__(
            split=split, rgb_dir=rgb_path, ir_dir=ir_path, ann_dir=ann_path)
