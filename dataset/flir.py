"""
dataset/flir.py — FLIR ADAS v2 Dataset Loader (Thermal-Only Mode)

Key design decisions:
  - RGB and thermal are from DIFFERENT video sequences in FLIR ADAS v2 (no pairing)
  - Solution: use thermal image for BOTH rgb and ir streams (unimodal thermal baseline)
  - Filter annotations to 4 target classes: person, bike, car, dog
  - Same augmentation pipeline as LLVIP for fair comparison
"""
import os
import math
import random
import json
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

# Only keep these 4 classes (ignore the other 76+ COCO categories)
TARGET_CLASSES  = ['person', 'bike', 'car', 'dog']
CLASS_TO_LABEL  = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}
# Result: person=1, bike=2, car=3, dog=4 | background=0 | num_classes=5


class FLIRDataset(Dataset):
    """
    FLIR ADAS v2 — thermal-only, 4-class filtered dataset.
    Both rgb and ir inputs receive the SAME thermal image.
    The dual-stream backbone still learns independently from each stream.
    """

    def __init__(self, rgb_path, ir_path, ann_path, split):
        """
        rgb_path  : ignored (thermal used for both streams)
        ir_path   : path to thermal images (e.g. images_thermal_train/data/)
        ann_path  : path to thermal coco.json
        split     : 'train' or 'val'
        """
        self.split  = split
        self.ir_dir = ir_path   # thermal images used for both streams

        print(f"Loading FLIR annotations: {ann_path}")
        with open(ann_path, 'r') as f:
            coco = json.load(f)

        # Build cat_id → class label mapping (filter to TARGET_CLASSES only)
        self.cat_id_to_label = {}
        self.label_to_name   = {0: 'background'}
        for cat in coco['categories']:
            if cat['name'] in CLASS_TO_LABEL:
                lbl = CLASS_TO_LABEL[cat['name']]
                self.cat_id_to_label[cat['id']] = lbl
                self.label_to_name[lbl] = cat['name']
        self.num_classes = len(TARGET_CLASSES) + 1   # 5
        print(f"Class mapping: {self.label_to_name}")

        # Build annotation index — only keep target-class annotations
        ann_by_img = {}
        skipped_anns = 0
        for ann in coco['annotations']:
            if ann.get('iscrowd', 0):
                continue
            if ann['category_id'] not in self.cat_id_to_label:
                skipped_anns += 1
                continue
            ann_by_img.setdefault(ann['image_id'], []).append(ann)
        print(f"Kept annotations: {sum(len(v) for v in ann_by_img.values())} "
              f"| Filtered out: {skipped_anns}")

        # Build valid sample list — thermal image must exist + have target annotations
        self.samples = []
        missing = 0
        for info in tqdm(coco['images']):
            img_id = info['id']
            if img_id not in ann_by_img:
                continue
            # file_name may contain subdir prefix like 'data/...'
            fname = os.path.basename(info['file_name'])
            if os.path.exists(os.path.join(ir_path, fname)):
                self.samples.append({
                    'img_id': img_id,
                    'fname':  fname,
                    'anns':   ann_by_img[img_id],
                    'orig_w': info['width'],
                    'orig_h': info['height'],
                })
            else:
                missing += 1

        print(f"Valid samples: {len(self.samples)} | Missing files: {missing}")

        # Transforms
        self.new_h     = 512
        self.new_w     = 640
        self.resize    = torchvision.transforms.Resize((self.new_h, self.new_w))
        self.to_tensor = torchvision.transforms.ToTensor()
        # Both streams normalised as IR (thermal)
        self.ir_norm   = torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.samples)

    # ── Augmentations ─────────────────────────────────────────

    def color_jitter_aug(self, img):
        if self.split != 'train':
            return img
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        return img

    def random_scale_aug(self, img, boxes, labels):
        if self.split != 'train':
            return img, boxes, labels
        scale = random.uniform(0.75, 1.25)
        h = int(self.new_h * scale)
        w = int(self.new_w * scale)
        img = TF.resize(img, (h, w))
        sx, sy = w / self.new_w, h / self.new_h
        scaled = [[x1*sx, y1*sy, x2*sx, y2*sy] for x1,y1,x2,y2 in boxes]
        if scale >= 1.0:
            cx = random.randint(0, max(w - self.new_w, 0))
            cy = random.randint(0, max(h - self.new_h, 0))
            img = TF.crop(img, cy, cx, self.new_h, self.new_w)
            ob, ol = [], []
            for i, (x1,y1,x2,y2) in enumerate(scaled):
                x1, x2 = max(0., x1-cx), min(float(self.new_w), x2-cx)
                y1, y2 = max(0., y1-cy), min(float(self.new_h), y2-cy)
                if x2 > x1+2 and y2 > y1+2:
                    ob.append([x1,y1,x2,y2]); ol.append(labels[i])
        else:
            ph = self.new_h - h; pw = self.new_w - w
            pt = random.randint(0, ph); pb = ph - pt
            pl = random.randint(0, pw); pr = pw - pl
            img   = TF.pad(img, [pl,pt,pr,pb], fill=0)
            ob    = [[x1+pl,y1+pt,x2+pl,y2+pt] for x1,y1,x2,y2 in scaled]
            ol    = list(labels)
        return img, ob, ol

    def random_flip(self, img, boxes, labels):
        if self.split != 'train':
            return img, boxes, labels
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ob, ol = [], []
            for i, (x1,y1,x2,y2) in enumerate(boxes):
                nx1, nx2 = self.new_w - x2, self.new_w - x1
                if nx2 > nx1 and y2 > y1:
                    ob.append([nx1,y1,nx2,y2]); ol.append(labels[i])
            return img, ob, ol
        return img, boxes, labels

    def random_erase_aug(self, tensor):
        if self.split != 'train' or random.random() > 0.3:
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

    # ── Main loader ────────────────────────────────────────────

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(os.path.join(self.ir_dir, s['fname'])).convert('RGB')

        ow, oh = s['orig_w'], s['orig_h']
        img = self.resize(img)
        sx, sy = self.new_w / ow, self.new_h / oh

        boxes, labels = [], []
        for ann in s['anns']:
            x, y, w, h = ann['bbox']   # COCO: x_min, y_min, w, h
            x1, y1 = x * sx, y * sy
            x2, y2 = (x + w) * sx, (y + h) * sy
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_label[ann['category_id']])

        # Augmentations (same transform applied to single thermal image)
        img = self.color_jitter_aug(img)
        img_s, b_s, l_s = self.random_scale_aug(img, list(boxes), list(labels))
        if len(b_s) > 0:
            img, boxes, labels = img_s, b_s, l_s
        img, boxes, labels = self.random_flip(img, boxes, labels)

        boxes  = torch.tensor(boxes,  dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        t = self.ir_norm(self.to_tensor(img))
        t = self.random_erase_aug(t)

        # Both streams get the same thermal tensor
        # (independent backbone weights still learn from shared input)
        return t, t, {'boxes': boxes, 'labels': labels}, s['fname']


def collate_fn(batch):
    rgbs, irs, targets, _ = zip(*batch)
    return torch.stack(rgbs), torch.stack(irs), list(targets)
