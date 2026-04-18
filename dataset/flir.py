import os
import xml.etree.ElementTree as ET
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.functional as TF

TARGET_CLASSES = ['person', 'bicycle', 'car']
CLASS_TO_LABEL = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}

class FLIRDataset(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        split_file = "align_train.txt" if split == 'train' else "align_validation.txt"
        split_path = os.path.join(root, split_file)
        
        self.ids = []
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if 'FLIR_' in line:
                    # Extracts '00258' from 'FLIR_00258_PreviewData'
                    parts = line.split('_')
                    if len(parts) > 1:
                        self.ids.append(parts[1])
        
        self.new_h, self.new_w = 512, 640
        self.resize = torchvision.transforms.Resize((self.new_h, self.new_w))
        self.to_tensor = torchvision.transforms.ToTensor()
        self.rgb_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.ir_norm = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.ids)

    def _parse_xml(self, xml_path, sx, sy):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if name in CLASS_TO_LABEL:
                xmlbox = obj.find('bndbox')
                boxes.append([
                    float(xmlbox.find('xmin').text) * sx,
                    float(xmlbox.find('ymin').text) * sy,
                    float(xmlbox.find('xmax').text) * sx,
                    float(xmlbox.find('ymax').text) * sy
                ])
                labels.append(CLASS_TO_LABEL[name])
        return boxes, labels

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        rgb_path = os.path.join(self.root, 'JPEGImages', f"FLIR_{img_id}_RGB.jpg")
        ir_path  = os.path.join(self.root, 'JPEGImages', f"FLIR_{img_id}_PreviewData.jpeg")
        xml_path = os.path.join(self.root, 'Annotations', f"FLIR_{img_id}_PreviewData.xml")

        rgb_img = Image.open(rgb_path).convert('RGB')
        ir_img  = Image.open(ir_path).convert('RGB')
        ow, oh = rgb_img.size
        sx, sy = self.new_w / ow, self.new_h / oh
        
        rgb_img, ir_img = self.resize(rgb_img), self.resize(ir_img)
        boxes, labels = self._parse_xml(xml_path, sx, sy)
        
        if not boxes: # Skip empty annotations after filtering dog
            return self.__getitem__((idx + 1) % len(self.ids))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return self.rgb_norm(self.to_tensor(rgb_img)), self.ir_norm(self.to_tensor(ir_img)), {"boxes": boxes, "labels": labels}, f"FLIR_{img_id}"

def collate_fn(batch):
    # This format is required by our train.py (stacked tensors, not lists)
    rgbs, irs, targets, _ = zip(*batch)
    return torch.stack(rgbs), torch.stack(irs), list(targets)
