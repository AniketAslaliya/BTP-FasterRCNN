import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class UnimodalFasterRCNN(nn.Module):
    def __init__(self, num_classes, modality="rgb", box_score_thresh=0.05, box_nms_thresh=0.5):
        super().__init__()
        self.modality = modality
        # Standard SOTA ResNet-50 FPN Baseline
        self.detector = fasterrcnn_resnet50_fpn(
            pretrained=True,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            min_size=512, max_size=640
        )
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, rgb, ir, targets=None):
        # Act entirely as a unimodal network by ignoring the other image
        inputs = rgb if self.modality == "rgb" else ir
        
        if isinstance(inputs, torch.Tensor):
            inputs_list = [inputs[i] for i in range(inputs.shape[0])]
        else:
            inputs_list = list(inputs)
            
        return self.detector(inputs_list, targets)
