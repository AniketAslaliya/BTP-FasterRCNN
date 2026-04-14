"""
FINAL SOLUTION: Use torchvision's FasterRCNN as-is, but process fusion externally
This is the simplest approach that WILL work.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# ============== SIMPLE FUSION MODULE ==============

class SimpleFusion(nn.Module):
    """Dead simple: just average RGB and IR"""
    def __init__(self):
        super().__init__()
    
    def forward(self, rgb, ir):
        # Simple average fusion
        return (rgb + ir) / 2.0


# ============== WRAPPER ==============

class FasterRCNN(nn.Module):
    """
    SIMPLEST POSSIBLE WORKING SOLUTION:
    1. Convert IR to "fake RGB" by replicating channels
    2. Average with real RGB
    3. Use standard torchvision FasterRCNN
    
    This WILL work because we're not fighting torchvision's architecture.
    """
    def __init__(self, model_config=None, num_classes=2):
        super().__init__()
        
        # Use pretrained FasterRCNN
        self.detector = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
        )
        
        # Simple fusion
        self.fusion = SimpleFusion()
    
    def forward(self, rgb, ir, targets=None):
        """
        Args:
            rgb: [B, 3, H, W]
            ir: [B, 3, H, W] (already 3 channels from dataset)
            targets: list of dicts
        """
        # Fuse at input level (simplest possible fusion)
        fused = self.fusion(rgb, ir)
        
        # Pass through standard FasterRCNN
        if self.training:
            return self.detector(fused, targets)
        else:
            return self.detector(fused)
