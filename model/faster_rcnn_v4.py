"""
faster_rcnn_v4.py — Version D (Improved over V3)
CBAM at C1/C2/C3 + Learnable alpha + CrossAttention at C4 + Pedestrian anchors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection import FasterRCNN as TorchFasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        return x * self.sigmoid(a + m).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


class LearnableCBAMFusion(nn.Module):
    """alpha * RGB + (1-alpha) * IR -> CBAM -> Conv1x1. Alpha is learned."""
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.alpha    = nn.Parameter(torch.tensor(0.0))
        self.cbam     = CBAM(in_ch, reduction=reduction)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True))

    def forward(self, rgb_feat, ir_feat):
        a = torch.sigmoid(self.alpha)
        fused = a * rgb_feat + (1.0 - a) * ir_feat
        return self.out_conv(self.cbam(fused))


class CrossAttentionFusion(nn.Module):
    """Q=RGB, K/V=IR cross-attention. Safe at C4 (32x40=1280 tokens)."""
    def __init__(self, in_ch, attn_dim=256, num_heads=8):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.proj_q   = nn.Linear(in_ch, attn_dim, bias=False)
        self.proj_k   = nn.Linear(in_ch, attn_dim, bias=False)
        self.proj_v   = nn.Linear(in_ch, attn_dim, bias=False)
        self.mha      = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True, dropout=0.0)
        self.proj_out = nn.Linear(attn_dim, in_ch, bias=False)
        self.norm     = nn.LayerNorm(in_ch)
        self.conv1x1  = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True))

    def forward(self, rgb_feat, ir_feat):
        B, C, H, W = rgb_feat.shape
        rgb_seq = rgb_feat.flatten(2).permute(0, 2, 1)
        ir_seq  = ir_feat.flatten(2).permute(0, 2, 1)
        Q = self.proj_q(rgb_seq)
        K = self.proj_k(ir_seq)
        V = self.proj_v(ir_seq)
        out, _ = self.mha(Q, K, V)
        out   = self.proj_out(out)
        fused = self.norm(out + rgb_seq)
        fused = fused.permute(0, 2, 1).reshape(B, C, H, W)
        return self.conv1x1(fused)


class DualResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        rgb = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.rgb_stem   = nn.Sequential(rgb.conv1, rgb.bn1, rgb.relu, rgb.maxpool)
        self.rgb_layer1 = rgb.layer1
        self.rgb_layer2 = rgb.layer2
        self.rgb_layer3 = rgb.layer3
        self.rgb_layer4 = rgb.layer4
        ir = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.ir_stem    = nn.Sequential(ir.conv1, ir.bn1, ir.relu, ir.maxpool)
        self.ir_layer1  = ir.layer1
        self.ir_layer2  = ir.layer2
        self.ir_layer3  = ir.layer3
        self.ir_layer4  = ir.layer4

    def forward(self, rgb, ir):
        r  = self.rgb_stem(rgb)
        r1 = self.rgb_layer1(r)
        r2 = self.rgb_layer2(r1)
        r3 = self.rgb_layer3(r2)
        r4 = self.rgb_layer4(r3)
        i  = self.ir_stem(ir)
        i1 = self.ir_layer1(i)
        i2 = self.ir_layer2(i1)
        i3 = self.ir_layer3(i2)
        i4 = self.ir_layer4(i3)
        return dict(rgb_c1=r1, rgb_c2=r2, rgb_c3=r3, rgb_c4=r4,
                    ir_c1=i1,  ir_c2=i2,  ir_c3=i3,  ir_c4=i4)


class MultimodalBackboneWithFPN(nn.Module):
    """
    V4: CBAM+alpha at C1/C2/C3, CrossAttn at C4, pedestrian anchors.
    C1(256) C2(512) C3(1024) -> LearnableCBAMFusion
    C4(2048)                  -> CrossAttentionFusion
    """
    def __init__(self, fpn_out=256):
        super().__init__()
        self.body      = DualResNetBackbone()
        self.fusion_c1 = LearnableCBAMFusion(in_ch=256,  reduction=8)
        self.fusion_c2 = LearnableCBAMFusion(in_ch=512,  reduction=16)
        self.fusion_c3 = LearnableCBAMFusion(in_ch=1024, reduction=16)
        self.fusion_c4 = CrossAttentionFusion(in_ch=2048, attn_dim=256, num_heads=8)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out,
            extra_blocks=LastLevelMaxPool())
        self.out_channels = fpn_out

    def forward(self, rgb, ir):
        f  = self.body(rgb, ir)
        c1 = self.fusion_c1(f["rgb_c1"], f["ir_c1"])
        c2 = self.fusion_c2(f["rgb_c2"], f["ir_c2"])
        c3 = self.fusion_c3(f["rgb_c3"], f["ir_c3"])
        c4 = self.fusion_c4(f["rgb_c4"], f["ir_c4"])
        return self.fpn(OrderedDict([("0",c1),("1",c2),("2",c3),("3",c4)]))


class FasterRCNN(nn.Module):
    def __init__(self, model_config=None, num_classes=2):
        super().__init__()
        backbone = MultimodalBackboneWithFPN(fpn_out=256)
        anchor_gen = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5)
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0","1","2","3"], output_size=7, sampling_ratio=2)
        self.detector = TorchFasterRCNN(
            backbone=backbone, num_classes=num_classes,
            rpn_anchor_generator=anchor_gen, box_roi_pool=roi_pooler,
            box_score_thresh=0.05, box_nms_thresh=0.5,
            min_size=512, max_size=640)

    def forward(self, rgb, ir, targets=None):
        if isinstance(rgb, torch.Tensor):
            rgb_list = [rgb[i] for i in range(rgb.shape[0])]
            ir_list  = [ir[i]  for i in range(ir.shape[0])]
        else:
            rgb_list, ir_list = list(rgb), list(ir)
        rgb_batch = torch.stack(rgb_list)
        ir_batch  = torch.stack(ir_list)
        images_t, targets = self.detector.transform(rgb_list, targets)
        B, _, H, W = images_t.tensors.shape
        ir_r = F.interpolate(ir_batch, size=(H, W), mode="bilinear", align_corners=False)
        features = self.detector.backbone(images_t.tensors, ir_r)
        proposals, prop_losses = self.detector.rpn(images_t, features, targets)
        detections, det_losses = self.detector.roi_heads(
            features, proposals, images_t.image_sizes, targets)
        detections = self.detector.transform.postprocess(
            detections, images_t.image_sizes,
            [(img.shape[-2], img.shape[-1]) for img in rgb_list])
        if self.training:
            losses = {}
            losses.update(prop_losses)
            losses.update(det_losses)
            return losses
        return detections
