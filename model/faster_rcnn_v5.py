"""
faster_rcnn_v5_fixed.py — FLIR Aligned 3-class (person / car / bicycle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork, sigmoid_focal_loss
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
        return self.out_conv(self.cbam(a * rgb_feat + (1.0 - a) * ir_feat))


class CrossAttentionFusion(nn.Module):
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
        out, _  = self.mha(self.proj_q(rgb_seq), self.proj_k(ir_seq), self.proj_v(ir_seq))
        fused   = self.norm(self.proj_out(out) + rgb_seq)
        return self.conv1x1(fused.permute(0, 2, 1).reshape(B, C, H, W))


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
        r1 = self.rgb_layer1(r);  r2 = self.rgb_layer2(r1)
        r3 = self.rgb_layer3(r2); r4 = self.rgb_layer4(r3)
        i  = self.ir_stem(ir)
        i1 = self.ir_layer1(i);   i2 = self.ir_layer2(i1)
        i3 = self.ir_layer3(i2);  i4 = self.ir_layer4(i3)
        return dict(rgb_c1=r1, rgb_c2=r2, rgb_c3=r3, rgb_c4=r4,
                    ir_c1=i1,  ir_c2=i2,  ir_c3=i3,  ir_c4=i4)


class MultimodalBackboneWithFPN(nn.Module):
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
        return self.fpn(OrderedDict([("0", c1), ("1", c2), ("2", c3), ("3", c4)]))


def _focal_loss_fastrcnn(class_logits, box_regression, labels, regression_targets,
                          alpha=0.25, gamma=2.0):
    labels_t     = torch.cat(labels, dim=0)
    reg_targets  = torch.cat(regression_targets, dim=0)
    num_classes  = class_logits.shape[1]
    one_hot      = F.one_hot(labels_t, num_classes=num_classes).float()
    cls_loss     = sigmoid_focal_loss(
        class_logits, one_hot, alpha=alpha, gamma=gamma, reduction="mean")
    sampled_pos  = torch.where(labels_t > 0)[0]
    labels_pos   = labels_t[sampled_pos]
    N, _         = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos, labels_pos],
        reg_targets[sampled_pos], beta=1.0/9, reduction="sum")
    box_loss = box_loss / max(labels_t.numel(), 1)
    return cls_loss, box_loss


FLIR_ANCHOR_SIZES  = ((16,32),(32,64),(64,128),(128,256),(256,512))
FLIR_ANCHOR_RATIOS = ((0.33, 0.5, 1.0, 2.0, 3.0),) * 5


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=4, box_score_thresh=0.30, box_nms_thresh=0.45,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()

        import torchvision.models.detection.roi_heads as _rh
        _alpha, _gamma = focal_alpha, focal_gamma
        def _patched(cls_logits, box_reg, labels, reg_targets):
            return _focal_loss_fastrcnn(cls_logits, box_reg, labels, reg_targets,
                                        alpha=_alpha, gamma=_gamma)
        _rh.fastrcnn_loss = _patched

        backbone   = MultimodalBackboneWithFPN(fpn_out=256)
        anchor_gen = AnchorGenerator(sizes=FLIR_ANCHOR_SIZES, aspect_ratios=FLIR_ANCHOR_RATIOS)
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0","1","2","3"],
                                        output_size=7, sampling_ratio=2)
        self.detector = TorchFasterRCNN(
            backbone=backbone, num_classes=num_classes,
            box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh,
            rpn_anchor_generator=anchor_gen, box_roi_pool=roi_pooler,
            min_size=512, max_size=640,
            rpn_pre_nms_top_n_train=3000, rpn_pre_nms_top_n_test=1500,
            rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000)

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
        ir_r = F.interpolate(ir_batch, size=(H,W), mode="bilinear", align_corners=False)
        features   = self.detector.backbone(images_t.tensors, ir_r)
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


def load_llvip_checkpoint_for_flir(model, llvip_ckpt_path, device="cuda"):
    """Load LLVIP V4 weights into V5 model, skipping shape-mismatched layers."""
    import torch
    ckpt = torch.load(llvip_ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Filter: only load tensors whose shape EXACTLY matches current model
    model_state = model.state_dict()
    to_load  = {}
    skipped  = []
    for k, v in state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                to_load[k] = v
            else:
                skipped.append(f"  shape mismatch: {k}  ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}")
        else:
            skipped.append(f"  not in model:   {k}")

    # Load only the compatible weights
    model_state.update(to_load)
    model.load_state_dict(model_state, strict=True)

    print(f"[checkpoint] loaded  : {len(to_load)} / {len(state)} tensors from LLVIP checkpoint")
    print(f"[checkpoint] skipped : {len(skipped)} tensors (re-initialized fresh)")
    for s in skipped:
        print(s)

    # Freeze C1-C3 backbone layers (preserve low-level features from LLVIP)
    freeze_targets = [
        model.detector.backbone.body.rgb_stem,
        model.detector.backbone.body.ir_stem,
        model.detector.backbone.body.rgb_layer1,
        model.detector.backbone.body.ir_layer1,
        model.detector.backbone.body.rgb_layer2,
        model.detector.backbone.body.ir_layer2,
        model.detector.backbone.body.rgb_layer3,
        model.detector.backbone.body.ir_layer3,
    ]
    frozen = 0
    for module in freeze_targets:
        for p in module.parameters():
            p.requires_grad = False
            frozen += p.numel()
    total = sum(p.numel() for p in model.parameters())
    print(f"[freeze] {frozen:,}/{total:,} params frozen ({100*frozen/total:.1f}%) — C1-C3 locked, C4+ trainable")

    return model.to(device)


