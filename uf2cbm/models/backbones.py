"""
Visual backbone wrappers for U-F2-CBM.

Each backbone wraps a pretrained torchvision classifier so that:
  - forward(x) returns visual features f  (B, n)  — the penultimate layer output
  - get_classifier_weights() returns W     (K, n)  — the frozen linear head weights
  - get_classifier_bias()    returns b     (K,)    — the frozen linear head bias (or None)

All backbone parameters (including W) are frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass
class BackboneSpec:
    feature_dim: int      # n
    num_classes: int      # K


# Maps backbone name → spec.  Weights are loaded inside VisualBackbone.__init__.
BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "resnet50":           BackboneSpec(feature_dim=2048, num_classes=1000),
    "resnet50_v1":        BackboneSpec(feature_dim=2048, num_classes=1000),
    "resnet101":          BackboneSpec(feature_dim=2048, num_classes=1000),
    "resnet101_v1":       BackboneSpec(feature_dim=2048, num_classes=1000),
    "wide_resnet50_2":    BackboneSpec(feature_dim=2048, num_classes=1000),
    "wide_resnet101_2":   BackboneSpec(feature_dim=2048, num_classes=1000),
    "resnext50_32x4d":    BackboneSpec(feature_dim=2048, num_classes=1000),
    "resnext101_64x4d":   BackboneSpec(feature_dim=2048, num_classes=1000),
    "densenet161":        BackboneSpec(feature_dim=2208, num_classes=1000),
    "efficientnet_v2_m":  BackboneSpec(feature_dim=1280, num_classes=1000),
    "convnext_base":      BackboneSpec(feature_dim=1024, num_classes=1000),
    "vit_b_16":           BackboneSpec(feature_dim=768,  num_classes=1000),
    "vit_l_16":           BackboneSpec(feature_dim=1024, num_classes=1000),
    "swin_b":             BackboneSpec(feature_dim=1024, num_classes=1000),
}


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------

class VisualBackbone(nn.Module):
    """
    Wraps a pretrained torchvision visual classifier.

    Usage:
        backbone = VisualBackbone("resnet50")
        f = backbone(images)              # (B, n)
        W = backbone.get_classifier_weights()  # (K, n)
    """

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        if name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}"
            )
        spec = BACKBONE_REGISTRY[name]
        self.name = name
        self.feature_dim = spec.feature_dim
        self.num_classes = spec.num_classes

        # --- load pretrained model ---
        model, W, b = _load_torchvision_model(name, pretrained)

        self.encoder = model          # outputs f ∈ R^n
        # Store classifier weights/bias as non-trainable buffers
        self.register_buffer("_W", W)          # (K, n)
        if b is not None:
            self.register_buffer("_b", b)      # (K,)
        else:
            self._b = None

        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return visual features f ∈ R^{B×n}."""
        with torch.no_grad():
            return self.encoder(x)

    def get_classifier_weights(self) -> torch.Tensor:
        """Return frozen classifier weight matrix W ∈ R^{K×n}."""
        return self._W

    def get_classifier_bias(self) -> Optional[torch.Tensor]:
        """Return frozen classifier bias b ∈ R^K, or None."""
        return self._b

    def get_logits(self, f: torch.Tensor) -> torch.Tensor:
        """Compute raw logits from features: f @ W^T + b  → (B, K)."""
        logits = f @ self._W.T
        if self._b is not None:
            logits = logits + self._b
        return logits

    def __repr__(self) -> str:
        return (
            f"VisualBackbone({self.name}, "
            f"feature_dim={self.feature_dim}, "
            f"num_classes={self.num_classes})"
        )


# ---------------------------------------------------------------------------
# Internal: torchvision model factory
# ---------------------------------------------------------------------------

def _load_torchvision_model(
    name: str, pretrained: bool
) -> tuple[nn.Module, torch.Tensor, Optional[torch.Tensor]]:
    """
    Load a torchvision model, extract (W, b) from the classifier head,
    replace the head with nn.Identity(), and return (encoder, W, b).
    """
    import torchvision.models as tvm

    W: torch.Tensor
    b: Optional[torch.Tensor] = None

    # ---- ResNet family ----
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        W, b = _extract_linear(m.fc)
        m.fc = nn.Identity()

    elif name == "resnet50_v1":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        W, b = _extract_linear(m.fc)
        m.fc = nn.Identity()

    elif name == "resnet101":
        m = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V2)
        W, b = _extract_linear(m.fc)
        m.fc = nn.Identity()

    elif name == "resnet101_v1":
        m = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)
        W, b = _extract_linear(m.fc)
        m.fc = nn.Identity()

    elif name == "wide_resnet50_2":
        m = tvm.wide_resnet50_2(weights=tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        W, b = _extract_linear(m.fc)
        m.fc = nn.Identity()

    elif name == "wide_resnet101_2":
        m = tvm.wide_resnet101_2(weights=tvm.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
        W, b = _extract_linear(m.fc)
        m.fc = nn.Identity()

    elif name == "resnext50_32x4d":
        m = tvm.resnext50_32x4d(weights=tvm.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        W, b = _extract_linear(m.fc)
        m.fc = nn.Identity()

    elif name == "resnext101_64x4d":
        m = tvm.resnext101_64x4d(weights=tvm.ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        W, b = _extract_linear(m.fc)
        m.fc = nn.Identity()

    # ---- DenseNet ----
    elif name == "densenet161":
        m = tvm.densenet161(weights=tvm.DenseNet161_Weights.IMAGENET1K_V1)
        W, b = _extract_linear(m.classifier)
        m.classifier = nn.Identity()

    # ---- EfficientNet ----
    elif name == "efficientnet_v2_m":
        m = tvm.efficientnet_v2_m(weights=tvm.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        # classifier is Sequential([Dropout, Linear])
        linear = m.classifier[1]
        W, b = _extract_linear(linear)
        m.classifier = nn.Sequential(m.classifier[0], nn.Identity())

    # ---- ConvNeXt ----
    elif name == "convnext_base":
        m = tvm.convnext_base(weights=tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        # classifier is Sequential([LayerNorm2d, Flatten, Linear])
        linear = m.classifier[2]
        W, b = _extract_linear(linear)
        m.classifier[2] = nn.Identity()

    # ---- ViT ----
    elif name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        W, b = _extract_linear(m.heads.head)
        m.heads.head = nn.Identity()

    elif name == "vit_l_16":
        m = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.IMAGENET1K_V1)
        W, b = _extract_linear(m.heads.head)
        m.heads.head = nn.Identity()

    # ---- Swin ----
    elif name == "swin_b":
        m = tvm.swin_b(weights=tvm.Swin_B_Weights.IMAGENET1K_V1)
        W, b = _extract_linear(m.head)
        m.head = nn.Identity()

    else:
        raise ValueError(f"No factory for backbone '{name}'")

    # Freeze entire model
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()

    return m, W.detach().clone(), b.detach().clone() if b is not None else None


def _extract_linear(
    layer: nn.Linear,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract weight and bias tensors from a nn.Linear layer."""
    W = layer.weight.data.clone()       # (K, n)
    b = layer.bias.data.clone() if layer.bias is not None else None
    return W, b
