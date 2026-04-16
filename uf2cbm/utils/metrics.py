"""
Evaluation metrics for U-F2-CBM.

  topk_accuracy          : compute top-k accuracy from logits + labels
  evaluate_imagenet_val  : full validation loop returning a metrics dict
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
) -> float:
    """
    Compute top-k accuracy for a batch.

    Args:
        logits : (B, K) unnormalised class scores
        labels : (B,)   ground-truth integer labels
        k      : number of top predictions to consider

    Returns:
        Accuracy in [0, 1].
    """
    with torch.no_grad():
        _, top_k = logits.topk(k, dim=-1)           # (B, k)
        correct = (top_k == labels.unsqueeze(1))    # (B, k)
        return correct.any(dim=-1).float().mean().item()


@torch.no_grad()
def evaluate_imagenet_val(
    model,
    val_loader: DataLoader,
    device: torch.device,
    mode: str = "cbm",          # "cbm" | "textunlock"
) -> Dict[str, float]:
    """
    Evaluate model on ImageNet validation data.

    Args:
        model      : UF2CBM instance  (or any object with .forward / .textunlock_logits)
        val_loader : DataLoader returning (images, labels)
        device     : inference device
        mode       : "cbm"         → use full CBM pipeline (concept bottleneck)
                     "textunlock"  → use TextUnlock-ed classifier directly (no bottleneck)

    Returns:
        dict with keys "top1" and "top5" (percentages)
    """
    model.eval()
    correct1 = correct5 = total = 0

    for images, labels in tqdm(val_loader, desc=f"Evaluating [{mode}]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mode == "cbm":
            logits, _ = model(images)
        elif mode == "textunlock":
            logits = model.textunlock_logits(images)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'cbm' or 'textunlock'.")

        B = labels.size(0)

        # top-1
        pred1 = logits.argmax(dim=-1)
        correct1 += (pred1 == labels).sum().item()

        # top-5
        _, top5 = logits.topk(5, dim=-1)
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=-1).sum().item()

        total += B

    return {
        "top1": 100.0 * correct1 / total,
        "top5": 100.0 * correct5 / total,
    }


@torch.no_grad()
def evaluate_original_backbone(
    backbone,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the original (pre-TextUnlock) backbone classifier as a baseline.
    """
    backbone.eval()
    W = backbone.get_classifier_weights().to(device)
    b = backbone.get_classifier_bias()
    if b is not None:
        b = b.to(device)

    correct1 = correct5 = total = 0

    for images, labels in tqdm(val_loader, desc="Evaluating [original]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        f = backbone(images)
        logits = f @ W.T
        if b is not None:
            logits = logits + b

        B = labels.size(0)
        pred1 = logits.argmax(dim=-1)
        correct1 += (pred1 == labels).sum().item()

        _, top5 = logits.topk(5, dim=-1)
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=-1).sum().item()
        total += B

    return {
        "top1": 100.0 * correct1 / total,
        "top5": 100.0 * correct5 / total,
    }
