"""
ImageNet data utilities for U-F2-CBM.

  ImageNetDataset      : thin wrapper around ImageFolder
  SoftLabelDataset     : pre-computes and caches soft label vectors (no image labels used)
  build_soft_label_cache : one-time extraction pass
  imagenet_class_names : loads the 1000 ImageNet class name strings
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# ---------------------------------------------------------------------------
# ImageNet class names
# ---------------------------------------------------------------------------

# The 1000 ImageNet-1K class names in synset order.
# Source: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
# We embed a compact JSON-compatible list here so the code works offline.
# Keys are integer indices 0..999, values are human-readable names.

_IMAGENET_CLASS_NAMES_FILE = Path(__file__).parent / "imagenet_classes.json"


def imagenet_class_names() -> List[str]:
    """
    Return the 1000 ImageNet-1K class names in synset order (index 0..999).

    If imagenet_classes.json exists next to this file, load from there.
    Otherwise fall back to torchvision's built-in mapping.
    """
    if _IMAGENET_CLASS_NAMES_FILE.exists():
        with open(_IMAGENET_CLASS_NAMES_FILE) as f:
            data = json.load(f)
        # Support both list and dict {"0": "tench", ...}
        if isinstance(data, list):
            return data
        return [data[str(i)] for i in range(len(data))]

    # Fallback: use torchvision's IMAGENET1K_V1 meta
    try:
        from torchvision.models import ResNet50_Weights
        return ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    except Exception:
        raise RuntimeError(
            "Cannot load ImageNet class names. "
            "Either place imagenet_classes.json next to imagenet_dataset.py "
            "or install torchvision>=0.13."
        )


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------

def standard_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def standard_val_transform(image_size: int = 224) -> transforms.Compose:
    resize = int(image_size * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# ImageNetDataset
# ---------------------------------------------------------------------------

class ImageNetDataset(Dataset):
    """
    Thin wrapper around torchvision.datasets.ImageFolder for ImageNet.

    Returns (image_tensor, label_int) pairs.
    Labels are used ONLY for validation accuracy; training needs only images.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",       # "train" | "val"
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
    ):
        if transform is None:
            transform = (
                standard_train_transform(image_size)
                if split == "train"
                else standard_val_transform(image_size)
            )
        self.dataset = ImageFolder(root=root, transform=transform)
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]

    @property
    def classes(self) -> List[str]:
        return self.dataset.classes


# ---------------------------------------------------------------------------
# Soft-label cache
# ---------------------------------------------------------------------------

class SoftLabelDataset(Dataset):
    """
    Dataset that returns (image_tensor, soft_label_vector) pairs.

    The soft label o = softmax(f @ W^T) is pre-computed by running one pass
    over the training set with the frozen backbone, and cached to disk as a
    float16 .npz file (~2.5 GB for ResNet50 on ImageNet train).

    Subsequent epochs only load (image, pre-computed o) without touching
    the backbone again — only the lightweight MLP is run during training.

    Args:
        image_root   : path to ImageNet split (passed to ImageNetDataset)
        cache_path   : path to .npz cache file (created if not exists)
        backbone     : VisualBackbone instance (only needed to build cache)
        image_size   : image resize target
        device       : device for backbone during cache build
    """

    def __init__(
        self,
        image_root: str,
        cache_path: str,
        backbone=None,
        image_size: int = 224,
        device: torch.device = torch.device("cpu"),
        build_batch_size: int = 256,
        num_workers: int = 8,
    ):
        self.image_dataset = ImageNetDataset(
            root=image_root, split="train", image_size=image_size
        )
        self.cache_path = Path(cache_path)

        if not self.cache_path.exists():
            if backbone is None:
                raise ValueError(
                    f"Cache file {cache_path} does not exist and no backbone "
                    "was provided to build it."
                )
            print(f"Building soft-label cache → {cache_path}")
            build_soft_label_cache(
                dataset=self.image_dataset,
                backbone=backbone,
                save_path=str(self.cache_path),
                device=device,
                batch_size=build_batch_size,
                num_workers=num_workers,
            )

        data = np.load(str(self.cache_path))
        # soft_labels: (N, K) float16
        self.soft_labels = torch.from_numpy(data["soft_labels"].astype(np.float32))

        assert len(self.soft_labels) == len(self.image_dataset), (
            f"Cache length {len(self.soft_labels)} != "
            f"dataset length {len(self.image_dataset)}"
        )

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.image_dataset[idx]
        soft_label = self.soft_labels[idx]  # (K,)
        return img, soft_label


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_soft_label_cache(
    dataset: ImageNetDataset,
    backbone,
    save_path: str,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 8,
) -> None:
    """
    One-time pass over `dataset` to compute soft labels o = softmax(f @ W^T).
    Saves result as float16 .npz to `save_path`.

    Note: labels from the dataset are NOT used — only the images and the
    backbone's own distribution.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    backbone = backbone.to(device)
    backbone.eval()
    W = backbone.get_classifier_weights().to(device)  # (K, n)
    b = backbone.get_classifier_bias()
    if b is not None:
        b = b.to(device)

    N = len(dataset)
    K = backbone.num_classes
    all_soft = np.empty((N, K), dtype=np.float16)

    idx_start = 0
    for images, _ in tqdm(loader, desc="Building soft-label cache"):
        images = images.to(device)
        f = backbone(images)                   # (B, n)
        logits = f @ W.T
        if b is not None:
            logits = logits + b
        o = F.softmax(logits, dim=-1)          # (B, K)
        B = o.size(0)
        all_soft[idx_start : idx_start + B] = o.cpu().numpy().astype(np.float16)
        idx_start += B

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, soft_labels=all_soft)
    print(f"Soft-label cache saved → {save_path}  ({all_soft.nbytes / 1e9:.2f} GB)")
