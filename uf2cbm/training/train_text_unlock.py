"""
TextUnlock training loop.

Trains a lightweight MLP to map visual features into the text embedding space
of a frozen sentence encoder, using the original classifier's soft predictions
as supervision (knowledge distillation — no image labels required).

Loss (Eq. 1 in the paper):
    L = -Σ_i  o_i  *  log( softmax(S)_i )
where
    o  = softmax( f @ W^T )   ← original classifier distribution (teacher)
    S  = f̃ @ U^T              ← text-space logits (student)
    f̃  = L2_norm( MLP(f) )
    U  = L2_norm( T("an image of a {class}") )  ← frozen text encoder
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model
    backbone: str = "resnet50"
    text_encoder: str = "all-MiniLM-L12-v1"
    mlp_out_dim: int = 384
    mlp_dropout: float = 0.5
    class_prompt_template: str = "an image of a {}"

    # Data
    imagenet_train: str = ""
    imagenet_val: str = ""
    image_size: int = 224
    num_workers: int = 8

    # Training
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-4
    lr_min: float = 1e-6
    warmup_epochs: int = 1
    grad_clip: float = 1.0

    # Paths
    cache_dir: str = "./cache"
    checkpoint_dir: str = "./checkpoints"

    # Misc
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 100     # batches between loss prints
    val_interval: int = 1       # epochs between validation

    @property
    def cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.backbone}_soft_labels.npz")

    @property
    def ckpt_dir(self) -> Path:
        return Path(self.checkpoint_dir) / self.backbone


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TextUnlockTrainer:

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self._build_components()

    # ------------------------------------------------------------------
    def _build_components(self):
        cfg = self.cfg

        # --- Backbone (frozen) ---
        from uf2cbm.models.backbones import VisualBackbone
        print(f"Loading backbone: {cfg.backbone}")
        self.backbone = VisualBackbone(cfg.backbone).to(self.device)

        # --- Text encoder (frozen) ---
        from uf2cbm.models.text_unlock import load_text_encoder, encode_class_prompts
        print(f"Loading text encoder: {cfg.text_encoder}")
        self.text_encoder = load_text_encoder(cfg.text_encoder)

        # --- Class prompts U ∈ R^{K×m} ---
        from uf2cbm.data.imagenet_dataset import imagenet_class_names
        self.class_names = imagenet_class_names()
        print(f"Encoding {len(self.class_names)} class prompts...")
        self.U = encode_class_prompts(
            self.class_names,
            self.text_encoder,
            prompt_template=cfg.class_prompt_template,
            device=self.device,
        ).clone()  # (K, m) — .clone() exits inference_mode context from sentence-transformers
        print(f"U shape: {self.U.shape}")

        # --- MLP (trainable) ---
        from uf2cbm.models.text_unlock import TextUnlockMLP
        self.mlp = TextUnlockMLP(
            in_dim=self.backbone.feature_dim,
            out_dim=cfg.mlp_out_dim,
            dropout=cfg.mlp_dropout,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"MLP parameters: {n_params:,}")

        # --- Soft-label training dataset ---
        from uf2cbm.data.imagenet_dataset import SoftLabelDataset
        print("Preparing training dataset (soft labels)...")
        self.train_dataset = SoftLabelDataset(
            image_root=cfg.imagenet_train,
            cache_path=cfg.cache_path,
            backbone=self.backbone,
            image_size=cfg.image_size,
            device=self.device,
            build_batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # --- Validation dataset ---
        from uf2cbm.data.imagenet_dataset import ImageNetDataset
        self.val_dataset = ImageNetDataset(
            root=cfg.imagenet_val,
            split="val",
            image_size=cfg.image_size,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        # --- Optimizer & scheduler ---
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=cfg.lr)

        total_steps = cfg.epochs * len(self.train_loader)
        warmup_steps = cfg.warmup_epochs * len(self.train_loader)

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps - warmup_steps,
                    eta_min=cfg.lr_min,
                ),
            ],
            milestones=[warmup_steps],
        )

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        # Checkpoint dir
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.start_epoch = 0

    # ------------------------------------------------------------------
    def _kd_loss(
        self, f_batch: torch.Tensor, o_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Knowledge-distillation cross-entropy loss (Eq. 1).

        f_batch : (B, n)  visual features from images
        o_batch : (B, K)  soft teacher distribution (float32)
        """
        f_tilde = F.normalize(self.mlp(f_batch), dim=-1)   # (B, m)
        S = f_tilde @ self.U.T                              # (B, K)
        log_pred = F.log_softmax(S, dim=-1)                 # (B, K)
        loss = -(o_batch * log_pred).sum(dim=-1).mean()
        return loss

    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> float:
        self.mlp.train()
        self.backbone.eval()

        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, (images, soft_labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            soft_labels = soft_labels.to(self.device, non_blocking=True)

            # Extract visual features (backbone is frozen, no grad needed)
            with torch.no_grad():
                f = self.backbone(images)   # (B, n)

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                loss = self._kd_loss(f, soft_labels)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                avg = total_loss / (batch_idx + 1)
                print(
                    f"  Epoch [{epoch+1}/{self.cfg.epochs}] "
                    f"Step [{batch_idx+1}/{n_batches}] "
                    f"Loss: {avg:.4f}  LR: {lr:.2e}"
                )

        return total_loss / n_batches

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> dict:
        """
        Evaluate on ImageNet validation set.
        Returns top-1 and top-5 accuracy of the TextUnlock-ed classifier.
        """
        self.mlp.eval()
        self.backbone.eval()

        correct1 = correct5 = total = 0

        for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            f = self.backbone(images)                           # (B, n)
            f_tilde = F.normalize(self.mlp(f), dim=-1)         # (B, m)
            logits = f_tilde @ self.U.T                         # (B, K)

            # top-1
            pred1 = logits.argmax(dim=-1)
            correct1 += (pred1 == labels).sum().item()

            # top-5
            _, top5 = logits.topk(5, dim=-1)
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=-1).sum().item()

            total += labels.size(0)

        return {
            "top1": 100.0 * correct1 / total,
            "top5": 100.0 * correct5 / total,
        }

    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "mlp_state_dict": self.mlp.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_top1": val_acc,
            "cfg": self.cfg.__dict__,
        }
        path = self.cfg.ckpt_dir / f"epoch_{epoch+1:03d}.pth"
        torch.save(ckpt, path)
        if is_best:
            best_path = self.cfg.ckpt_dir / "best.pth"
            torch.save(ckpt, best_path)
            print(f"  New best checkpoint saved → {best_path}  (top-1: {val_acc:.2f}%)")

    def load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.mlp.load_state_dict(ckpt["mlp_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_acc = ckpt.get("val_top1", 0.0)
        print(f"Resumed from {ckpt_path} (epoch {self.start_epoch}, best top-1: {self.best_val_acc:.2f}%)")

    # ------------------------------------------------------------------
    def run(self, resume: Optional[str] = None) -> None:
        if resume:
            self.load_checkpoint(resume)

        print(f"\nStarting TextUnlock training for {self.cfg.backbone}")
        print(f"  Epochs: {self.cfg.epochs}   Batch size: {self.cfg.batch_size}")
        print(f"  LR: {self.cfg.lr}   Warmup: {self.cfg.warmup_epochs} epoch(s)\n")

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            elapsed = time.time() - t0

            print(f"Epoch [{epoch+1}/{self.cfg.epochs}]  "
                  f"Train loss: {train_loss:.4f}  "
                  f"({elapsed:.0f}s)")

            if (epoch + 1) % self.cfg.val_interval == 0:
                metrics = self.validate()
                top1 = metrics["top1"]
                top5 = metrics["top5"]
                print(f"  Val top-1: {top1:.2f}%  top-5: {top5:.2f}%")

                is_best = top1 > self.best_val_acc
                self.best_val_acc = max(self.best_val_acc, top1)
                self.save_checkpoint(epoch, top1, is_best)

        print(f"\nTraining complete. Best val top-1: {self.best_val_acc:.2f}%")
