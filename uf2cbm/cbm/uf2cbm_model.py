"""
U-F2-CBM: full inference module.

Assembles backbone + TextUnlock MLP + ConceptBank into a single nn.Module
that exposes:
  - forward(x)          → (logits, concept_activations)
  - textunlock_logits(x)→ logits from the TextUnlock-ed classifier (no CBM bottleneck)
  - get_top_concepts(x) → human-readable concept explanations

The CBM prediction pipeline (Eq. 2):
    f         = Fv(x)                    visual features          (B, n)
    f̃         = L2_norm(MLP(f))          text-space features      (B, m)
    acts      = f̃ @ C^T                  concept activations      (B, Z)
    logits    = acts @ W_con              class logits             (B, K)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UF2CBM(nn.Module):
    """
    Full U-F2-CBM inference module.

    All sub-components (backbone, MLP, concept bank) are frozen at inference.

    Args:
        backbone     : VisualBackbone (frozen)
        mlp          : TextUnlockMLP  (frozen after training)
        concept_bank : ConceptBank    (C and W_con)
        class_embeddings_U : (K, m) L2-normalised class prompt embeddings
        class_names  : list of K class name strings
    """

    def __init__(
        self,
        backbone,
        mlp,
        concept_bank,
        class_embeddings_U: torch.Tensor,
        class_names: List[str],
    ):
        super().__init__()
        self.backbone = backbone
        self.mlp = mlp
        self.concept_bank = concept_bank
        self.register_buffer("U", class_embeddings_U)   # (K, m)
        self.class_names = class_names

        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def _map_to_text_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        x → f̃ ∈ R^{B×m}, L2-normalised.
        Handles both training (backbone already produces f) and image input.
        """
        with torch.no_grad():
            f = self.backbone(x)                        # (B, n)
        f_tilde = F.normalize(self.mlp(f), dim=-1)     # (B, m)
        return f_tilde

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full U-F2-CBM forward pass.

        Returns:
            logits       : (B, K)  CBM class logits
            concept_acts : (B, Z)  concept activation scores
        """
        f_tilde = self._map_to_text_space(x)
        concept_acts = self.concept_bank.concept_activations(f_tilde)   # (B, Z)
        logits       = self.concept_bank.cbm_logits(f_tilde)            # (B, K)
        return logits, concept_acts

    def textunlock_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        TextUnlock-only classification logits (no CBM bottleneck).
        Used to verify that the MLP preserved the original accuracy.

        Returns: (B, K)
        """
        f_tilde = self._map_to_text_space(x)
        return f_tilde @ self.U.T   # (B, K)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_top_concepts(
        self,
        x: torch.Tensor,
        k: int = 10,
        use_importance: bool = True,
    ) -> list[list[tuple[str, float]]]:
        """
        Return top-k concept explanations for each image in a batch.

        If use_importance=True, ranks by  activation × W_con[:,predicted_class]
        (matches Figure 3 in the paper).
        If use_importance=False, ranks by raw concept activation only.

        Args:
            x               : (B, 3, H, W) input images
            k               : number of concepts per image
            use_importance  : weight activations by W_con

        Returns:
            list of B lists, each containing k (concept_word, score) tuples
        """
        f_tilde = self._map_to_text_space(x)                         # (B, m)
        logits, concept_acts = (
            self.concept_bank.cbm_logits(f_tilde),
            self.concept_bank.concept_activations(f_tilde),
        )
        predicted = logits.argmax(dim=-1)                             # (B,)

        results = []
        for b in range(f_tilde.size(0)):
            cls_idx = predicted[b].item()
            if use_importance:
                # importance = activation × W_con weight for predicted class
                w = self.concept_bank.W_con[:, cls_idx]               # (Z,)
                scores_vec = concept_acts[b] * w                      # (Z,)
            else:
                scores_vec = concept_acts[b]                          # (Z,)

            top_scores, top_indices = scores_vec.topk(k)
            results.append([
                (self.concept_bank.concept_words[i.item()], top_scores[j].item())
                for j, i in enumerate(top_indices)
            ])
        return results

    @torch.no_grad()
    def predict_with_explanation(
        self, x: torch.Tensor, k: int = 10
    ) -> list[dict]:
        """
        Convenience method: run inference and return a list of dicts, one per image.

        Each dict has:
            predicted_class : str
            predicted_idx   : int
            top_concepts    : list of (word, importance) tuples
        """
        logits, _ = self.forward(x)
        pred_indices = logits.argmax(dim=-1)
        explanations = self.get_top_concepts(x, k=k, use_importance=True)

        return [
            {
                "predicted_class": self.class_names[pred_indices[b].item()],
                "predicted_idx":   pred_indices[b].item(),
                "top_concepts":    explanations[b],
            }
            for b in range(x.size(0))
        ]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        mlp_ckpt_path: str,
        concept_bank_dir: str,
        backbone_name: str = "resnet50",
        text_encoder_name: str = "all-MiniLM-L12-v1",
        class_prompt_template: str = "an image of a {}",
        device: torch.device = torch.device("cpu"),
    ) -> "UF2CBM":
        """
        Load all components from disk and assemble a UF2CBM ready for inference.

        Args:
            mlp_ckpt_path      : path to .pth checkpoint saved by TextUnlockTrainer
            concept_bank_dir   : directory containing concept_bank files
            backbone_name      : backbone identifier (must match the checkpoint)
            text_encoder_name  : sentence-transformer model name
            class_prompt_template : prompt format string
            device             : inference device
        """
        from uf2cbm.models.backbones import VisualBackbone
        from uf2cbm.models.text_unlock import (
            TextUnlockMLP, load_text_encoder, encode_class_prompts
        )
        from uf2cbm.cbm.concept_bank import ConceptBank
        from uf2cbm.data.imagenet_dataset import imagenet_class_names

        # Backbone
        backbone = VisualBackbone(backbone_name).to(device)

        # MLP
        ckpt = torch.load(mlp_ckpt_path, map_location=device)
        mlp = TextUnlockMLP(
            in_dim=backbone.feature_dim,
            out_dim=ckpt["cfg"].get("mlp_out_dim", 384),
            dropout=ckpt["cfg"].get("mlp_dropout", 0.5),
        ).to(device)
        mlp.load_state_dict(ckpt["mlp_state_dict"])
        mlp.eval()
        for p in mlp.parameters():
            p.requires_grad_(False)

        # Text encoder + class embeddings
        text_encoder = load_text_encoder(text_encoder_name)
        class_names = imagenet_class_names()
        U = encode_class_prompts(
            class_names, text_encoder,
            prompt_template=class_prompt_template,
            device=device,
        )

        # Concept bank
        concept_bank = ConceptBank.load(concept_bank_dir, device=device)

        return cls(
            backbone=backbone,
            mlp=mlp,
            concept_bank=concept_bank,
            class_embeddings_U=U,
            class_names=class_names,
        ).to(device)

    def to(self, device) -> "UF2CBM":
        super().to(device)
        self.concept_bank.to(device)
        return self
