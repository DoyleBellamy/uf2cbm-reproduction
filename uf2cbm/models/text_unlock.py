"""
TextUnlock components:
  - TextUnlockMLP   : trainable MLP  f ∈ R^n  →  f̃ ∈ R^m
  - load_text_encoder()              : frozen sentence encoder T
  - encode_class_prompts()           : U ∈ R^{K×m}, L2-normalised
  - encode_texts()                   : generic batch text → embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# MLP architecture (paper Appendix §8)
# ---------------------------------------------------------------------------

class TextUnlockMLP(nn.Module):
    """
    Lightweight MLP that projects visual features f ∈ R^n into the
    text embedding space f̃ ∈ R^m.

    Architecture (Appendix §8):
        Linear(n → 2n) → LayerNorm(2n) → GELU → Dropout(p)
        Linear(2n → 2n) → LayerNorm(2n) → GELU
        Linear(2n → m)

    The output is NOT normalised here; callers must apply F.normalize(..., dim=-1).
    """

    def __init__(self, in_dim: int, out_dim: int = 384, dropout: float = 0.5):
        super().__init__()
        hidden = in_dim * 2
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            # Layer 3
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f : (B, in_dim)  visual features
        Returns:
            f̃ : (B, out_dim)  projected features (unnormalised)
        """
        return self.net(f)

    @property
    def in_dim(self) -> int:
        return self.net[0].in_features

    @property
    def out_dim(self) -> int:
        return self.net[-1].out_features


# ---------------------------------------------------------------------------
# Text encoder helpers
# ---------------------------------------------------------------------------

def load_text_encoder(model_name: str = "all-MiniLM-L12-v1") -> object:
    """
    Load a frozen sentence encoder from the sentence-transformers library.

    Returns a SentenceTransformer instance with all parameters frozen.
    The model encodes text into 384-dimensional L2-normalised vectors.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        )
    encoder = SentenceTransformer(model_name)
    # Freeze all parameters
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def encode_texts(
    texts: List[str],
    encoder,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Encode a list of strings with the frozen sentence encoder.

    Args:
        texts      : list of N strings
        encoder    : SentenceTransformer instance
        batch_size : encoding batch size (no grad, so can be large)
        device     : target device for the returned tensor
        normalize  : if True, L2-normalise each row

    Returns:
        embeddings : (N, m) float32 tensor
    """
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=str(device),
        normalize_embeddings=normalize,
    )
    return embeddings.to(device)


def encode_class_prompts(
    class_names: List[str],
    encoder,
    prompt_template: str = "an image of a {}",
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Encode K class names as text prompts and return U ∈ R^{K×m}.

    Each class name is formatted as: prompt_template.format(class_name)
    e.g. "an image of a goldfish"

    Returns L2-normalised embeddings U ∈ R^{K×m}.
    """
    prompts = [prompt_template.format(name) for name in class_names]
    U = encode_texts(prompts, encoder, batch_size=batch_size, device=device, normalize=True)
    return U  # (K, m), already L2-normalised
