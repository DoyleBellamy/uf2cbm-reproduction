"""
ConceptBank: encodes a word list with the frozen text encoder and builds
the unsupervised concept-to-class classifier weight matrix.

Key outputs:
  C       ∈ R^{Z×m}   concept embeddings (L2-normalised)
  W_con   ∈ R^{Z×K}   concept-to-class weights = C @ U^T
                        (pure text-to-text cosine similarity — no training)

Usage:
    bank = ConceptBank.build(
        concept_words, text_encoder, class_embeddings_U, save_dir="./concept_bank"
    )
    # or load a previously built bank:
    bank = ConceptBank.load("./concept_bank")

    concept_acts = bank.concept_activations(f_tilde)   # (B, Z)
    logits       = bank.cbm_logits(f_tilde)            # (B, K)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


_EMBEDDINGS_FILE = "concept_embeddings.pt"
_WCON_FILE       = "W_con.pt"
_WORDS_FILE      = "concept_words.json"


class ConceptBank:
    """
    Stores concept embeddings C and the unsupervised classifier W_con.

    All tensors are kept on the device passed at load/build time.
    """

    def __init__(
        self,
        concept_words: List[str],
        C: torch.Tensor,        # (Z, m) L2-normalised concept embeddings
        W_con: torch.Tensor,    # (Z, K) concept-to-class weights
    ):
        assert len(concept_words) == C.shape[0], (
            f"words length {len(concept_words)} != C rows {C.shape[0]}"
        )
        assert C.shape[0] == W_con.shape[0], (
            f"C rows {C.shape[0]} != W_con rows {W_con.shape[0]}"
        )
        self.concept_words = concept_words
        self.C = C          # (Z, m)
        self.W_con = W_con  # (Z, K)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_concepts(self) -> int:
        return len(self.concept_words)

    @property
    def num_classes(self) -> int:
        return self.W_con.shape[1]

    @property
    def text_dim(self) -> int:
        return self.C.shape[1]

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def concept_activations(self, f_tilde: torch.Tensor) -> torch.Tensor:
        """
        Compute concept activations for a batch of text-space image features.

        Args:
            f_tilde : (B, m) L2-normalised mapped visual features
        Returns:
            acts    : (B, Z) cosine similarity scores with each concept
        """
        return f_tilde @ self.C.T   # (B, Z)

    def cbm_logits(self, f_tilde: torch.Tensor) -> torch.Tensor:
        """
        Full U-F2-CBM classification logits (Eq. 2 in the paper).

        S_cn = (f̃ @ C^T) @ W_con
             = f̃ @ (C^T C) @ U^T      ← gram-matrix form

        Args:
            f_tilde : (B, m) L2-normalised mapped visual features
        Returns:
            logits  : (B, K)
        """
        concept_acts = self.concept_activations(f_tilde)   # (B, Z)
        logits = concept_acts @ self.W_con                  # (B, K)
        return logits

    def top_concepts(
        self, f_tilde: torch.Tensor, k: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the top-k concept indices and their activation scores
        for a single image (f_tilde shape (m,) or (1, m)).

        Returns:
            indices : (k,)  indices into concept_words
            scores  : (k,)  activation values
        """
        if f_tilde.dim() == 1:
            f_tilde = f_tilde.unsqueeze(0)
        acts = self.concept_activations(f_tilde).squeeze(0)  # (Z,)
        scores, indices = acts.topk(k)
        return indices, scores

    def importance_scores(
        self, f_tilde: torch.Tensor, class_idx: int, k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Return top-k concepts for a given image and predicted class,
        weighted by concept activation × W_con weight (Figure 3 style).

        Args:
            f_tilde   : (m,) or (1, m)  L2-normalised mapped visual feature
            class_idx : predicted class index
            k         : number of concepts to return

        Returns:
            list of (concept_word, importance) sorted descending
        """
        if f_tilde.dim() == 1:
            f_tilde = f_tilde.unsqueeze(0)
        acts = self.concept_activations(f_tilde).squeeze(0)          # (Z,)
        w    = self.W_con[:, class_idx]                               # (Z,)
        importance = acts * w                                          # (Z,)
        scores, indices = importance.topk(k)
        return [
            (self.concept_words[i.item()], scores[j].item())
            for j, i in enumerate(indices)
        ]

    # ------------------------------------------------------------------
    # Build & persist
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        concept_words: List[str],
        text_encoder,
        class_embeddings_U: torch.Tensor,   # (K, m) L2-normalised
        save_dir: Optional[str] = None,
        batch_size: int = 4096,
        device: torch.device = torch.device("cpu"),
    ) -> "ConceptBank":
        """
        Encode `concept_words` with the frozen text encoder and build W_con.

        Args:
            concept_words       : Z concept strings
            text_encoder        : frozen SentenceTransformer
            class_embeddings_U  : (K, m) L2-normalised class prompt embeddings
            save_dir            : if given, save C, W_con, and words to disk
            batch_size          : encoding batch size
            device              : target device

        Returns:
            ConceptBank instance
        """
        print(f"Encoding {len(concept_words)} concept words...")
        from uf2cbm.models.text_unlock import encode_texts
        C = encode_texts(
            concept_words,
            text_encoder,
            batch_size=batch_size,
            device=device,
            normalize=True,
        )  # (Z, m)

        U = class_embeddings_U.to(device)  # (K, m)

        # Unsupervised concept-to-class weights: W_con = C @ U^T  ∈ R^{Z×K}
        print("Building unsupervised concept-to-class weights (W_con = C @ U^T)...")
        W_con = C @ U.T   # (Z, K)

        bank = cls(concept_words=concept_words, C=C, W_con=W_con)

        if save_dir is not None:
            bank.save(save_dir)

        return bank

    def save(self, save_dir: str) -> None:
        """Persist C, W_con, and concept_words to disk."""
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)

        torch.save(self.C.cpu(), d / _EMBEDDINGS_FILE)
        torch.save(self.W_con.cpu(), d / _WCON_FILE)
        with open(d / _WORDS_FILE, "w") as f:
            json.dump(self.concept_words, f)

        print(f"ConceptBank saved → {save_dir}  "
              f"(Z={self.num_concepts}, K={self.num_classes}, m={self.text_dim})")

    @classmethod
    def load(
        cls,
        save_dir: str,
        device: torch.device = torch.device("cpu"),
    ) -> "ConceptBank":
        """Load a previously built ConceptBank from disk."""
        d = Path(save_dir)

        C     = torch.load(d / _EMBEDDINGS_FILE, map_location=device)
        W_con = torch.load(d / _WCON_FILE,       map_location=device)
        with open(d / _WORDS_FILE) as f:
            concept_words = json.load(f)

        print(f"ConceptBank loaded from {save_dir}  "
              f"(Z={len(concept_words)}, K={W_con.shape[1]}, m={C.shape[1]})")
        return cls(concept_words=concept_words, C=C, W_con=W_con)

    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "ConceptBank":
        """Move tensors to device in-place and return self."""
        self.C     = self.C.to(device)
        self.W_con = self.W_con.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"ConceptBank(Z={self.num_concepts}, "
            f"K={self.num_classes}, m={self.text_dim})"
        )
