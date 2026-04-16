"""
Visualization utilities for U-F2-CBM.

Reproduces Figure 3 from the paper: horizontal bar charts showing the
top concept activations responsible for a prediction, each weighted by
their importance score (activation × W_con weight).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for cluster)
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_concept_importance_bar(
    concept_names: List[str],
    importance_scores: List[float],
    predicted_class: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    max_concepts: int = 10,
    color: str = "#4C9BE8",
    figsize: Tuple[int, int] = (6, 4),
) -> plt.Figure:
    """
    Horizontal bar chart of top concept importances (Figure 3 style).

    Args:
        concept_names     : concept labels (highest importance first)
        importance_scores : corresponding importance values
        predicted_class   : model's predicted class name (shown in title)
        save_path         : if given, save figure to this path
        title             : override figure title
        max_concepts      : cap number of bars
        color             : bar colour
        figsize           : (width, height) in inches

    Returns:
        matplotlib Figure
    """
    n = min(len(concept_names), max_concepts)
    names  = concept_names[:n][::-1]   # reverse so highest is on top
    scores = importance_scores[:n][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n)
    bars = ax.barh(y_pos, scores, color=color, alpha=0.85, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("importance", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _title = title if title else f"prediction : {predicted_class}"
    ax.set_title(_title, fontsize=11, fontweight="bold", pad=8)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_concept_grid(
    images: List[Image.Image],
    explanations: List[dict],
    save_path: Optional[str] = None,
    concepts_per_image: int = 8,
    figsize_per_image: Tuple[int, int] = (3, 4),
) -> plt.Figure:
    """
    Grid figure showing images alongside their top concept explanations.

    Args:
        images              : list of PIL images (length B)
        explanations        : list of dicts from UF2CBM.predict_with_explanation()
                              each has 'predicted_class' and 'top_concepts'
        save_path           : optional save path
        concepts_per_image  : number of concepts to show per image
        figsize_per_image   : (width, height) per image+bar pair

    Returns:
        matplotlib Figure
    """
    B = len(images)
    fig_w = figsize_per_image[0] * 2 * B
    fig_h = figsize_per_image[1]
    fig, axes = plt.subplots(1, B * 2, figsize=(fig_w, fig_h))

    axes = axes.flatten()  # ensure 1D regardless of B

    for b in range(B):
        img_ax   = axes[b * 2]
        bar_ax   = axes[b * 2 + 1]
        expl     = explanations[b]
        concepts = expl["top_concepts"][:concepts_per_image]

        # Image
        img_ax.imshow(images[b])
        img_ax.axis("off")
        img_ax.set_title(
            f"pred: {expl['predicted_class']}", fontsize=8, fontweight="bold"
        )

        # Bar chart
        names  = [c[0] for c in concepts][::-1]
        scores = [c[1] for c in concepts][::-1]
        y_pos  = np.arange(len(names))
        bar_ax.barh(y_pos, scores, color="#4C9BE8", alpha=0.85, edgecolor="none")
        bar_ax.set_yticks(y_pos)
        bar_ax.set_yticklabels(names, fontsize=7)
        bar_ax.set_xlabel("importance", fontsize=7)
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)
        bar_ax.tick_params(axis="x", labelsize=7)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def visualize_global_concept_distribution(
    concept_words: List[str],
    class_concept_freq: torch.Tensor,   # (Z,) normalised frequency for one class
    class_name: str,
    top_k: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Bar chart of the most frequently activated concepts across all images
    of a given class (global class-wise analysis, Figure 3 in Appendix §15).

    Args:
        concept_words        : full concept vocabulary
        class_concept_freq   : (Z,) mean activation frequency for this class
        class_name           : display name of the class
        top_k                : number of concepts to show
        save_path            : optional save path
    """
    scores, indices = class_concept_freq.topk(top_k)
    names  = [concept_words[i.item()] for i in indices][::-1]
    vals   = scores.cpu().numpy()[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(top_k)
    ax.barh(y_pos, vals, color="#E8884C", alpha=0.85, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("normalised frequency", fontsize=10)
    ax.set_title(f"Global concepts — {class_name}", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
