#!/usr/bin/env python3
"""
Qualitative concept visualisation (reproduces Figure 3 in the paper).

For a set of ImageNet validation images, runs the U-F2-CBM and plots
the top-K concept importances as horizontal bar charts alongside the
original image.

Example:
    python visualize_concepts.py \\
        --backbone      resnet50 \\
        --mlp_ckpt      ./checkpoints/resnet50/best.pth \\
        --concept_bank  ./concept_bank/resnet50 \\
        --imagenet_val  /path/to/imagenet/val \\
        --n_images      9 \\
        --top_k         10 \\
        --output_dir    ./figures
"""

import argparse
import random
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from uf2cbm.cbm.uf2cbm_model import UF2CBM
from uf2cbm.utils.visualization import plot_concept_importance_bar, plot_concept_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise U-F2-CBM concept explanations")
    parser.add_argument("--backbone",      default="resnet50")
    parser.add_argument("--mlp_ckpt",      required=True)
    parser.add_argument("--concept_bank",  required=True)
    parser.add_argument("--imagenet_val",  required=True)
    parser.add_argument("--text_encoder",  default="all-MiniLM-L12-v1")
    parser.add_argument("--n_images",      type=int, default=9)
    parser.add_argument("--top_k",         type=int, default=10)
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--output_dir",    default="./figures")
    parser.add_argument("--device",        default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print("Loading U-F2-CBM...")
    model = UF2CBM.from_checkpoint(
        mlp_ckpt_path=args.mlp_ckpt,
        concept_bank_dir=args.concept_bank,
        backbone_name=args.backbone,
        text_encoder_name=args.text_encoder,
        device=device,
    )
    model.eval()

    # --- Collect random validation images ---
    from torchvision.datasets import ImageFolder
    transform_for_model = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_for_display = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
    ])

    dataset = ImageFolder(root=args.imagenet_val)
    n_images = min(args.n_images, len(dataset))
    indices = random.sample(range(len(dataset)), n_images)

    pil_images  = []
    tensor_imgs = []

    for idx in indices:
        path, _ = dataset.samples[idx]
        pil_img = Image.open(path).convert("RGB")
        pil_images.append(transform_for_display(pil_img))
        tensor_imgs.append(transform_for_model(pil_img))

    batch = torch.stack(tensor_imgs).to(device)

    # --- Run inference ---
    with torch.no_grad():
        explanations = model.predict_with_explanation(batch, k=args.top_k)

    # --- Per-image bar charts ---
    for i, (pil_img, expl) in enumerate(zip(pil_images, explanations)):
        names  = [c[0] for c in expl["top_concepts"]]
        scores = [c[1] for c in expl["top_concepts"]]
        save_path = output_dir / f"concepts_{i:03d}_{expl['predicted_class'].replace(' ', '_')}.png"
        plot_concept_importance_bar(
            concept_names=names,
            importance_scores=scores,
            predicted_class=expl["predicted_class"],
            save_path=str(save_path),
        )
        print(f"  [{i+1}/{args.n_images}] {expl['predicted_class']:30s} "
              f"→ top concepts: {', '.join(names[:3])}")

    # --- Grid figure (all images together) ---
    grid_path = output_dir / "concept_grid.png"
    plot_concept_grid(
        images=pil_images,
        explanations=explanations,
        save_path=str(grid_path),
        concepts_per_image=min(args.top_k, 8),
        figsize_per_image=(3, max(4, min(args.top_k, 8) * 0.4 + 1)),
    )
    print(f"\nGrid figure saved → {grid_path}")
    print(f"Individual bar charts saved → {output_dir}/concepts_*.png")


if __name__ == "__main__":
    main()
