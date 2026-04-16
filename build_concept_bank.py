#!/usr/bin/env python3
"""
Build the concept bank: encode the concept word list and compute W_con.

This only needs to run once per (backbone, concept_set) combination.
No GPU needed (runs on CPU with sentence-transformers).

Example:
    python build_concept_bank.py \\
        --backbone      resnet50 \\
        --mlp_ckpt      ./checkpoints/resnet50/best.pth \\
        --concept_size  20000 \\
        --save_dir      ./concept_bank/resnet50
"""

import argparse
import torch

from uf2cbm.data.concept_words import load_concept_words, filter_concepts
from uf2cbm.data.imagenet_dataset import imagenet_class_names
from uf2cbm.models.text_unlock import load_text_encoder, encode_class_prompts
from uf2cbm.cbm.concept_bank import ConceptBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build U-F2-CBM concept bank")
    parser.add_argument("--backbone",      default="resnet50")
    parser.add_argument("--mlp_ckpt",      required=True,
                        help="Trained MLP checkpoint (used only to check config)")
    parser.add_argument("--text_encoder",  default="all-MiniLM-L12-v1")
    parser.add_argument("--concept_size",  type=int, default=20000)
    parser.add_argument("--no_wordnet",    action="store_true",
                        help="Disable WordNet filtering (faster but less rigorous)")
    parser.add_argument("--save_dir",      default="./concept_bank")
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--batch_size",    type=int, default=4096,
                        help="Encoding batch size for sentence-transformers")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Class names & class prompt embeddings U
    print("Loading ImageNet class names...")
    class_names = imagenet_class_names()

    print(f"Loading text encoder: {args.text_encoder}")
    text_encoder = load_text_encoder(args.text_encoder)

    print("Encoding class prompts...")
    U = encode_class_prompts(
        class_names, text_encoder,
        prompt_template="an image of a {}",
        device=device,
    )  # (K, m)
    print(f"Class embeddings U: {U.shape}")

    # 2. Load & filter concept words
    print(f"\nLoading {args.concept_size} concept words...")
    raw_words = load_concept_words(n=args.concept_size)
    print(f"Raw concept words: {len(raw_words)}")

    filtered_words = filter_concepts(
        raw_words,
        class_names,
        use_wordnet=not args.no_wordnet,
    )

    # 3. Build & save concept bank
    print(f"\nBuilding concept bank with {len(filtered_words)} concepts...")
    ConceptBank.build(
        concept_words=filtered_words,
        text_encoder=text_encoder,
        class_embeddings_U=U,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"\nDone. Concept bank saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
