#!/usr/bin/env python3
"""
Entry point for TextUnlock MLP training.

Example:
    python train_text_unlock.py \\
        --backbone resnet50 \\
        --imagenet_train /path/to/imagenet/train \\
        --imagenet_val   /path/to/imagenet/val \\
        --cache_dir      ./cache \\
        --checkpoint_dir ./checkpoints \\
        --epochs 30 \\
        --batch_size 256
"""

import argparse
import yaml

from uf2cbm.training.train_text_unlock import TextUnlockTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TextUnlock MLP")

    # Model
    parser.add_argument("--backbone",     default="resnet50")
    parser.add_argument("--text_encoder", default="all-MiniLM-L12-v1")
    parser.add_argument("--mlp_out_dim",  type=int,   default=384)
    parser.add_argument("--mlp_dropout",  type=float, default=0.5)

    # Data
    parser.add_argument("--imagenet_train", required=True)
    parser.add_argument("--imagenet_val",   required=True)
    parser.add_argument("--image_size",     type=int, default=224)
    parser.add_argument("--num_workers",    type=int, default=8)

    # Training
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--lr_min",        type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int,   default=1)
    parser.add_argument("--grad_clip",     type=float, default=1.0)

    # Paths
    parser.add_argument("--cache_dir",      default="./cache")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")

    # Misc
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--resume",        default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--config",        default=None,
                        help="Optional YAML config (CLI args override)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Start from defaults, then apply YAML config, then CLI overrides
    cfg = TrainConfig()

    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        for k, v in yaml_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # CLI args always win
    for k, v in vars(args).items():
        if k in ("resume", "config"):
            continue
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)

    trainer = TextUnlockTrainer(cfg)
    trainer.run(resume=args.resume)


if __name__ == "__main__":
    main()
