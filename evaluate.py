#!/usr/bin/env python3
"""
Evaluate U-F2-CBM on ImageNet validation set.

Reports three numbers (Table 1 & 2 in the paper):
  1. Original backbone top-1 (baseline)
  2. TextUnlock-ed classifier top-1 (should be ≈ original − 0.2%)
  3. U-F2-CBM top-1 (full concept bottleneck pipeline)

MLP Ablation modes (Appendix Table 2):
  --ablation mean_feat     : replace features with dataset mean (constant input)
  --ablation random_feat   : replace features with Gaussian noise
  --ablation shuffled_feat : shuffle features across batch (breaks correspondence)
  --ablation random_weights: reinitialise MLP weights randomly

Example:
    python evaluate.py \\
        --backbone      resnet50 \\
        --mlp_ckpt      ./checkpoints/resnet50/best.pth \\
        --concept_bank  ./concept_bank/resnet50 \\
        --imagenet_val  /path/to/imagenet/val \\
        --batch_size    256

    # Ablation:
    python evaluate.py ... --ablation random_weights
"""

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from uf2cbm.models.backbones import VisualBackbone
from uf2cbm.models.text_unlock import load_text_encoder, encode_class_prompts, TextUnlockMLP
from uf2cbm.cbm.concept_bank import ConceptBank
from uf2cbm.cbm.uf2cbm_model import UF2CBM
from uf2cbm.data.imagenet_dataset import ImageNetDataset, imagenet_class_names
from uf2cbm.utils.metrics import (
    evaluate_imagenet_val,
    evaluate_original_backbone,
)

ABLATION_MODES = ("none", "mean_feat", "random_feat", "shuffled_feat", "random_weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-F2-CBM")
    parser.add_argument("--backbone",      default="resnet50")
    parser.add_argument("--mlp_ckpt",      required=True)
    parser.add_argument("--concept_bank",  required=True)
    parser.add_argument("--imagenet_val",  required=True)
    parser.add_argument("--text_encoder",  default="all-MiniLM-L12-v1")
    parser.add_argument("--batch_size",    type=int, default=256)
    parser.add_argument("--num_workers",   type=int, default=8)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--output_json",   default=None,
                        help="Optional path to save results as JSON")
    parser.add_argument("--ablation",      default="none", choices=ABLATION_MODES,
                        help="MLP ablation mode (Appendix Table 2). Default: none")
    return parser.parse_args()


@torch.no_grad()
def compute_mean_feature(backbone, val_loader, device, num_batches=50):
    """Compute approximate mean feature vector over first num_batches of val set."""
    backbone.eval()
    accum = None
    count = 0
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        images = images.to(device, non_blocking=True)
        f = backbone(images)
        if accum is None:
            accum = f.sum(0)
        else:
            accum += f.sum(0)
        count += f.size(0)
    return (accum / count)  # (feat_dim,)


@torch.no_grad()
def evaluate_ablation(model, backbone, val_loader, device, ablation, mean_feat=None):
    """
    Evaluate the CBM pipeline with an ablated MLP input/weights.

    Ablation modes:
      mean_feat     : every sample gets the same constant feature (dataset mean)
      random_feat   : feature replaced with Gaussian noise (same shape, same norm)
      shuffled_feat : features shuffled across batch positions
      random_weights: MLP weights reinitialised randomly (architecture kept)
    """
    model.eval()
    mlp   = model.mlp
    C     = model.concept_bank.C      # (Z, m)
    W_con = model.concept_bank.W_con  # (Z, K)

    if ablation == "random_weights":
        # Reinit in-place, restore after evaluation
        orig_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        for m in mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    correct1 = correct5 = total = 0

    for images, labels in tqdm(val_loader, desc=f"Evaluating [ablation={ablation}]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        f = backbone(images)  # (B, feat_dim)

        if ablation == "mean_feat":
            f = mean_feat.unsqueeze(0).expand(B, -1)
        elif ablation == "random_feat":
            f_noise = torch.randn_like(f)
            # Match the original norm so the MLP receives similar magnitude input
            orig_norm = f.norm(dim=-1, keepdim=True).mean()
            f = F.normalize(f_noise, dim=-1) * orig_norm
        elif ablation == "shuffled_feat":
            idx = torch.randperm(B, device=device)
            f = f[idx]
        # random_weights: f unchanged, MLP weights already re-inited above

        f_tilde = F.normalize(mlp(f), dim=-1)         # (B, m)
        concept_acts = f_tilde @ C.T                  # (B, Z)
        logits = concept_acts @ W_con                 # (B, K)

        pred1 = logits.argmax(dim=-1)
        correct1 += (pred1 == labels).sum().item()

        _, top5 = logits.topk(5, dim=-1)
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=-1).sum().item()
        total += B

    if ablation == "random_weights":
        mlp.load_state_dict(orig_state)

    return {
        "top1": 100.0 * correct1 / total,
        "top5": 100.0 * correct5 / total,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.ablation != "none":
        print(f"Ablation mode: {args.ablation}")

    # --- Shared components ---
    class_names  = imagenet_class_names()
    text_encoder = load_text_encoder(args.text_encoder)
    U = encode_class_prompts(class_names, text_encoder, device=device)

    # --- Validation loader ---
    val_dataset = ImageNetDataset(root=args.imagenet_val, split="val")
    val_loader  = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    results: dict = {}

    # ----------------------------------------------------------------
    # 1. Original backbone
    # ----------------------------------------------------------------
    print("\n[1/3] Evaluating original backbone...")
    backbone = VisualBackbone(args.backbone).to(device)
    orig_metrics = evaluate_original_backbone(backbone, val_loader, device)
    results["original"] = orig_metrics
    print(f"  Original top-1: {orig_metrics['top1']:.2f}%  "
          f"top-5: {orig_metrics['top5']:.2f}%")

    # ----------------------------------------------------------------
    # 2. TextUnlock-ed classifier (MLP only, no CBM bottleneck)
    # ----------------------------------------------------------------
    print("\n[2/3] Evaluating TextUnlock-ed classifier...")
    ckpt = torch.load(args.mlp_ckpt, map_location=device)
    mlp  = TextUnlockMLP(
        in_dim=backbone.feature_dim,
        out_dim=ckpt["cfg"].get("mlp_out_dim", 384),
        dropout=0.0,   # no dropout at eval
    ).to(device)
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)

    concept_bank = ConceptBank.load(args.concept_bank, device=device)

    model = UF2CBM(
        backbone=backbone,
        mlp=mlp,
        concept_bank=concept_bank,
        class_embeddings_U=U,
        class_names=class_names,
    )

    if args.ablation == "none":
        tu_metrics = evaluate_imagenet_val(model, val_loader, device, mode="textunlock")
    else:
        # Ablation skips the TextUnlock eval (not meaningful to ablate TextUnlock separately)
        tu_metrics = {"top1": float("nan"), "top5": float("nan")}

    results["textunlock"] = tu_metrics
    if args.ablation == "none":
        delta_tu = tu_metrics["top1"] - orig_metrics["top1"]
        print(f"  TextUnlock top-1: {tu_metrics['top1']:.2f}%  "
              f"(Δ={delta_tu:+.2f}%)")
    else:
        print("  TextUnlock skipped (ablation mode)")

    # ----------------------------------------------------------------
    # 3. Full U-F2-CBM (or ablated version)
    # ----------------------------------------------------------------
    print("\n[3/3] Evaluating U-F2-CBM (concept bottleneck)...")

    if args.ablation == "none":
        cbm_metrics = evaluate_imagenet_val(model, val_loader, device, mode="cbm")
    else:
        mean_feat = None
        if args.ablation == "mean_feat":
            print("  Computing dataset mean feature...")
            mean_feat = compute_mean_feature(backbone, val_loader, device).to(device)
        cbm_metrics = evaluate_ablation(model, backbone, val_loader, device,
                                        ablation=args.ablation, mean_feat=mean_feat)

    results["cbm"] = cbm_metrics
    delta_cbm = cbm_metrics["top1"] - orig_metrics["top1"]
    label = f"ablation={args.ablation}" if args.ablation != "none" else "U-F2-CBM"
    print(f"  {label} top-1: {cbm_metrics['top1']:.2f}%  "
          f"(Δ={delta_cbm:+.2f}%)")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "="*50)
    print(f"  {'Method':<20}  {'Top-1':>7}  {'Top-5':>7}")
    print("  " + "-"*40)
    for method, m in results.items():
        t1 = f"{m['top1']:>6.2f}%" if m['top1'] == m['top1'] else "    N/A"
        t5 = f"{m['top5']:>6.2f}%" if m['top5'] == m['top5'] else "    N/A"
        print(f"  {method:<20}  {t1}  {t5}")
    print("="*50)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
