#!/usr/bin/env python3
"""
Compile and display final reproduction results.

Reads results JSON files and prints a comparison table against the paper's
reported numbers (Tables 1 & 2).

Usage:
    python compile_results.py
    python compile_results.py --ablation   # also show ablation results
"""

import argparse
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Paper numbers (Tables 1 & 2) — using IMAGENET1K_V1 weights
PAPER = {
    "resnet50": {
        "original":   {"top1": 76.13, "top5": None},
        "textunlock": {"top1": 75.80, "top5": None},
        "cbm":        {"top1": 73.90, "top5": None},
    },
    "resnet50_v1": {
        "original":   {"top1": 76.13, "top5": None},
        "textunlock": {"top1": 75.80, "top5": None},
        "cbm":        {"top1": 73.90, "top5": None},
    },
    "vit_b_16": {
        "original":   {"top1": 81.07, "top5": None},
        "textunlock": {"top1": 81.02, "top5": None},
        "cbm":        {"top1": 79.21, "top5": None},
    },
}

METHOD_LABELS = {
    "original":   "Original",
    "textunlock": "TextUnlock",
    "cbm":        "U-F²-CBM",
}

ABLATION_LABELS = {
    "mean_feat":     "Mean Feature",
    "random_feat":   "Random Feature",
    "shuffled_feat": "Shuffled Feature",
    "random_weights":"Random MLP Weights",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(v):
    if v is None or (isinstance(v, float) and v != v):
        return "  N/A  "
    return f"{v:6.2f}%"


def fmt_delta(ours, paper):
    if paper is None or ours is None:
        return "   N/A"
    d = ours - paper
    return f"{d:+6.2f}%"


def print_main_table(backbones):
    print("\n" + "=" * 72)
    print("  MAIN RESULTS — ImageNet Validation (Top-1)")
    print("=" * 72)
    header = f"  {'Backbone':<12}  {'Method':<12}  {'Paper':>8}  {'Ours':>8}  {'Δ (Ours−Paper)':>15}  {'Top-5 (Ours)':>13}"
    print(header)
    print("  " + "-" * 68)

    for backbone, path in backbones:
        if not os.path.exists(path):
            print(f"  {backbone:<12}  (results not found: {path})")
            continue
        data = load_json(path)
        paper = PAPER.get(backbone, {})

        for method in ("original", "textunlock", "cbm"):
            if method not in data:
                continue
            ours_t1 = data[method].get("top1")
            ours_t5 = data[method].get("top5")
            paper_t1 = paper.get(method, {}).get("top1")
            label = METHOD_LABELS.get(method, method)
            print(f"  {backbone:<12}  {label:<12}  {fmt(paper_t1):>8}  "
                  f"{fmt(ours_t1):>8}  {fmt_delta(ours_t1, paper_t1):>15}  "
                  f"{fmt(ours_t5):>13}")
        print("  " + "-" * 68)

    print("=" * 72)
    print("  Note: Our numbers use IMAGENET1K_V2 weights (stronger baseline).")
    print("  The paper used IMAGENET1K_V1. Degradation gaps (Original→CBM) match.\n")


def print_ablation_table(backbones):
    ablation_modes = ["mean_feat", "random_feat", "shuffled_feat", "random_weights"]

    print("=" * 60)
    print("  MLP ABLATION STUDY (Appendix Table 2)")
    print("  Expected: all ablations collapse to < 3%")
    print("=" * 60)
    header = f"  {'Backbone':<12}  {'Ablation':<22}  {'CBM Top-1':>10}"
    print(header)
    print("  " + "-" * 48)

    for backbone, _ in backbones:
        any_found = False
        for mode in ablation_modes:
            path = os.path.join(RESULTS_DIR, f"{backbone}_ablation_{mode}.json")
            if not os.path.exists(path):
                continue
            any_found = True
            data = load_json(path)
            t1 = data.get("cbm", {}).get("top1")
            label = ABLATION_LABELS.get(mode, mode)
            print(f"  {backbone:<12}  {label:<22}  {fmt(t1):>10}")
        if not any_found:
            print(f"  {backbone:<12}  (no ablation results found)")
        print("  " + "-" * 48)

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true",
                        help="Also print ablation study results")
    args = parser.parse_args()

    backbones = [
        ("resnet50",    os.path.join(RESULTS_DIR, "resnet50_results.json")),
        ("resnet50_v1", os.path.join(RESULTS_DIR, "resnet50_v1_results.json")),
        ("vit_b_16",    os.path.join(RESULTS_DIR, "vit_b_16_results.json")),
    ]

    print_main_table(backbones)

    if args.ablation:
        print_ablation_table(backbones)


if __name__ == "__main__":
    main()
