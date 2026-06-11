#!/usr/bin/env python3
"""
Evaluate trained SSMT model.

Reports:
- Source test performance
- Individual target performance
- Combined target performance
"""

import argparse
import json
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

try:
    import timm
except ImportError as e:
    raise SystemExit(
        "timm is required. Install with: pip install timm"
    ) from e


# --------------------------------------------------
# Model
# --------------------------------------------------
class BackboneWithHead(nn.Module):
    def __init__(self, backbone="xception"):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = getattr(self.backbone, "num_features", 2048)

        self.classifier = nn.Linear(
            feat_dim,
            2,
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return feat, logits


# --------------------------------------------------
# Dataset
# --------------------------------------------------
def build_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])


def label_map_from_imagefolder(ds):
    name_to_idx = ds.class_to_idx

    remap = torch.empty(
        len(name_to_idx),
        dtype=torch.long,
    )

    remap[name_to_idx["real"]] = 0
    remap[name_to_idx["fake"]] = 1

    return remap


class RemapDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, remap):
        self.base_ds = base_ds
        self.remap = remap

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, int(self.remap[y])


def build_loader(
    test_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    tfm = build_transforms(img_size)

    raw = datasets.ImageFolder(
        test_dir,
        transform=tfm,
    )

    remap = label_map_from_imagefolder(raw)

    ds = RemapDataset(raw, remap)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    preds = []
    labels = []

    for x, y in loader:
        x = x.to(device)

        _, logits = model(x)

        pred = logits.argmax(1)

        preds.extend(
            pred.cpu().numpy().tolist()
        )

        labels.extend(
            y.numpy().tolist()
        )

    acc = accuracy_score(
        labels,
        preds,
    )

    f1 = f1_score(
        labels,
        preds,
    )

    return {
        "acc": float(acc * 100.0),
        "f1": float(f1 * 100.0),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        required=True,
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--combined_name",
        default="TARGET_COMBINED",
    )

    parser.add_argument(
        "--backbone",
        default="xception",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    device = torch.device(
        args.device
    )

    # -------------------------
    # Load model
    # -------------------------
    model = BackboneWithHead(
        backbone=args.backbone
    )

    try:
        ckpt = torch.load(
            args.ckpt,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        ckpt = torch.load(
            args.ckpt,
            map_location="cpu",
        )

    state = (
        ckpt["model"]
        if isinstance(ckpt, dict)
        and "model" in ckpt
        else ckpt
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)

    # -------------------------
    # Evaluate
    # -------------------------
    results = {}

    # Source
    src_loader = build_loader(
        dataset_dir / args.source / "test",
        args.img_size,
        args.batch_size,
        args.num_workers,
    )

    results[args.source] = evaluate(
        model,
        src_loader,
        device,
    )

    # Targets
    for target in args.targets:
        loader = build_loader(
            dataset_dir / target / "test",
            args.img_size,
            args.batch_size,
            args.num_workers,
        )

        results[target] = evaluate(
            model,
            loader,
            device,
        )

    # Combined
    combined_loader = build_loader(
        dataset_dir
        / args.combined_name
        / "test",
        args.img_size,
        args.batch_size,
        args.num_workers,
    )

    results[args.combined_name] = evaluate(
        model,
        combined_loader,
        device,
    )

    # -------------------------
    # Print
    # -------------------------
    print()
    print("=" * 40)
    print("Evaluation Results")
    print("=" * 40)

    for name, metrics in results.items():
        print()
        print(name)
        print(
            f"Accuracy : {metrics['acc']:.2f}"
        )
        print(
            f"F1 Score : {metrics['f1']:.2f}"
        )

    # -------------------------
    # Save
    # -------------------------
    out_file = (
        Path(args.ckpt)
        .parent
        / "test_results.json"
    )

    with open(out_file, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"Results saved to: {out_file}"
    )


if __name__ == "__main__":
    main()