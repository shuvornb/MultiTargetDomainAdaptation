#!/usr/bin/env python3
"""
Generate t-SNE plots for a trained SSMT Xception model.

The plot visualizes feature alignment between:
- Source fake
- Source real
- Target fake
- Target real

Target samples are taken from the union of all provided target test sets.

Example:
python src/tsne_ssmt_xception.py \
    --dataset_dir dataset_ssmt_DF__FS_F2F_seed123 \
    --ckpt runs/ssmt_DF__FS_F2F_seed123/checkpoint_last.pt \
    --source DF \
    --targets FS F2F \
    --img_size 128 \
    --out_png results/figures/tsne_df_fs_f2f.png
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

try:
    import timm
except ImportError as e:
    raise SystemExit("timm is required. Install with: pip install timm") from e


# --------------------------------------------------
# Model
# --------------------------------------------------
class BackboneWithHead(nn.Module):
    def __init__(self, backbone: str = "xception"):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = getattr(self.backbone, "num_features", 2048)
        self.classifier = nn.Linear(feat_dim, 2)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return feat, logits


# --------------------------------------------------
# Data
# --------------------------------------------------
def build_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])


def label_map_from_imagefolder(ds: datasets.ImageFolder) -> torch.Tensor:
    name_to_idx = ds.class_to_idx

    if "real" not in name_to_idx or "fake" not in name_to_idx:
        raise RuntimeError(
            f"Expected class folders named 'real' and 'fake', got: {name_to_idx}"
        )

    remap = torch.empty(len(name_to_idx), dtype=torch.long)
    remap[name_to_idx["real"]] = 0
    remap[name_to_idx["fake"]] = 1

    return remap


class RemapDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, remap: torch.Tensor):
        self.base_ds = base_ds
        self.remap = remap

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, int(self.remap[y])


def build_loader(
    split_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transform = build_transforms(img_size)

    raw = datasets.ImageFolder(
        split_dir,
        transform=transform,
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
# Feature extraction
# --------------------------------------------------
@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_feats = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        feats, _ = model(x)

        all_feats.append(feats.cpu().numpy())
        all_labels.append(y.numpy())

    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def subsample_by_label(
    features: np.ndarray,
    labels: np.ndarray,
    label: int,
    max_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    idx = np.where(labels == label)[0]

    if len(idx) == 0:
        return np.empty((0, features.shape[1]), dtype=features.dtype)

    if max_samples > 0 and len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)

    return features[idx]


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)

    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--targets", required=True, nargs="+", type=str)

    parser.add_argument("--backbone", type=str, default="xception")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--max_per_group",
        type=int,
        default=2000,
        help=(
            "Maximum number of samples per group: "
            "source fake, source real, target fake, target real. "
            "Use 0 to use all samples."
        ),
    )

    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_lr", type=float, default=200.0)
    parser.add_argument("--tsne_iters", type=int, default=1500)

    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title.",
    )

    parser.add_argument(
        "--out_png",
        required=True,
        type=str,
        help="Output path for the generated t-SNE PNG figure.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    device = torch.device(args.device)

    print("=== t-SNE Feature Visualization ===")
    print(f"dataset_dir   : {dataset_dir}")
    print(f"checkpoint    : {args.ckpt}")
    print(f"source        : {args.source}")
    print(f"targets       : {', '.join(args.targets)}")
    print(f"device        : {device}")
    print(f"img_size      : {args.img_size}")
    print(f"max_per_group : {args.max_per_group}")
    print()

    # -------------------------
    # Load model
    # -------------------------
    model = BackboneWithHead(
        backbone=args.backbone,
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

    model.load_state_dict(state, strict=True)
    model.to(device)

    # -------------------------
    # Extract source features
    # -------------------------
    source_loader = build_loader(
        dataset_dir / args.source / "test",
        args.img_size,
        args.batch_size,
        args.num_workers,
    )

    source_feats, source_labels = extract_features(
        model,
        source_loader,
        device,
    )

    # -------------------------
    # Extract target features
    # -------------------------
    target_feats_list = []
    target_labels_list = []

    for target in args.targets:
        loader = build_loader(
            dataset_dir / target / "test",
            args.img_size,
            args.batch_size,
            args.num_workers,
        )

        feats, labels = extract_features(
            model,
            loader,
            device,
        )

        target_feats_list.append(feats)
        target_labels_list.append(labels)

    target_feats = np.concatenate(target_feats_list, axis=0)
    target_labels = np.concatenate(target_labels_list, axis=0)

    # -------------------------
    # Build four visualization groups
    # label convention: real=0, fake=1
    # -------------------------
    rng = np.random.RandomState(args.seed)

    source_fake = subsample_by_label(
        source_feats,
        source_labels,
        label=1,
        max_samples=args.max_per_group,
        rng=rng,
    )

    source_real = subsample_by_label(
        source_feats,
        source_labels,
        label=0,
        max_samples=args.max_per_group,
        rng=rng,
    )

    target_fake = subsample_by_label(
        target_feats,
        target_labels,
        label=1,
        max_samples=args.max_per_group,
        rng=rng,
    )

    target_real = subsample_by_label(
        target_feats,
        target_labels,
        label=0,
        max_samples=args.max_per_group,
        rng=rng,
    )

    print("Samples used for t-SNE:")
    print(f"  Source fake : {len(source_fake)}")
    print(f"  Source real : {len(source_real)}")
    print(f"  Target fake : {len(target_fake)}")
    print(f"  Target real : {len(target_real)}")
    print()

    features = np.vstack([
        source_fake,
        source_real,
        target_fake,
        target_real,
    ])

    group_labels = (
        ["source_fake"] * len(source_fake)
        + ["source_real"] * len(source_real)
        + ["target_fake"] * len(target_fake)
        + ["target_real"] * len(target_real)
    )

    group_labels = np.array(group_labels)

    # L2 normalize features before t-SNE
    features = features / (
        np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    )

    # -------------------------
    # Run t-SNE
    # -------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        learning_rate=args.tsne_lr,
        max_iter=args.tsne_iters,
        init="pca",
        random_state=args.seed,
        verbose=1,
    )

    coords = tsne.fit_transform(features)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(8, 6), dpi=250)

    # Colors:
    #   fake = red
    #   real = blue
    # Markers:
    #   source = +
    #   target = o
    plt.scatter(
        coords[group_labels == "source_fake", 0],
        coords[group_labels == "source_fake", 1],
        c="red",
        marker="+",
        s=28,
        alpha=0.85,
        label="Source Domain Fake",
    )

    plt.scatter(
        coords[group_labels == "source_real", 0],
        coords[group_labels == "source_real", 1],
        c="blue",
        marker="+",
        s=28,
        alpha=0.85,
        label="Source Domain Real",
    )

    plt.scatter(
        coords[group_labels == "target_fake", 0],
        coords[group_labels == "target_fake", 1],
        c="red",
        marker="o",
        s=8,
        alpha=0.45,
        label="Target Domain Fake",
    )

    plt.scatter(
        coords[group_labels == "target_real", 0],
        coords[group_labels == "target_real", 1],
        c="blue",
        marker="o",
        s=8,
        alpha=0.45,
        label="Target Domain Real",
    )

    if args.title:
        plt.title(args.title, fontsize=14, fontweight="bold")
    else:
        plt.title(
            f"{args.source} → {', '.join(args.targets)}",
            fontsize=14,
            fontweight="bold",
        )

    plt.xticks([])
    plt.yticks([])
    plt.legend(frameon=True, fontsize=8, loc="best")
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE plot to: {out_png}")


if __name__ == "__main__":
    main()