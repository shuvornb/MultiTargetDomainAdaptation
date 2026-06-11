#!/usr/bin/env python3
"""
Train Xception for Single-Source Multi-Target Domain Adaptation (SSMT-DA).

Supports:
- One source domain
- Two or more target domains
- Optional labeled target data, including zero labeled target samples
- BCE supervised loss
- Moment matching loss
- Unsupervised class-alignment loss
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import timm
except ImportError as e:
    raise SystemExit("timm is required. Install with: pip install timm") from e


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# -------------------------
# Utilities
# -------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


# -------------------------
# Loss functions
# -------------------------
def covariance(feat: torch.Tensor) -> torch.Tensor:
    feat = feat - feat.mean(dim=0, keepdim=True)
    return (feat.t() @ feat) / max(feat.shape[0] - 1, 1)


def moment_matching_loss(
    source_feat: torch.Tensor,
    target_feats: List[torch.Tensor],
) -> torch.Tensor:
    """
    Align:
    - source with each target
    - each target with every other target
    using mean and covariance matching.
    """
    loss = torch.tensor(0.0, device=source_feat.device)

    mu_s = source_feat.mean(dim=0)
    cov_s = covariance(source_feat)

    target_mus = [f.mean(dim=0) for f in target_feats]
    target_covs = [covariance(f) for f in target_feats]

    # Source-to-target alignment
    for mu_t, cov_t in zip(target_mus, target_covs):
        loss = loss + F.mse_loss(mu_s, mu_t)
        loss = loss + F.mse_loss(cov_s, cov_t)

    # Target-to-target alignment
    for i in range(len(target_feats)):
        for j in range(i + 1, len(target_feats)):
            loss = loss + F.mse_loss(target_mus[i], target_mus[j])
            loss = loss + F.mse_loss(target_covs[i], target_covs[j])

    return loss


@torch.no_grad()
def update_prototypes(
    prototypes: Dict[int, Optional[torch.Tensor]],
    feats: torch.Tensor,
    labels: torch.Tensor,
    ema: float = 0.95,
) -> None:
    for c in [0, 1]:
        mask = labels == c
        if mask.any():
            proto = feats[mask].mean(dim=0).detach()
            if prototypes[c] is None:
                prototypes[c] = proto.clone()
            else:
                prototypes[c] = ema * prototypes[c] + (1.0 - ema) * proto


def class_alignment_entropy(
    unlabeled_feats: torch.Tensor,
    proto_real: torch.Tensor,
    proto_fake: torch.Tensor,
    temp: float = 0.07,
) -> torch.Tensor:
    feats = F.normalize(unlabeled_feats, dim=1)
    proto_real = F.normalize(proto_real.unsqueeze(0), dim=1)
    proto_fake = F.normalize(proto_fake.unsqueeze(0), dim=1)

    logits = torch.cat(
        [feats @ proto_real.t(), feats @ proto_fake.t()],
        dim=1,
    ) / temp

    probs = F.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
    return entropy


# -------------------------
# Model
# -------------------------
class BackboneWithHead(nn.Module):
    def __init__(self, backbone: str = "xception", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = getattr(self.backbone, "num_features", 2048)
        self.classifier = nn.Linear(feat_dim, 2)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return feat, logits


# -------------------------
# Data
# -------------------------
def build_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def label_map_from_imagefolder(ds: datasets.ImageFolder) -> torch.Tensor:
    name_to_idx = ds.class_to_idx

    if "real" not in name_to_idx or "fake" not in name_to_idx:
        raise RuntimeError(
            f"Expected subfolders named 'real' and 'fake', got: {name_to_idx}"
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


class UnlabeledFolder(torch.utils.data.Dataset):
    def __init__(self, folder: Path, transform):
        self.paths = sorted([
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])

        if not self.paths:
            raise RuntimeError(f"No unlabeled images found in: {folder}")

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_labeled_loader(
    folder: Path,
    transform,
    batch_size: int,
    num_workers: int,
) -> Optional[DataLoader]:
    if count_images(folder) == 0:
        return None

    raw = datasets.ImageFolder(folder, transform=transform)
    remap = label_map_from_imagefolder(raw)
    ds = RemapDataset(raw, remap)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )


def build_loaders(args):
    dataset_dir = Path(args.dataset_dir)
    transform = build_transforms(args.img_size)

    # Source labeled loader
    source_train_dir = dataset_dir / args.source / "train"
    source_raw = datasets.ImageFolder(source_train_dir, transform=transform)
    source_remap = label_map_from_imagefolder(source_raw)
    source_loader = DataLoader(
        RemapDataset(source_raw, source_remap),
        batch_size=args.batch_source,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    target_labeled_loaders = []
    target_unlabeled_loaders = []

    for target in args.targets:
        labeled_dir = dataset_dir / target / "train" / "labeled"
        unlabeled_dir = dataset_dir / target / "train" / "unlabeled"

        labeled_loader = build_labeled_loader(
            labeled_dir,
            transform,
            args.batch_t_l,
            args.num_workers,
        )

        unlabeled_loader = DataLoader(
            UnlabeledFolder(unlabeled_dir, transform),
            batch_size=args.batch_t_u,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
        )

        target_labeled_loaders.append(labeled_loader)
        target_unlabeled_loaders.append(unlabeled_loader)

    return source_loader, target_labeled_loaders, target_unlabeled_loaders


# -------------------------
# Training
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--targets", required=True, nargs="+", type=str)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=500)

    parser.add_argument("--batch_source", type=int, default=16)
    parser.add_argument("--batch_t_l", type=int, default=8)
    parser.add_argument("--batch_t_u", type=int, default=16)

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--backbone", type=str, default="xception")
    parser.add_argument("--pretrained", action="store_true", default=True)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_mm", type=float, default=1.0)
    parser.add_argument("--lambda_ca", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    if len(args.targets) < 1:
        raise ValueError("At least one target domain must be provided.")

    seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("=== Train SSMT Xception ===")
    print(f"dataset_dir     : {args.dataset_dir}")
    print(f"out_dir         : {out_dir}")
    print(f"source          : {args.source}")
    print(f"targets         : {', '.join(args.targets)}")
    print(f"device          : {device}")
    print(f"backbone        : {args.backbone}")
    print(f"img_size        : {args.img_size}")
    print(f"epochs          : {args.epochs}")
    print(f"steps_per_epoch : {args.steps_per_epoch}")
    print(f"batch_source    : {args.batch_source}")
    print(f"batch_t_l       : {args.batch_t_l}")
    print(f"batch_t_u       : {args.batch_t_u}")
    print(f"lambda_mm       : {args.lambda_mm}")
    print(f"lambda_ca       : {args.lambda_ca}")
    print("================================\n")

    source_loader, target_labeled_loaders, target_unlabeled_loaders = build_loaders(args)

    print("Target labeled availability:")
    for target, loader in zip(args.targets, target_labeled_loaders):
        print(f"  {target}: {'available' if loader is not None else 'none'}")
    print()

    source_iter = infinite_loader(source_loader)
    target_labeled_iters = [
        infinite_loader(loader) if loader is not None else None
        for loader in target_labeled_loaders
    ]
    target_unlabeled_iters = [
        infinite_loader(loader)
        for loader in target_unlabeled_loaders
    ]

    model = BackboneWithHead(
        backbone=args.backbone,
        pretrained=args.pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and device.type == "cuda")
    )

    prototypes: Dict[int, Optional[torch.Tensor]] = {0: None, 1: None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()

        running = {
            "bce": 0.0,
            "mm": 0.0,
            "ca": 0.0,
            "total": 0.0,
        }

        for step in range(1, args.steps_per_epoch + 1):
            x_s, y_s = next(source_iter)
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)

            target_labeled_batches = []
            for it in target_labeled_iters:
                if it is None:
                    target_labeled_batches.append(None)
                else:
                    x_t_l, y_t_l = next(it)
                    target_labeled_batches.append((
                        x_t_l.to(device, non_blocking=True),
                        y_t_l.to(device, non_blocking=True),
                    ))

            target_unlabeled_batches = []
            for it in target_unlabeled_iters:
                x_t_u = next(it)
                target_unlabeled_batches.append(
                    x_t_u.to(device, non_blocking=True)
                )

            if (step - 1) % args.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=(args.amp and device.type == "cuda")
            ):
                f_s, logits_s = model(x_s)

                # Source BCE
                loss_bce = criterion(logits_s, y_s)

                # Target labeled BCE
                labeled_loss_terms = []
                target_labeled_feats: List[Optional[torch.Tensor]] = []

                for batch in target_labeled_batches:
                    if batch is None:
                        target_labeled_feats.append(None)
                        continue

                    x_t_l, y_t_l = batch
                    f_t_l, logits_t_l = model(x_t_l)
                    labeled_loss_terms.append(criterion(logits_t_l, y_t_l))
                    target_labeled_feats.append(f_t_l)

                if labeled_loss_terms:
                    loss_bce = loss_bce + sum(labeled_loss_terms) / len(labeled_loss_terms)

                # Target unlabeled features
                target_unlabeled_feats = []
                for x_t_u in target_unlabeled_batches:
                    f_t_u, _ = model(x_t_u)
                    target_unlabeled_feats.append(f_t_u)

                # Update source prototypes
                update_prototypes(prototypes, f_s.detach(), y_s.detach())

                # Class-alignment loss
                loss_ca = torch.tensor(0.0, device=device)
                if prototypes[0] is not None and prototypes[1] is not None:
                    ca_terms = [
                        class_alignment_entropy(
                            f_u,
                            prototypes[0],
                            prototypes[1],
                        )
                        for f_u in target_unlabeled_feats
                    ]
                    loss_ca = sum(ca_terms) / len(ca_terms)

                # Moment matching loss
                target_feats_for_mm = []
                for f_l, f_u in zip(target_labeled_feats, target_unlabeled_feats):
                    if f_l is None:
                        target_feats_for_mm.append(f_u)
                    else:
                        target_feats_for_mm.append(torch.cat([f_l, f_u], dim=0))

                loss_mm = moment_matching_loss(f_s, target_feats_for_mm)

                loss = (
                    loss_bce
                    + args.lambda_mm * loss_mm
                    + args.lambda_ca * loss_ca
                ) / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                scaler.step(optimizer)
                scaler.update()

            running["bce"] += loss_bce.item()
            running["mm"] += loss_mm.item()
            running["ca"] += loss_ca.item()
            running["total"] += loss.item() * args.grad_accum

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} [{step}/{args.steps_per_epoch}] "
                    f"total={running['total'] / step:.4f} "
                    f"bce={running['bce'] / step:.4f} "
                    f"mm={running['mm'] / step:.4f} "
                    f"ca={running['ca'] / step:.4f}"
                )

        epoch_time = time.time() - start_time
        stats = {
            k: v / args.steps_per_epoch
            for k, v in running.items()
        }

        ckpt = {
            "model": model.state_dict(),
            "prototypes": prototypes,
            "args": vars(args),
            "epoch": epoch,
        }

        torch.save(ckpt, out_dir / "checkpoint_last.pt")

        print(
            f"==> Epoch {epoch} done in {epoch_time:.1f}s | "
            f"total={stats['total']:.4f} "
            f"bce={stats['bce']:.4f} "
            f"mm={stats['mm']:.4f} "
            f"ca={stats['ca']:.4f}"
        )

    print(f"\nTraining complete. Saved checkpoint to: {out_dir / 'checkpoint_last.pt'}")


if __name__ == "__main__":
    main()