#!/usr/bin/env python3
"""
Dataset generator for Single-Source Multi-Target Domain Adaptation (SSMT-DA)
for deepfake detection.

Expected raw directory:

RAW_DIR/
  DF/          # fake images
  FS/          # fake images
  F2F/         # fake images
  NT/          # fake images
  Pristine/    # real images

Generated dataset structure:

OUT_DIR/
  <source>/
    train/fake/
    train/real/
    test/fake/
    test/real/

  <target>/
    train/labeled/fake/
    train/labeled/real/
    train/unlabeled/        # mixed fake + real
    test/fake/
    test/real/

  <combined_name>/
    test/fake/
    test/real/

Default paper setting per target:
  labeled   : 500 fake + 500 real  = 1000 total
  unlabeled : 4500 fake + 4500 real = 9000 total
  test      : 2500 fake + 2500 real
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    files.sort()

    if not files:
        raise RuntimeError(f"No image files found in: {folder}")

    return files


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sample_without_overlap(
    pool: List[Path],
    n: int,
    rng: random.Random,
) -> Tuple[List[Path], List[Path]]:
    if n < 0:
        raise ValueError(f"Sample size cannot be negative: {n}")

    if n > len(pool):
        raise ValueError(f"Need {n} files but only {len(pool)} available.")

    idx = list(range(len(pool)))
    rng.shuffle(idx)

    taken = [pool[i] for i in idx[:n]]
    remaining = [pool[i] for i in idx[n:]]

    return taken, remaining


def copy_block(
    tag: str,
    files: List[Path],
    out_dir: Path,
    prefix: str = "",
    log_every: int = 500,
) -> None:
    safe_mkdir(out_dir)

    total = len(files)
    if total == 0:
        print(f"[{tag}] 0/0")
        return

    for i, src in enumerate(files, 1):
        dst = out_dir / f"{prefix}{src.name}"
        shutil.copy2(src, dst)

        if i % log_every == 0 or i == total:
            print(f"[{tag}] {i}/{total}")


def validate_even(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    if value % 2 != 0:
        raise ValueError(f"{name} must be even to keep fake/real balanced.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SSMT-DA deepfake dataset splits."
    )

    parser.add_argument("--raw_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--source",
        type=str,
        default="DF",
        help="Source domain folder name inside raw_dir.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        required=True,
        help="Target domain folder names inside raw_dir. Example: FS F2F or FS F2F NT",
    )
    parser.add_argument(
        "--combined_name",
        type=str,
        default="TARGET_COMBINED",
        help="Folder name for combined target test set.",
    )

    parser.add_argument(
        "--t_labeled_total",
        type=int,
        default=1000,
        help=(
            "Total labeled target samples per target domain. "
            "Default 1000 = 500 fake + 500 real."
        ),
    )
    parser.add_argument(
        "--t_unlabeled_total",
        type=int,
        default=9000,
        help=(
            "Total unlabeled target samples per target domain. "
            "Default 9000 = 4500 fake + 4500 real."
        ),
    )

    args = parser.parse_args()

    validate_even("--t_labeled_total", args.t_labeled_total)
    validate_even("--t_unlabeled_total", args.t_unlabeled_total)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    source = args.source
    targets = list(args.targets)
    combined_name = args.combined_name

    if len(targets) < 1:
        raise ValueError("At least one target domain must be provided.")

    # Source split sizes
    SRC_TRAIN_FAKE = 20000
    SRC_TRAIN_REAL = 20000
    SRC_TEST_FAKE = 5000
    SRC_TEST_REAL = 5000

    # Target split sizes
    T_LAB_FAKE = args.t_labeled_total // 2
    T_LAB_REAL = args.t_labeled_total // 2
    T_UNLAB_FAKE = args.t_unlabeled_total // 2
    T_UNLAB_REAL = args.t_unlabeled_total // 2
    T_TEST_FAKE = 2500
    T_TEST_REAL = 2500

    print("=== SSMT Dataset Generation ===")
    print(f"raw_dir              : {raw_dir}")
    print(f"out_dir              : {out_dir}")
    print(f"seed                 : {args.seed}")
    print(f"source               : {source}")
    print(f"targets              : {', '.join(targets)}")
    print(f"combined_name        : {combined_name}")
    print(f"target labeled total : {args.t_labeled_total} "
          f"({T_LAB_FAKE} fake + {T_LAB_REAL} real)")
    print(f"target unlabeled total: {args.t_unlabeled_total} "
          f"({T_UNLAB_FAKE} fake + {T_UNLAB_REAL} real)")
    print()

    # Load image pools
    source_fake = list_images(raw_dir / source)
    pristine_real = list_images(raw_dir / "Pristine")
    target_fake_pools: Dict[str, List[Path]] = {
        t: list_images(raw_dir / t) for t in targets
    }

    print("Pool sizes:")
    print(f"  {source} fake: {len(source_fake)}")
    for t in targets:
        print(f"  {t} fake: {len(target_fake_pools[t])}")
    print(f"  Pristine real: {len(pristine_real)}")
    print()

    # Source splits
    src_train_fake, source_fake_rem = sample_without_overlap(
        source_fake, SRC_TRAIN_FAKE, rng
    )
    src_test_fake, source_fake_rem = sample_without_overlap(
        source_fake_rem, SRC_TEST_FAKE, rng
    )

    src_train_real, pristine_rem = sample_without_overlap(
        pristine_real, SRC_TRAIN_REAL, rng
    )
    src_test_real, pristine_rem = sample_without_overlap(
        pristine_rem, SRC_TEST_REAL, rng
    )

    # Target splits
    splits = {}

    for t in targets:
        fake_pool = target_fake_pools[t]

        lab_fake, fake_rem = sample_without_overlap(fake_pool, T_LAB_FAKE, rng)
        unlab_fake, fake_rem = sample_without_overlap(fake_rem, T_UNLAB_FAKE, rng)
        test_fake, fake_rem = sample_without_overlap(fake_rem, T_TEST_FAKE, rng)

        lab_real, pristine_rem = sample_without_overlap(pristine_rem, T_LAB_REAL, rng)
        unlab_real, pristine_rem = sample_without_overlap(pristine_rem, T_UNLAB_REAL, rng)
        test_real, pristine_rem = sample_without_overlap(pristine_rem, T_TEST_REAL, rng)

        splits[t] = {
            "lab_fake": lab_fake,
            "unlab_fake": unlab_fake,
            "test_fake": test_fake,
            "lab_real": lab_real,
            "unlab_real": unlab_real,
            "test_real": test_real,
            "fake_unused": fake_rem,
        }

    # Write source
    copy_block(f"{source} train fake", src_train_fake, out_dir / source / "train" / "fake")
    copy_block(f"{source} train real", src_train_real, out_dir / source / "train" / "real")
    copy_block(f"{source} test fake", src_test_fake, out_dir / source / "test" / "fake")
    copy_block(f"{source} test real", src_test_real, out_dir / source / "test" / "real")

    # Write targets
    for t in targets:
        prefix = f"{t}_"

        copy_block(
            f"{t} train labeled fake",
            splits[t]["lab_fake"],
            out_dir / t / "train" / "labeled" / "fake",
            prefix=prefix,
        )
        copy_block(
            f"{t} train labeled real",
            splits[t]["lab_real"],
            out_dir / t / "train" / "labeled" / "real",
            prefix="P_",
        )

        copy_block(
            f"{t} train unlabeled fake",
            splits[t]["unlab_fake"],
            out_dir / t / "train" / "unlabeled",
            prefix=prefix,
        )
        copy_block(
            f"{t} train unlabeled real",
            splits[t]["unlab_real"],
            out_dir / t / "train" / "unlabeled",
            prefix="P_",
        )

        copy_block(
            f"{t} test fake",
            splits[t]["test_fake"],
            out_dir / t / "test" / "fake",
            prefix=prefix,
        )
        copy_block(
            f"{t} test real",
            splits[t]["test_real"],
            out_dir / t / "test" / "real",
            prefix="P_",
        )

    # Combined target test set
    combined_fake_dir = out_dir / combined_name / "test" / "fake"
    combined_real_dir = out_dir / combined_name / "test" / "real"
    safe_mkdir(combined_fake_dir)
    safe_mkdir(combined_real_dir)

    for t in targets:
        for p in (out_dir / t / "test" / "fake").iterdir():
            if p.is_file():
                shutil.copy2(p, combined_fake_dir / p.name)

        for p in (out_dir / t / "test" / "real").iterdir():
            if p.is_file():
                shutil.copy2(p, combined_real_dir / p.name)

    print()
    print("✅ Dataset generation complete.")
    print(f"Output dataset: {out_dir}")
    print()
    print("Counts summary:")
    print(
        f"  {source} train: {SRC_TRAIN_FAKE} fake, {SRC_TRAIN_REAL} real | "
        f"test: {SRC_TEST_FAKE} fake, {SRC_TEST_REAL} real"
    )

    for t in targets:
        print(
            f"  {t} labeled: {T_LAB_FAKE} fake, {T_LAB_REAL} real | "
            f"unlabeled: {T_UNLAB_FAKE} fake + {T_UNLAB_REAL} real | "
            f"test: {T_TEST_FAKE} fake, {T_TEST_REAL} real"
        )

    print(
        f"  {combined_name} test: "
        f"{len(targets) * T_TEST_FAKE} fake, "
        f"{len(targets) * T_TEST_REAL} real"
    )

    print()
    print("Remaining pools:")
    print(f"  Pristine real unused: {len(pristine_rem)}")
    print(f"  {source} fake unused: {len(source_fake_rem)}")
    for t in targets:
        print(f"  {t} fake unused: {len(splits[t]['fake_unused'])}")


if __name__ == "__main__":
    main()