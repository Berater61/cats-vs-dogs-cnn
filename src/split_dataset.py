"""Create a deterministic cats-vs-dogs training/validation split."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
CLASS_NAMES = ("cats", "dogs")


def find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_by_content(files: Iterable[Path]) -> list[list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[file_sha256(path)].append(path)
    return list(groups.values())


def split_content_groups(
    groups: list[list[Path]], train_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(groups) < 2:
        raise ValueError("At least two unique images are required per class")

    shuffled_groups = list(groups)
    random.Random(seed).shuffle(shuffled_groups)
    target_train_size = int(sum(len(group) for group in groups) * train_ratio)

    train_files: list[Path] = []
    validation_files: list[Path] = []
    for group in shuffled_groups:
        destination = train_files if len(train_files) < target_train_size else validation_files
        destination.extend(group)

    if not train_files or not validation_files:
        raise ValueError("The requested ratio produced an empty partition")
    return train_files, validation_files


def copy_files(files: Iterable[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in files:
        shutil.copy2(source_path, destination / source_path.name)


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()

    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output directory must not be inside the source dataset")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    class_files: dict[str, list[Path]] = {}
    hashes_to_classes: dict[str, set[str]] = defaultdict(set)
    for class_name in CLASS_NAMES:
        class_directory = source_root / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_directory}")
        files = find_images(class_directory)
        if not files:
            raise ValueError(f"No supported images found in {class_directory}")
        class_files[class_name] = files
        for path in files:
            hashes_to_classes[file_sha256(path)].add(class_name)

    conflicting_hashes = [
        content_hash
        for content_hash, labels in hashes_to_classes.items()
        if len(labels) > 1
    ]
    if conflicting_hashes:
        raise ValueError(
            f"Found {len(conflicting_hashes)} identical image group(s) with conflicting labels"
        )

    summary: dict[str, dict[str, int]] = {}
    for class_offset, class_name in enumerate(CLASS_NAMES):
        files = class_files[class_name]
        groups = group_by_content(files)
        train_files, validation_files = split_content_groups(
            groups, train_ratio, seed + class_offset
        )

        copy_files(train_files, output_root / "train" / class_name)
        copy_files(validation_files, output_root / "val" / class_name)
        summary[class_name] = {
            "source_images": len(files),
            "unique_images": len(groups),
            "duplicate_files": len(files) - len(groups),
            "train_images": len(train_files),
            "validation_images": len(validation_files),
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        source_root=args.source,
        output_root=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    for class_name, values in summary.items():
        print(
            f"{class_name}: {values['train_images']} training, "
            f"{values['validation_images']} validation, "
            f"{values['duplicate_files']} duplicate file(s) grouped"
        )


if __name__ == "__main__":
    main()
