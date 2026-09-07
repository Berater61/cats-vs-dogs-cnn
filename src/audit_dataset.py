"""Report image counts, duplicate content, and split overlap."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from split_dataset import CLASS_NAMES, file_sha256, find_images


def audit_raw_dataset(root: Path) -> dict[str, object]:
    report: dict[str, object] = {"type": "raw", "classes": {}}
    all_hashes: dict[str, set[str]] = defaultdict(set)

    for class_name in CLASS_NAMES:
        directory = root / class_name
        files = find_images(directory)
        hashes = [file_sha256(path) for path in files]
        for content_hash in hashes:
            all_hashes[content_hash].add(class_name)
        report["classes"][class_name] = {
            "images": len(files),
            "unique_images": len(set(hashes)),
            "duplicate_files": len(files) - len(set(hashes)),
        }

    report["conflicting_label_groups"] = sum(
        1 for labels in all_hashes.values() if len(labels) > 1
    )
    return report


def audit_split_dataset(root: Path) -> dict[str, object]:
    split_names = [name for name in ("train", "val", "test") if (root / name).is_dir()]
    report: dict[str, object] = {"type": "split", "splits": {}, "overlap": {}}
    split_hashes: dict[str, set[str]] = {}

    for split_name in split_names:
        hashes: set[str] = set()
        class_counts: dict[str, int] = {}
        for class_name in CLASS_NAMES:
            files = find_images(root / split_name / class_name)
            class_counts[class_name] = len(files)
            hashes.update(file_sha256(path) for path in files)
        split_hashes[split_name] = hashes
        report["splits"][split_name] = class_counts

    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            key = f"{left_name}_vs_{right_name}"
            report["overlap"][key] = len(split_hashes[left_name] & split_hashes[right_name])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--split-root",
        action="store_true",
        help="Treat the directory as a root containing train/val/test partitions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = (
        audit_split_dataset(args.dataset)
        if args.split_root
        else audit_raw_dataset(args.dataset)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
