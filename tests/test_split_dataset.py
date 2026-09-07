from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from split_dataset import file_sha256, prepare_dataset


class SplitDatasetTest(unittest.TestCase):
    def create_source(self, root: Path) -> Path:
        source = root / "raw"
        for class_name in ("cats", "dogs"):
            class_directory = source / class_name
            class_directory.mkdir(parents=True)
            for index in range(4):
                (class_directory / f"{class_name}.{index}.jpg").write_bytes(
                    f"{class_name}-{index}".encode()
                )
            (class_directory / f"{class_name}.duplicate.jpg").write_bytes(
                f"{class_name}-0".encode()
            )
        return source

    def split_hashes(self, output: Path, split: str) -> set[str]:
        return {
            file_sha256(path)
            for class_name in ("cats", "dogs")
            for path in (output / split / class_name).glob("*.jpg")
        }

    def test_duplicate_content_never_crosses_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_source(root)
            output = root / "processed"

            summary = prepare_dataset(source, output, train_ratio=0.6, seed=7)

            self.assertEqual(summary["cats"]["duplicate_files"], 1)
            self.assertEqual(summary["dogs"]["duplicate_files"], 1)
            self.assertFalse(
                self.split_hashes(output, "train")
                & self.split_hashes(output, "val")
            )

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_source(root)
            output = root / "processed"
            output.mkdir()
            (output / "keep.txt").write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_dataset(source, output)

    def test_invalid_ratio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_source(root)

            with self.assertRaises(ValueError):
                prepare_dataset(source, root / "processed", train_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
