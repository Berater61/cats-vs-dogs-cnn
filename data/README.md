# Local dataset directory

Dataset images are intentionally not stored in this repository.

Place legally obtained source images in this structure:

```text
data/raw/
  cats/
  dogs/
```

Then create the deterministic training and validation split:

```bash
python src/split_dataset.py --source data/raw --output data/processed
```

The generated `data/processed/` directory is ignored by Git. The splitter groups byte-identical images before assigning them to a partition, preventing duplicate copies from appearing in both training and validation data.
