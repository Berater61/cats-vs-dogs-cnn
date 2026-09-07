# Cats vs. Dogs CNN

A learning-focused binary image-classification project built with TensorFlow and Keras. It trains a convolutional neural network from scratch to distinguish cats from dogs and includes reproducible dataset preparation, training, evaluation, and single-image prediction scripts.

> **Learning project:** This repository documents a practical machine-learning exercise. It is not a novel model, a production AI service, or a benchmark-quality study.

## What was implemented

The original project contained four Python scripts for inspecting folders, creating an 80/20 split, training a Sequential CNN for 10 epochs, and classifying one local image. A saved `.keras` model was also present. The portfolio version retains that model architecture and workflow while replacing machine-specific paths, making the split deterministic, adding duplicate protection, saving future training metrics, and adding lightweight automated tests.

Not present in the original implementation:

- Data augmentation
- Transfer learning
- Hyperparameter search
- Early stopping or model checkpoints
- A separate held-out test evaluation
- Confusion matrices, precision/recall, or misclassification analysis

## Technologies

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- `tf.data`
- `unittest`

## Dataset and preparation

The local source contained images named like the well-known Kaggle Dogs vs. Cats data, but no provenance or license file was included. The image dataset is therefore **not redistributed**. Kaggle lists the competition data as subject to its competition rules, and those rules restrict republishing without permission from Microsoft Research: [dataset page](https://www.kaggle.com/competitions/dogs-vs-cats/data), [competition rules](https://www.kaggle.com/competitions/dogs-vs-cats/rules).

Audit of the original local files:

| Partition | Cats | Dogs | Total |
| --- | ---: | ---: | ---: |
| Full source folder | 1,011 | 1,012 | 2,023 |
| Training folder | 808 | 810 | 1,618 |
| Validation folder | 203 | 202 | 405 |

The folder called `test_set` was the full pre-split source, not an independent test set. Hash-based inspection found 23 groups of byte-identical duplicate images; seven duplicate groups crossed the original training/validation boundary. Any validation metrics from that split could therefore be slightly optimistic.

The revised `split_dataset.py` uses a fixed seed and keeps identical image copies in the same partition. Place a legally obtained dataset under `data/raw/cats` and `data/raw/dogs`, then run:

```bash
python src/split_dataset.py --source data/raw --output data/processed
```

Use `--overwrite` only when an existing generated split should be replaced.

## Preprocessing

- Images are decoded and resized to **180 × 180 RGB** by `image_dataset_from_directory`.
- Batches contain **32 images** by default.
- Pixel values are normalized from `[0, 255]` to `[0, 1]` by a `Rescaling(1/255)` model layer.
- Training data is cached, shuffled with a fixed seed, and prefetched.
- Validation data is cached and prefetched without shuffling.
- No data augmentation is applied.

## CNN architecture

| Layer | Output shape | Parameters |
| --- | --- | ---: |
| Input / rescaling | 180 × 180 × 3 | 0 |
| Conv2D, 32 filters, 3 × 3, ReLU | 178 × 178 × 32 | 896 |
| MaxPooling2D | 89 × 89 × 32 | 0 |
| Conv2D, 64 filters, 3 × 3, ReLU | 87 × 87 × 64 | 18,496 |
| MaxPooling2D | 43 × 43 × 64 | 0 |
| Conv2D, 128 filters, 3 × 3, ReLU | 41 × 41 × 128 | 73,856 |
| MaxPooling2D | 20 × 20 × 128 | 0 |
| Flatten | 51,200 | 0 |
| Dense, 128 units, ReLU | 128 | 6,553,728 |
| Dense, 1 unit, sigmoid | 1 | 129 |

**Total trainable parameters: 6,647,105**

This architecture was verified against the configuration stored in the original Keras 3.14.1 model archive.

## Training

The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy as the training metric. The original script used 10 epochs. The cleaned training command is:

```bash
python src/train_model.py --data data/processed --epochs 10
```

Generated output is written to the ignored `artifacts/` directory:

- `cats_dogs_model.keras`
- `training_curves.png`
- `metrics.json`

## Evaluation and existing results

The original training used Keras validation during `model.fit`, and the saved model proves that training produced a model artifact. However, no console log, metrics file, TensorBoard run, saved Accuracy/Loss plot, or independent test evaluation remained in the project. Numeric Accuracy or Loss values cannot be reconstructed reliably and are intentionally not claimed here.

The original `.keras` file was approximately 76 MB and is treated as a generated artifact rather than committed source. Because its validation split contained duplicate leakage, it should not be presented as a trustworthy benchmark even if its original metrics become available.

## Single-image prediction

The original project contained two local cat images and a script that loaded `Bild1.png`, printed the raw sigmoid value, and classified values below `0.5` as cat and values at or above `0.5` as dog. The prediction output was not retained. The images are not redistributed because their reuse rights were not documented.

After training, classify a local image with:

```bash
python src/predict_image.py path/to/image.jpg
```

## Installation

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow installation details can vary by operating system and available acceleration hardware. CPU training is supported but may be slow for this model.

## Dataset audit

Inspect raw class counts and duplicates:

```bash
python src/audit_dataset.py data/raw
```

Check a generated split for byte-identical overlap:

```bash
python src/audit_dataset.py data/processed --split-root
```

## Tests

The repository tests the deterministic split and duplicate-isolation behavior without requiring TensorFlow:

```bash
python -m compileall src tests
python -m unittest discover -s tests -v
```

## What I learned

This project demonstrates the complete structure of a small supervised computer-vision experiment: organizing labeled image folders, building `tf.data` pipelines, designing a CNN, selecting a binary loss and output activation, monitoring validation behavior, saving a Keras model, and running inference on a local image. The repository audit also showed why deterministic splits, provenance, deduplication, and retained experiment metrics are essential for credible evaluation.

## Known limitations

- The original local split had seven groups of identical images across training and validation.
- There is no independent held-out test set or cross-validation.
- The CNN has about 6.65 million parameters, most of them in one dense layer, which makes overfitting likely on roughly 2,000 images.
- No augmentation, dropout, regularization, or callbacks are used.
- Only accuracy and binary cross-entropy are configured; class-specific errors are not measured.
- Training reproducibility still depends on TensorFlow, hardware, and library versions.

## Possible next steps

- Create deduplicated train, validation, and held-out test partitions.
- Add realistic data augmentation and compare it with the current baseline.
- Replace the large Flatten/Dense section with global average pooling.
- Compare the baseline with a small transfer-learning model.
- Add confusion matrices, precision, recall, and a misclassification gallery.
- Track experiment configuration and metrics for every training run.

## Status

**Completed learning experiment with reproducibility cleanup.** The repository documents the original implementation honestly; it does not claim production readiness or state-of-the-art results.
