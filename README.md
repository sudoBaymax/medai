# MedAI

MedAI is a **Python medical-image segmentation research prototype**. The repository currently focuses on a small end-to-end experimentation flow:

- basic preprocessing with an Apache Airflow DAG,
- 2D segmentation training with PyTorch + MONAI,
- simple evaluation on a test split.

> **Research-only status (important):** this repository is not a production system, not clinically validated, and not intended for diagnosis or hospital deployment.

## Tech stack

- Python
- PyTorch, MONAI
- Apache Airflow
- NiBabel, SimpleITK, pydicom
- Albumentations
- TensorBoard
- scikit-learn

## Architecture and data flow (current)

1. Raw data is expected under `data/raw` (DICOM/NIfTI).
2. `airflow/dags/preprocessing_pipeline.py` defines DAG `medical_image_preprocessing`, which performs basic:
   - DICOM series to NIfTI conversion, and
   - NIfTI intensity normalization.
   Outputs are written under `data/processed`.
3. `training/scripts/dataset.py` loads paired `.nii.gz` files from `images/` and `masks/` folders for each split.
4. For 3D volumes, `MedicalSegDataset` extracts the **middle slice** and applies 2D augmentations/transforms.
5. `training/scripts/train_unet.py` trains a MONAI 2D U-Net, logs to TensorBoard, and saves the best checkpoint.
6. `evaluation/metrics/evaluate.py` loads the checkpoint, evaluates the test split, and writes `evaluation_metrics.csv`.

## Repository structure

```text
airflow/
  dags/
    preprocessing_pipeline.py      # Airflow DAG: medical_image_preprocessing

evaluation/
  metrics/
    evaluate.py                    # Dice/IoU/sensitivity/specificity evaluation script

training/
  configs/
    base_config.yaml               # Data/training/model/logging/metrics configuration
  scripts/
    dataset.py                     # MedicalSegDataset for NIfTI image/mask pairs
    train_unet.py                  # MONAI 2D U-Net training + validation loop
  parallel_launcher/
    train_parallel.py              # Experimental prototype launcher (incomplete)

requirements.txt                   # Minimum dependency versions (not pinned)
setup.py                           # Creates expected project directory structure
README.md                          # Project documentation
```

## Prerequisites

- Python 3.10+ recommended
- `pip`
- (Optional, for GPU training) CUDA-compatible PyTorch environment
- (Optional, for preprocessing DAG) a working Apache Airflow installation

## Setup (fresh clone)

```bash
git clone https://github.com/sudoBaymax/medai.git
cd medai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

python setup.py
```

`requirements.txt` currently uses unpinned minimum versions (e.g., `package>=x.y.z`).

## Expected data layout

Training/evaluation scripts expect NIfTI pairs (`.nii.gz`) in this structure:

```text
data/processed/
  train/
    images/
      *.nii.gz
    masks/
      *.nii.gz
  val/
    images/
      *.nii.gz
    masks/
      *.nii.gz
  test/
    images/
      *.nii.gz
    masks/
      *.nii.gz
```

`MedicalSegDataset` reads from `images/` and `masks/` and pairs files by sorted order.

## Run training

```bash
python training/scripts/train_unet.py
```

What this script currently includes:
- MONAI 2D U-Net
- `DiceCELoss`
- Adam optimizer
- cosine annealing LR scheduler
- TensorBoard logging
- early stopping
- best-checkpoint saving to `results/best_model.pth`

## Run evaluation

```bash
python evaluation/metrics/evaluate.py
```

This computes mean Dice, IoU, sensitivity, and specificity on the configured test split and saves:

- `results/evaluation_metrics.csv`

## TensorBoard

```bash
tensorboard --logdir tensorboard_logs
```

## Airflow preprocessing DAG

The DAG is defined in:

- `airflow/dags/preprocessing_pipeline.py`
- DAG ID: `medical_image_preprocessing`

It is designed for an existing Airflow environment and performs basic preprocessing from `data/raw` to `data/processed`.

> This repository does **not** include Airflow deployment/infrastructure configuration (no Docker/Compose stack, no production scheduler/executor setup, no CI workflow around DAG deployment).

## `training/configs/base_config.yaml` (high-level fields)

- `data`: split paths (`train_path`, `val_path`, `test_path`), `input_size`, `batch_size`, `num_workers`
- `training`: `epochs`, `learning_rate`, `weight_decay`, optimizer/scheduler names, `early_stopping_patience`
- `augmentation`: augmentation toggles (training script currently relies on defaults in `dataset.py`)
- `model`: `in_channels`, `num_classes`, `pretrained`
- `logging`: TensorBoard directory, checkpoint directory, log/save intervals
- `metrics`: names of evaluation metrics

## Current limitations and caveats

- **2D-only slice extraction:** 3D volumes are reduced to the middle slice in `MedicalSegDataset`.
- **Fragile image/mask pairing:** files are paired by independent sorted lists, without robust ID matching.
- **No dataset/checkpoints included:** users must provide data and generate weights.
- **Parallel launcher is incomplete:** `training/parallel_launcher/train_parallel.py` references training scripts that are not present (prototype/experimental only).
- **Metric edge cases:** sensitivity/specificity computation can fail or be unstable for class-absent edge cases.
- **Limited reproducibility artifacts:** no pinned lockfile, no CI workflows, and no automated test suite in the repository.

## What is not included

- No deployment service/API
- No Docker/Compose deployment config
- No clinical validation claims or benchmark guarantees
- No bundled datasets or pretrained weights

## Short, honest LinkedIn/project description

Built a Python research prototype for medical-image segmentation using PyTorch, MONAI, and Airflow. Implemented preprocessing (DICOM/NIfTI), dataset loading with augmentations, 2D U-Net training with TensorBoard/checkpointing, and test-set evaluation (Dice, IoU, sensitivity, specificity). The project is intended for experimentation and workflow development, not clinical deployment.
