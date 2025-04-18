# 🧠 Medical Image Segmentation Benchmarking Pipeline

This repository contains the full Phase 1 pipeline for benchmarking state-of-the-art medical image segmentation models using DICOM/NIfTI data. The goal is to identify which model performs best for real-world hospital workflows, based on various medical imaging modalities (MRI, CT, Ultrasound, X-ray). The end-goal is to create a self-learning artificial intelligence system that can annotate patient scans, whilst creating a massive dataset of anonymized and annotated medical images to solve every medical problem known to physicians. 

![Med-AI Banner](https://i.imgur.com/2sNQKP5.jpg)

---

## 📌 Project Phases

### ✅ Phase 1: Research Pipeline
1. **Data Source**: 10,000+ anonymized DICOM/NIfTI scans (collected from open source medical database with annotated scans)
2. **ETL Pipeline**: Apache Airflow + SimpleITK preprocessing
3. **Training**: 12+ segmentation models with PyTorch + MONAI
4. **Evaluation**: Dice, IoU, Sensitivity, Specificity
5. **Visualization**: TensorBoard dashboards
6. **Interpretation**: Human-in-the-loop paper insights

### ⏳ Phase 2 (Planned, not implemented here):
A hospital-like workflow API for doctors to label & improve model predictions. The model will self-learn over time from human feedback.

---

## 🛠️ Tech Stack

| Layer       | Tools/Frameworks                                      |
|------------|--------------------------------------------------------|
| **ETL**     | Apache Airflow, Python                                |
| **Preprocessing** | SimpleITK, pydicom, nibabel                     |
| **Training** | PyTorch, MONAI, CUDA                                 |
| **Parallelization** | Voltage Park Cloud GPU, torch.multiprocessing |
| **Tracking** | TensorBoard                                          |
| **Data Format** | DICOM (.dcm), NIfTI (.nii), JSON metadata         |

---

## 📋 Development Guidelines

### 🗂️ 1. Repository Structure

```
med-segmentation-pipeline/
├── airflow/              # DAGs for ETL preprocessing
├── data/
│   ├── raw/              # Original DICOM & NIfTI files
│   ├── processed/        # Preprocessed, normalized images
│   └── metadata/         # CSV or JSON metadata parsed from DICOM
├── training/
│   ├── configs/          # YAML configs for each model
│   ├── scripts/          # Train + Evaluate logic
│   └── parallel_launcher/ # Parallel GPU orchestrator
├── evaluation/
│   └── metrics/          # Dice, IoU, Sensitivity, Specificity
├── tensorboard_logs/     # Training visualizations
├── notebooks/            # EDA, sample test runs
├── requirements.txt
└── README.md
```

### 🚀 2. Preprocessing Pipeline

Using Apache Airflow + SimpleITK:

- **Airflow DAG**
  - Ingests DICOM/NIfTI files
  - Extracts metadata
  - Applies transformations (resize, normalize, crop)
  - Stores processed data in organized structure

- **Operators & Parallelism**
  - Custom PythonOperators for processing tasks
  - `multiprocessing.Pool` for file-level parallelism

### 🧠 3. Model Architecture

**Models to Implement (12+):**

| Type | Architectures |
|------|--------------|
| CNN-based | U-Net, U-Net++, DeepLabv3, ResUNet |
| Transformer | Swin-UNet, TransUNet |
| Hybrid | SAM (Segment Anything Model), DenseNet |
| Object Seg | Mask R-CNN |
| Lightweight | Fast-SCNN, BiSeNet |

**Framework:**
- PyTorch Lightning or vanilla PyTorch
- MONAI transforms and datasets
- Custom `MedicalSegDataset` class

**Config System:**
- YAML configs for each model
- CLI parameter override support

### ⚡ 4. Parallel Training with Voltage Park GPUs

```python
# train_parallel.py
import os
import torch.multiprocessing as mp

models = [
    "unet", "deeplabv3", "densenet", "resunet", "maskrcnn",
    "pspnet", "fcn", "segformer", "attentionunet", "unetpp",
    "sam", "transunet"
]

def train_model(model_name, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.system(f"python training/scripts/train_{model_name}.py --gpu {gpu_id}")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    processes = []
    for idx, model in enumerate(models):
        p = mp.Process(target=train_model, args=(model, idx % torch.cuda.device_count()))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
```

**Environment Setup:**
- One GPU per model on Voltage Park Cloud
- Docker with CUDA, MONAI, PyTorch preinstalled
- Shared volume mounts for data access

**Monitoring:**
- Separate logging per model
- Consolidated TensorBoard visualization

### 📊 5. Evaluation Metrics

After each training job:
- Save best model checkpoint
- Run evaluation on validation and test sets
- Calculate and store metrics:
  - Dice coefficient
  - IoU (Intersection over Union)
  - Sensitivity
  - Specificity
  - Execution time

### 📁 6. Research Notebooks

Exploratory notebooks for:
- Dataset analysis
- Metadata visualization
- Image preprocessing verification
- Model output inspection

### 🧪 7. Reporting

**Research-grade reporting:**
- CSV tables per model
- TensorBoard visualizations
- Per-class metrics
- Confusion matrices
- Box plots of Dice scores across models

### 🔧 8. Configuration Examples

**requirements.txt:**
```
monai
torch
torchvision
pydicom
nibabel
simpleitk
numpy
opencv-python
matplotlib
pandas
scikit-learn
tensorboard
apache-airflow
```

**Sample YAML Training Config:**
```yaml
model: unet
epochs: 50
batch_size: 8
lr: 0.0001
optimizer: adam
loss: dice
input_size: [256, 256]
augmentation:
  horizontal_flip: true
  random_crop: true
```

### ✅ 9. Team Workflow

- Each team member owns 1-2 models
- PRs submitted against `dev` branch
- GitHub issues/discussions for progress tracking
- Consolidated results for final paper

### 📦 10. Output Structure

```
/results/
├── unet/
│   ├── model.pt
│   ├── metrics.csv
│   ├── tensorboard/
├── sam/
│   ├── model.pt
│   └── ...
```

---

## ⚡ Running the Full Pipeline

1. Add raw DICOM/NIfTI scans to `data/raw/`
2. Launch Airflow ETL pipeline:
   ```bash
   airflow dags trigger preprocessing_pipeline
   ```
3. Preprocessed data lands in `data/processed/`
4. Launch all models in parallel:
   ```bash
   python training/parallel_launcher/train_parallel.py
   ```
5. Generate evaluation metrics:
   ```bash
   python evaluation/calculate_metrics.py
   ```
6. View results in TensorBoard:
   ```bash
   tensorboard --logdir tensorboard_logs/
   ```

---

## 📬 Contact

Got questions or want to collaborate?

Email: jatoujoseph@gmail.com
GitHub: @sudoBaymax

---

## ⚠️ Disclaimer

This project is for research use only. It is not intended for real-world clinical decision-making.
