# Brain Tumor Detection Using Custom CNN and ResNet18

This repository contains a complete PyTorch pipeline for Brain Tumor MRI classification into four categories: `glioma`, `meningioma`, `notumor`, and `pituitary`. It implements a Custom CNN architecture built from scratch alongside a ResNet18 model utilizing Transfer Learning.

## Objective
Develop, train, and compare the performance of Custom CNN vs Pretrained ResNet18 on classifying Brain Tumor MRI images.

## Features
- **Data Augmentation**: Robust transformations (Flip, Rotation, Affine, Color Jitter).
- **Custom CNN**: 4 Conv Layers, Batch Normalization, Dropout.
- **Transfer Learning**: ResNet18 (ImageNet weights).
- **Training Pipeline**: Adam, CrossEntropy, LR Scheduler, Early Stopping.
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrices.
- **XAI (Explainable AI)**: Grad-CAM implementation for visualizing ResNet18 predictions.
- **Bonus Feature**: Test Time Augmentation (TTA) integrated in inference.

## Directory Structure
```
brain_tumor_detection/
├── dataset/                     # Place 'Tumor' & 'No_Tumor' subdirectories here
├── models/                      # Network Definitions
│   ├── custom_cnn.py
│   └── resnet18_model.py
├── outputs/                     # Generated visual items
│   ├── plots/
│   ├── confusion_matrix/
│   ├── metrics/
│   └── gradcam/
├── saved_models/                # Checkpoints (automatically generated)
├── train_custom_cnn.py          # Train CNN
├── train_resnet18.py            # Train ResNet18
├── evaluate.py                  # Generate classification reports and plots
├── gradcam.py                   # Run Explainable AI scripts
├── inference.py                 # Predicts using normal rules & TTA
├── utils.py                     # Helper functions (plotting, early stopping)
└── requirements.txt             # Python dependencies
```

## Setup & Execution

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Dataset
Download the Brain Tumor Dataset from Kaggle.
Extract and ensure the folder structure looks like this:
```
brain_tumor_detection/
└── dataset/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 3. Training
Run training for the Custom CNN:
```bash
python train_custom_cnn.py
```
Run training for ResNet18:
```bash
python train_resnet18.py
```
This saves the `.pth` files in `saved_models/` and charts in `outputs/plots/`.

### 4. Evaluation 
Generate Confusion matrix and evaluation comparisons (Acc, F1, ROC-AUC):
```bash
python evaluate.py
```

### 5. Explainable AI (Grad-CAM)
Modify the `sample_image` variable path inside `gradcam.py` to point to a valid MRI image, then run:
```bash
python gradcam.py
```

### 7. User Interface (Frontend Web App)
You can launch an interactive graphical web interface to upload MRIs and test the models visually.
```bash
streamlit run app.py
```
