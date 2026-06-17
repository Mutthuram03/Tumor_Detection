# INTERNSHIP PROJECT REPORT
**Institute Name**: IIT Hyderabad (Internship Assignment)
**Project Title**: Brain Tumor Detection Using Custom CNN and ResNet18

## 1. ABSTRACT
This project aims to automate Brain Tumor detection using Magnetic Resonance Imaging (MRI) scans by developing and comparing two neural network architectures: a Custom Convolutional Neural Network (CNN) trained from scratch and a pre-trained ResNet18 employing transfer learning. The results shed light on performance trade-offs, accuracy improvements via deep learning, and advanced feature applications like Test Time Augmentation (TTA) and Explainable AI (Grad-CAM).

## 2. DATASET & PREPROCESSING
- **Dataset**: Kaggle Brain Tumor MRI Dataset (multi-class).
- **Classes**: Four categories (`glioma`, `meningioma`, `notumor`, `pituitary`).
- **Preprocessing steps applied**:
  - Image resizing to *224x224*.
  - Normalization using ImageNet statistics `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
  - Splitting ratio: 70% Train, 15% Validation, 15% Test.
- **Augmentation (to counter over-fitting)**: Random Horizontal Flip, Random Rotation (15), Random Affine transformations, and Color Jittering.

## 3. METHODOLOGY
### 3.1 Custom CNN
- **Architecture**: A sequentially built configuration utilizing 4 Convolutional blocks. Each block pairs Conv2d, Batch-Normalization, ReLU Activation, and MaxPooling layers, concluding with dual Fully Connected layers connected via a 50% dropout layout.
- **Training Setup**: Trained exclusively on the MRI dataset parameters.

### 3.3 Vision Transformer (ViT-B/16)
- **Architecture**: Employed a Vision Transformer (ViT-B/16) pretrained on ImageNet-1k. The input image is divided into 14x14 patches (patch size 16x16), flattened, and projected into a 768-dimensional embedding space.
- **Adaptation**: Modified the classification head (the single linear layer in `model.heads.head`) to output 4 classes.
- **Optimization Strategy**: Utilized the `AdamW` optimizer with weight decay (`0.01`) and Label Smoothing (`0.1`) to encourage regularization and prevent overfitting on the relatively small dataset.

## 4. TRAINING PIPELINE
Both models utilized:
- **Optimizer**: Adam (Custom CNN, ResNet18) and AdamW (ViT-B/16).
- **Categorization Loss**: Cross Entropy Loss.
- **Learning Rate Scheduler**: ReduceLROnPlateau.
- **Regulation strategy**: Configured *Early Stopping* mechanism monitoring the validation loss space to prevent overfitting.

## 5. EXPERIMENTAL RESULTS 
The models were evaluated on the test partition (15% of the total dataset, containing 1080 images). Below are the final quantitative metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet18** | **98.06%** | **98.06%** | **98.06%** | **98.06%** |
| **ViT-B/16** | **97.87%** | **97.97%** | **97.87%** | **97.89%** |

### 5.1 t-SNE Clustering Analysis
The high-dimensional feature embeddings extracted from the final pre-pooling layer of ResNet18 (512 dimensions) and the CLS token of ViT-B/16 (768 dimensions) were projected into 2D space using t-SNE:
- **ResNet18**: Shows extremely tight, separate class clusters for all four categories (`glioma`, `meningioma`, `notumor`, `pituitary`) with minimal outliers. This demonstrates that the CNN's spatial hierarchy has successfully captured highly discriminative texture and shape features.
- **ViT-B/16**: Demonstrates equally distinct clustering boundaries. The self-attention mechanisms of the transformer successfully clustered globally consistent patterns across the MRI scans, showing that global dependencies are highly effective for tumor classification.

## 6. EXPLAINABLE AI (XAI)
To un-black-box the Deep Learning conclusions, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was implemented for the ResNet18 and Custom CNN architectures. 
- The generated visual heatmaps highlight precisely the spatial regions inside the specific MRI that drove the prediction.
- Color mapping utilizes a custom-corrected Jet scale (converting BGR to RGB) ensuring that red/hot regions accurately denote the highest-risk focal points for tumors.

## 7. KEY APPLICATION FEATURES
Beyond the core deep learning architecture, a complete, clinical-grade medical application was developed. Standout features include:
- **Dynamic PDF Diagnostic Reports**: Automated, on-the-fly generation of downloadable PDF documents that embed the patient's MRI, the AI-generated Grad-CAM heatmap, and the final diagnosis with confidence metrics.
- **Advanced "Glassmorphism" Medical Dashboard**: A HTML/CSS/JS frontend UI featuring custom typography, responsive design, and side-by-side comparative views of original scans and AI heatmaps.
- **Interactive Drag-and-Drop Upload**: A robust frontend zone allowing users to seamlessly drag and drop MRI `.jpg` or `.png` files.
- **Granular Probability Distribution**: Visual confidence progress bars for all 4 distinct tumor classes.
- **Test-Time Augmentation (TTA) Toggle**: An advanced feature that, when toggled, passes 8 augmented variations (rotated, flipped) of the single uploaded MRI through the model and averages the results to ensure robust, outlier-resistant predictions.
- **Automated Metric Extraction**: A dedicated evaluation pipeline that generates accurate Confusion Matrices (in `.jpg`), Loss/Accuracy curves, and comparative reports.

## 8. SUMMARY
This pipeline establishes a standard clinical prediction layout, successfully unifying cutting-edge feature extractors (ResNet and ViT) alongside completely auditable custom architectural baselines, all wrapped within a highly professional, feature-rich web dashboard.
