import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.resnet18_model import get_resnet18
from models.vit_model import get_vit_b_16

DATA_DIR = './dataset'
RESNET_MODEL_PATH = './saved_models/best_resnet18.pth'
VIT_MODEL_PATH = './saved_models/best_vit.pth'
OUTPUT_DIR = './outputs/comparison'

def get_metrics(model, dataloader, device):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return acc, precision, recall, f1

def compare_models():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
    except FileNotFoundError:
        print(f"Error: Dataset not found in {DATA_DIR}.")
        return

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_classes = len(full_dataset.classes)

    # 2. Load Models and Evaluate
    print("Evaluating ResNet18...")
    resnet = get_resnet18(num_classes=num_classes).to(device)
    if os.path.exists(RESNET_MODEL_PATH):
        resnet.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
        res_acc, res_prec, res_rec, res_f1 = get_metrics(resnet, test_loader, device)
    else:
        print("ResNet18 weights not found. Using 0 for metrics.")
        res_acc, res_prec, res_rec, res_f1 = 0.0, 0.0, 0.0, 0.0

    print("Evaluating ViT-B/16...")
    vit = get_vit_b_16(num_classes=num_classes).to(device)
    if os.path.exists(VIT_MODEL_PATH):
        vit.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
        vit_acc, vit_prec, vit_rec, vit_f1 = get_metrics(vit, test_loader, device)
    else:
        print("ViT-B/16 weights not found. Using 0 for metrics.")
        vit_acc, vit_prec, vit_rec, vit_f1 = 0.0, 0.0, 0.0, 0.0

    # 3. Generate Markdown Report
    report_content = f"""# Model Comparison Report: ResNet18 vs ViT-B/16

## 1. Performance Comparison

| Metric | ResNet18 | ViT-B/16 |
|----------|----------|----------|
| **Accuracy** | {res_acc:.4f} | {vit_acc:.4f} |
| **Precision**| {res_prec:.4f} | {vit_prec:.4f} |
| **Recall**   | {res_rec:.4f} | {vit_rec:.4f} |
| **F1 Score** | {res_f1:.4f} | {vit_f1:.4f} |

## 2. Feature Separability Comparison

*(Note: Review the generated t-SNE plots in `outputs/tsne/` to fully analyze feature separability.)*

- **ResNet18**: Convolutional Neural Networks like ResNet18 often capture localized spatial hierarchies well. The t-SNE plot for ResNet18 should display distinct clustering based on textural and structural differences in the MRI scans.
- **ViT-B/16**: Vision Transformers capture global context through self-attention mechanisms. The t-SNE plot for ViT is expected to show potentially tighter intra-class clusters and wider inter-class margins if the global context provides a stronger discriminative signal for tumor classification.

## 3. t-SNE Clustering Analysis

- Are the four classes (Glioma, Meningioma, No Tumor, Pituitary) well-separated in the 2D projected space?
- Any overlapping clusters generally indicate morphological similarities between certain tumor types that both models struggle to distinguish. (For instance, differentiating some meningiomas from gliomas).

## 4. Conclusion: Which model learned more discriminative features?

Based on the quantitative metrics above:
- The model with the higher **F1 Score** generally maintains a better balance of precision and recall.
- Qualitatively, the model whose t-SNE visualization shows more clearly defined, separate clusters has learned stronger discriminative deep features.
"""

    report_path = os.path.join(OUTPUT_DIR, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"Comparison report generated and saved to {report_path}")

if __name__ == '__main__':
    compare_models()
