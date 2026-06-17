import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from models.vit_model import get_vit_b_16
from utils import plot_confusion_matrix

DATA_DIR = './dataset'
MODEL_PATH = './saved_models/best_vit.pth'
OUTPUT_DIR = './outputs/vit'

def evaluate():
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

    # Use the same split logic and seed
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    classes = full_dataset.classes

    # 2. Load Model
    model = get_vit_b_16(num_classes=len(classes)).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}. Train the model first.")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Evaluate
    all_preds = []
    all_labels = []

    print(f"Evaluating ViT-B/16 on {device}...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 5. Save Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, classes, "ViT-B_16", OUTPUT_DIR)
    print(f"Confusion matrix saved to {OUTPUT_DIR}/ViT-B_16_confusion_matrix.jpg")

if __name__ == '__main__':
    evaluate()
