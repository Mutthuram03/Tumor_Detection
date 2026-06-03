import os
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from models.custom_cnn import CustomCNN
from models.resnet18_model import get_resnet18
from utils import plot_confusion_matrix

def evaluate_model(model, dataloader, device, num_classes=4):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            if num_classes == 2:
                y_scores_batch = probs[:, 1].cpu().numpy()
            else:
                y_scores_batch = probs.cpu().numpy()
                
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(y_scores_batch)
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, average='weighted' if num_classes > 2 else 'binary')
    rec = recall_score(y_true, y_pred, average='weighted' if num_classes > 2 else 'binary')
    f1 = f1_score(y_true, y_pred, average='weighted' if num_classes > 2 else 'binary')
    
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, y_scores)
        else:
            roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
    except ValueError:
        roc_auc = 0.0 # handles edge case where test set might have only 1 class

    return y_true, y_pred, acc, prec, rec, f1, roc_auc

def main():
    DATA_DIR = './dataset'
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
    classes = full_dataset.classes

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init models
    cnn_model = CustomCNN(num_classes=4).to(device)
    resnet_model = get_resnet18(num_classes=4).to(device)

    # Load Weights
    cnn_model.load_state_dict(torch.load('./saved_models/best_custom_cnn.pth', map_location=device))
    resnet_model.load_state_dict(torch.load('./saved_models/best_resnet18.pth', map_location=device))

    # Evaluate Custom CNN
    print("Evaluating Custom CNN...")
    y_true_cnn, y_pred_cnn, acc_cnn, prec_cnn, rec_cnn, f1_cnn, auc_cnn = evaluate_model(cnn_model, test_loader, device)
    plot_confusion_matrix(y_true_cnn, y_pred_cnn, classes, "Custom CNN", "./outputs/confusion_matrix")
    
    # Evaluate ResNet18
    print("Evaluating ResNet18...")
    y_true_res, y_pred_res, acc_res, prec_res, rec_res, f1_res, auc_res = evaluate_model(resnet_model, test_loader, device)
    plot_confusion_matrix(y_true_res, y_pred_res, classes, "ResNet18", "./outputs/confusion_matrix")

    os.makedirs('./outputs/metrics', exist_ok=True)
    
    # Save Classification Reports
    with open('./outputs/metrics/classification_reports.txt', 'w') as f:
        f.write("Custom CNN Classification Report:\n")
        f.write(classification_report(y_true_cnn, y_pred_cnn, target_names=classes))
        f.write("\n\nResNet18 Classification Report:\n")
        f.write(classification_report(y_true_res, y_pred_res, target_names=classes))

    # Parameter counting
    cnn_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    resnet_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)

    # Create Comparison Table
    results = {
        "Model": ["Custom CNN", "ResNet18"],
        "Accuracy": [f"{acc_cnn:.4f}", f"{acc_res:.4f}"],
        "Precision": [f"{prec_cnn:.4f}", f"{prec_res:.4f}"],
        "Recall": [f"{rec_cnn:.4f}", f"{rec_res:.4f}"],
        "F1 Score": [f"{f1_cnn:.4f}", f"{f1_res:.4f}"],
        "ROC-AUC": [f"{auc_cnn:.4f}", f"{auc_res:.4f}"],
        "Parameters": [cnn_params, resnet_params]
    }
    
    df = pd.DataFrame(results)
    df.to_csv('./outputs/metrics/model_comparison.csv', index=False)
    print("\nComparison Table:")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
