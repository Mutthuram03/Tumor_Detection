import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, random_split
from models.vit_model import get_vit_b_16
from utils import EarlyStopping, save_plots
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001  # Lowered for better fine-tuning
DATA_DIR = './dataset'
MODEL_SAVE_PATH = './saved_models/best_vit.pth'
OUTPUT_DIR = './outputs/vit'

def main():
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data Preprocessing & Augmentation
    # Using advanced AutoAugment for robust ViT training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset_full = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
        val_dataset_full = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
    except FileNotFoundError:
        print(f"Error: Dataset not found in {DATA_DIR}. Please check the folder structure.")
        return

    # Train (70%), Validation (15%), Test (15%) split matching existing logic
    train_size = int(0.7 * len(train_dataset_full))
    val_size = int(0.15 * len(train_dataset_full))
    test_size = len(train_dataset_full) - train_size - val_size

    # Using generator with fixed seed to maintain identical splits
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset_full), generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    test_dataset = torch.utils.data.Subset(val_dataset_full, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vit_b_16(num_classes=4).to(device)

    # Starting fresh to allow AdamW and CosineAnnealing to work perfectly from scratch
    # Using label smoothing to prevent overfitting and AdamW for better ViT optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=7, verbose=True, path=MODEL_SAVE_PATH)

    # 3. Training Loop
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # Colorful terminal progress styling (Sleek Minimalist - Magenta & Green)
    MAGENTA = "\033[95m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    train_bar_format = f"{MAGENTA}{{desc}}{RESET}: {{percentage:3.0f}}%|{MAGENTA}{{bar:25}}{RESET}| {GREEN}{{n_fmt}}/{{total_fmt}}{RESET} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    val_bar_format = f"{GREEN}{{desc}}{RESET}: {{percentage:3.0f}}%|{GREEN}{{bar:25}}{RESET}| {MAGENTA}{{n_fmt}}/{{total_fmt}}{RESET} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"

    print(f"Training ViT-B/16 on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", 
            bar_format=train_bar_format,
            ascii=" ━"
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Running Acc': f'{correct/total*100:.2f}%'})

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", 
                bar_format=val_bar_format,
                ascii=" ━"
            )
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Running Acc': f'{val_correct/val_total*100:.2f}%'})

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        scheduler.step()
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Save plots
    save_plots(train_losses, val_losses, train_accs, val_accs, "ViT-B_16", OUTPUT_DIR)
    print("Training finished.")

if __name__ == '__main__':
    main()
