import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.resnet18_model import get_resnet18
from models.vit_model import get_vit_b_16

DATA_DIR = './dataset'
RESNET_MODEL_PATH = './saved_models/best_resnet18.pth'
VIT_MODEL_PATH = './saved_models/best_vit.pth'
OUTPUT_DIR = './outputs/embeddings'

def extract_features():
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

    # 2. Setup ResNet18
    resnet = get_resnet18(num_classes=num_classes).to(device)
    if os.path.exists(RESNET_MODEL_PATH):
        resnet.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
    else:
        print(f"Warning: {RESNET_MODEL_PATH} not found. Using untrained ResNet18.")
    
    # Remove final classification layer to get feature embeddings
    resnet.fc = nn.Identity()
    resnet.eval()

    # 3. Setup ViT-B/16
    vit = get_vit_b_16(num_classes=num_classes).to(device)
    if os.path.exists(VIT_MODEL_PATH):
        vit.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
    else:
        print(f"Warning: {VIT_MODEL_PATH} not found. Using untrained ViT-B/16.")
    
    # Remove final classification head to get CLS token representation
    vit.heads = nn.Identity()
    vit.eval()

    resnet_embeddings = []
    vit_embeddings = []
    all_labels = []

    print(f"Extracting embeddings on {device}...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Extract features
            res_feat = resnet(images)
            vit_feat = vit(images)
            
            resnet_embeddings.append(res_feat.cpu().numpy())
            vit_embeddings.append(vit_feat.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate lists into arrays
    resnet_embeddings = np.concatenate(resnet_embeddings, axis=0)
    vit_embeddings = np.concatenate(vit_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"ResNet18 embeddings shape: {resnet_embeddings.shape}")
    print(f"ViT-B/16 embeddings shape: {vit_embeddings.shape}")
    print(f"Labels shape: {all_labels.shape}")

    # 4. Save to Disk
    np.save(os.path.join(OUTPUT_DIR, 'resnet_embeddings.npy'), resnet_embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'vit_embeddings.npy'), vit_embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), all_labels)
    
    print(f"Embeddings saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    extract_features()
