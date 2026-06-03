import torch
from torchvision import transforms
from PIL import Image
import os
from models.resnet18_model import get_resnet18

def predict_single_image(image_path, model, device):
    """Normal inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    return probs

def predict_tta(image_path, model, device, num_augmentations=5):
    """Test Time Augmentation inference to handle variability in inputs."""
    transform_tta = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    model.eval()
    probs_list = []
    
    with torch.no_grad():
        for _ in range(num_augmentations):
            tensor = transform_tta(image).unsqueeze(0).to(device)
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            probs_list.append(probs)
            
    # Average the probabilities across TTA steps
    avg_probs = torch.mean(torch.stack(probs_list), dim=0)
    return avg_probs

def main():
    model_path = "./saved_models/best_resnet18.pth"
    sample_image = "dataset/glioma/sample.jpg"  # Update with real image
    
    # Class names in alphabetical order (how PyTorch ImageFolder loads them)
    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    if not os.path.exists(model_path) or not os.path.exists(sample_image):
        print(f"Model or target image not found. Please ensure {sample_image} exists.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet18(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f"--- Inference Comparison on {os.path.basename(sample_image)} ---")
    
    # 1. Normal Inference
    normal_probs = predict_single_image(sample_image, model, device)
    print(f"Normal Inference Probabilities: {normal_probs.cpu().numpy()[0]}")
    normal_pred_idx = torch.argmax(normal_probs, dim=1).item()
    print(f"Predicted Class (Normal): {CLASSES[normal_pred_idx]}")
    
    # 2. TTA Inference
    tta_probs = predict_tta(sample_image, model, device, num_augmentations=10)
    print(f"TTA Inference Probabilities (Avg): {tta_probs.cpu().numpy()[0]}")
    tta_pred_idx = torch.argmax(tta_probs, dim=1).item()
    print(f"Predicted Class (TTA): {CLASSES[tta_pred_idx]}")

if __name__ == '__main__':
    main()
