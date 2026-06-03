import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.resnet18_model import get_resnet18
import os

class GradCAM:
    """Simple Grad-CAM implementation for ResNet18."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook connections
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[:, class_idx]
        
        self.model.zero_grad()
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # Global average pooling gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0) # ReLU
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam
        return cam


    def smooth_gradcam(grad_cam, input_tensor, n_samples=20, stdev_spread=0.15, class_idx=None):
        """
        Generates a SmoothGrad-style averaged CAM by adding noise to the input and
        averaging the resulting CAMs. Returns a normalized CAM array.
        input_tensor: torch tensor (1,C,H,W) normalized as model expects.
        grad_cam: instance of GradCAM already initialized with model and layer.
        """
        device = input_tensor.device
        cams = []
        mean = 0.0
        # stdev relative to input range (assumes normalized inputs ~[-2,2])
        for i in range(n_samples):
            noise = torch.randn_like(input_tensor).to(device) * stdev_spread
            noisy = input_tensor + noise
            cam = grad_cam(noisy, class_idx=class_idx)
            cams.append(cam)
        avg_cam = np.mean(cams, axis=0)
        avg_cam = np.maximum(avg_cam, 0)
        avg_cam = avg_cam - np.min(avg_cam)
        avg_cam = avg_cam / np.max(avg_cam) if np.max(avg_cam) != 0 else avg_cam
        return avg_cam

# Alias to allow importing at module level (used in export_report.py)
smooth_gradcam = GradCAM.smooth_gradcam

def generate_gradcam(image_path, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = get_resnet18(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # For ResNet18, the last spatial layer before pooling is layer4
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Generate CAM
    cam = grad_cam(input_tensor)
    
    # Processing for visualization
    orig_img_resized = np.array(original_image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    orig_rgb = np.float32(orig_img_resized) / 255
    overlay = heatmap + orig_rgb
    overlay = overlay / np.max(overlay)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(orig_img_resized)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Heatmap')
    plt.imshow(heatmap)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(overlay)
    plt.axis('off')
    
    file_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f'gradcam_{file_name}')
    plt.savefig(save_path)
    print(f"Grad-CAM saved at: {save_path}")
    plt.close()

if __name__ == '__main__':
    # Usage example - point to a real image. Make sure this image exists.
    sample_image = "dataset/Tumor/sample.jpg"  # Update this path
    if os.path.exists(sample_image):
        generate_gradcam(sample_image, "./saved_models/best_resnet18.pth", "./outputs/gradcam")
    else:
        print(f"Sample image not found at {sample_image}. Put an image there to test Grad-CAM.")
