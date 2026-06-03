import os
import io
import base64
from PIL import Image
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
import numpy as np

from models.resnet18_model import get_resnet18
from models.custom_cnn import CustomCNN
from gradcam import GradCAM

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

DATA_DIR = './dataset'

# Utility: get class names from dataset folders (sorted)
def get_class_names():
    if not os.path.exists(DATA_DIR):
        return []
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    return classes

# Load models once
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet = get_resnet18(num_classes=len(get_class_names())).to(device)
    cnn = CustomCNN(num_classes=len(get_class_names())).to(device)

    resnet_path = './saved_models/best_resnet18.pth'
    cnn_path = './saved_models/best_custom_cnn.pth'

    if os.path.exists(resnet_path):
        resnet.load_state_dict(torch.load(resnet_path, map_location=device))
    if os.path.exists(cnn_path):
        cnn.load_state_dict(torch.load(cnn_path, map_location=device))

    resnet.eval()
    cnn.eval()

    return resnet, cnn, device

RESNET_MODEL, CNN_MODEL, DEVICE = load_models()
CLASSES = get_class_names()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_tta = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def tensor_probs_to_dict(probs_tensor):
    probs = probs_tensor.cpu().numpy()[0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


def compute_gradcam_overlay(pil_img, model, model_choice):
    # Prepare input
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    # target layer depending on selected model
    if model_choice == 'resnet':
        target_layer = model.layer4[-1]
    else:
        target_layer = model.relu4
        
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor)

    orig_img_resized = np.array(pil_img.resize((224, 224)))
    heatmap = cv2_apply_colormap(cam)
    overlay = (heatmap.astype(float) / 255.0) + (orig_img_resized.astype(float) / 255.0)
    overlay = overlay / np.max(overlay)
    # convert to uint8
    overlay_img = Image.fromarray(np.uint8(255 * overlay))
    return overlay_img


def cv2_apply_colormap(cam):
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def to_base64(img_pil):
    buffered = io.BytesIO()
    img_pil.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html', classes=CLASSES)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    model_choice = request.form.get('model', 'resnet')
    tta = request.form.get('tta', 'false').lower() == 'true'

    pil_img = Image.open(file.stream).convert('RGB')

    model = RESNET_MODEL if model_choice == 'resnet' else CNN_MODEL

    # Inference
    with torch.no_grad():
        if tta:
            # average several augmented predictions
            probs_list = []
            for _ in range(8):
                t = transform_tta(pil_img).unsqueeze(0).to(DEVICE)
                out = model(t)
                probs = torch.softmax(out, dim=1)
                probs_list.append(probs)
            avg = torch.mean(torch.stack(probs_list), dim=0)
            probs = avg
        else:
            inp = transform(pil_img).unsqueeze(0).to(DEVICE)
            out = model(inp)
            probs = torch.softmax(out, dim=1)

        pred_idx = int(torch.argmax(probs, dim=1).item())
        predicted_class = CLASSES[pred_idx] if CLASSES else str(pred_idx)
        confidence = float(probs[0][pred_idx].item())

    probs_dict = tensor_probs_to_dict(probs)

    # Grad-CAM overlay for both models
    overlay_b64 = None
    try:
        overlay_img = compute_gradcam_overlay(pil_img, model, model_choice)
        overlay_b64 = to_base64(overlay_img)
    except Exception as e:
        import traceback
        traceback.print_exc()
        overlay_b64 = None

    return jsonify({'predicted': predicted_class, 'confidence': confidence, 'probs': probs_dict, 'overlay': overlay_b64})

@app.route('/download_report', methods=['POST'])
def download_report():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    import time
    
    # Save the file temporarily in a temp folder
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{int(time.time())}_{file.filename}")
    file.save(temp_path)

    try:
        from export_report import generate_report
        pdf_path = generate_report(temp_path, out_folder=temp_dir)
        from flask import send_file
        return send_file(pdf_path, as_attachment=True, download_name=f"NeuroScan_Report_{int(time.time())}.pdf")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run local dev server
    app.run(host='0.0.0.0', port=8501, debug=True)
