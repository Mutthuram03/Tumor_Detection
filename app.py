import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Import our models and GradCAM
from models.custom_cnn import CustomCNN
from models.resnet18_model import get_resnet18
from gradcam import GradCAM

# Configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# Custom CSS for premium medical AI look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #070913 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.06) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: #f1f5f9 !important;
    }
    
    .stAppHeader {
        background-color: transparent !important;
    }
    
    h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }
    
    .title-logo {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 5px;
    }
    
    .title-logo i {
        font-size: 40px;
        background: linear-gradient(135deg, #0ea5e9, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Styled uploader card */
    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(14, 165, 233, 0.25) !important;
        border-radius: 12px !important;
        background-color: rgba(13, 19, 36, 0.5) !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #0ea5e9 !important;
        background-color: rgba(14, 165, 233, 0.04) !important;
        box-shadow: 0 6px 24px rgba(14, 165, 233, 0.1) !important;
    }
    
    /* Standardized Streamlit buttons */
    button {
        background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
        color: #0f172a !important;
        border: none !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.25) !important;
        transition: all 0.2s ease !important;
        padding: 10px 24px !important;
        min-height: 42px;
    }
    
    button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4) !important;
        filter: brightness(1.05) !important;
    }
    
    /* Alert and success boxes */
    div[data-testid="stAlert"] {
        background-color: rgba(13, 19, 36, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }
    
    /* Image frames */
    div[data-testid="stImage"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background-color: #020205 !important;
        padding: 5px !important;
    }
    
    /* Card panel styling */
    .card-panel {
        background: rgba(13, 19, 36, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 24px !important;
    }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="title-logo">
    <i class="fa-solid fa-circle-nodes"></i>
    <h1 style="margin:0; font-family:'Outfit',sans-serif; letter-spacing:0.5px;">Brain Tumor Detection</h1>
</div>
<p style="color:#94a3b8; font-size:14px; margin-bottom:20px;">Deep learning classification of brain MRI scans</p>
""", unsafe_allow_html=True)

CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CNN
    cnn_model = CustomCNN(num_classes=4).to(device)
    if os.path.exists("./saved_models/best_custom_cnn.pth"):
        cnn_model.load_state_dict(torch.load("./saved_models/best_custom_cnn.pth", map_location=device))
    cnn_model.eval()

    # Load ResNet
    resnet_model = get_resnet18(num_classes=4).to(device)
    if os.path.exists("./saved_models/best_resnet18.pth"):
        resnet_model.load_state_dict(torch.load("./saved_models/best_resnet18.pth", map_location=device))
    resnet_model.eval()
    
    return cnn_model, resnet_model, device

cnn_model, resnet_model, device = load_models()

def predict(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    return probabilities

def get_gradcam_overlay(image, model, device):
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    cam = grad_cam(input_tensor)
    
    orig_img_resized = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    orig_rgb = np.float32(orig_img_resized) / 255
    overlay = heatmap + orig_rgb
    overlay = overlay / np.max(overlay)
    
    return overlay

# File Uploader
uploaded_file = st.file_uploader("Upload Brain MRI Image (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.subheader("Uploaded MRI Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.subheader("Configuration")
        model_choice = st.radio("Select Model:", ("ResNet18 (Transfer Learning)", "Custom CNN"))
        
        selected_model = resnet_model if model_choice == "ResNet18 (Transfer Learning)" else cnn_model
        
        if st.button("Run Prediction"):
            with st.spinner("Running prediction..."):
                probs = predict(image, selected_model, device)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item() * 100
                predicted_class = CLASSES[pred_idx]
                
                is_healthy = predicted_class.lower() == 'no tumor'
                if is_healthy:
                    st.success(f"**Prediction:** NO TUMOR")
                else:
                    st.error(f"**Prediction:** {predicted_class.upper()}")
                    
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                # Bar chart of probabilities
                st.write("Class Probabilities:")
                prob_dict = {CLASSES[i]: probs[i].item() * 100 for i in range(4)}
                st.bar_chart(prob_dict)

                # Export report PDF
                try:
                    import time
                    temp_dir = "./temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, f"streamlit_temp_{int(time.time())}.jpg")
                    image.save(temp_path)
                    
                    from export_report import generate_report
                    pdf_report_path = generate_report(temp_path, out_folder=temp_dir)
                    
                    with open(pdf_report_path, "rb") as f:
                        pdf_data = f.read()
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_data,
                        file_name=f"Report_{int(time.time())}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Could not generate PDF report: {e}")

                # Explainable AI
                if model_choice == "ResNet18 (Transfer Learning)":
                    st.subheader("Saliency Map (Grad-CAM)")
                    st.write("Highlights image regions contributing most to current prediction:")
                    overlay = get_gradcam_overlay(image, selected_model, device)
                    st.image(overlay, clamp=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
