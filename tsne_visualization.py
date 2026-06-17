import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision import datasets

EMBEDDINGS_DIR = './outputs/embeddings'
OUTPUT_DIR = './outputs/tsne'
DATA_DIR = './dataset'

def plot_tsne(features, labels, class_names, model_name):
    print(f"Running t-SNE for {model_name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    
    # Define a distinct color palette
    palette = sns.color_palette("husl", len(class_names))
    
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=[class_names[l] for l in labels],
        palette=palette,
        alpha=0.8,
        s=60
    )
    
    plt.title(f't-SNE Visualization - {model_name} Features', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(title='Classes', title_fontsize='13', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot
    save_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_tsne.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {model_name} t-SNE plot to {save_path}")

def generate_tsne():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try to load class names dynamically, or fallback to defaults
    try:
        class_names = datasets.ImageFolder(DATA_DIR).classes
    except Exception:
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    try:
        resnet_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, 'resnet_embeddings.npy'))
        vit_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, 'vit_embeddings.npy'))
        labels = np.load(os.path.join(EMBEDDINGS_DIR, 'labels.npy'))
    except FileNotFoundError as e:
        print(f"Error loading embeddings: {e}")
        print("Please run extract_embeddings.py first.")
        return

    plot_tsne(resnet_embeddings, labels, class_names, "ResNet18")
    plot_tsne(vit_embeddings, labels, class_names, "ViT-B")

if __name__ == "__main__":
    generate_tsne()
