import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_vit_b_16(num_classes=4):
    """
    Loads a pretrained ViT-B/16 model and modifies the classification head
    for the specified number of classes.
    """
    # Load pretrained Vision Transformer (ViT-B/16)
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # Freeze earlier layers if needed (optional, keeping it unfrozen for fine-tuning)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the classification head
    # The classification head in torchvision's ViT is named 'heads' and is a Sequential
    # containing a single Linear layer.
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model

if __name__ == "__main__":
    # Test the model structure
    model = get_vit_b_16(num_classes=4)
    print(model)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: [1, 4]
