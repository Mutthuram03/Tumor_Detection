import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=4):
    """
    Loads a pretrained ResNet18 model and modifies the final fully connected 
    layer to match the number of classes for our task.
    """
    # Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
