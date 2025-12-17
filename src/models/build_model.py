
import torch
import torch.nn as nn
import torchvision.models as models

def build_model(model_name='resnet18', num_classes=10, pretrained=False):
    """
    Builds a model from torchvision.models.
    
    Args:
        model_name (str): Name of the model (e.g., 'resnet18').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load ImageNet weights.
        
    Returns:
        model (nn.Module): The requested model.
    """
    if model_name == 'resnet18':
        # Weights.DEFAULT corresponds to the best available weights
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        # Modify the final layer for CIFAR-10 (10 classes instead of 1000)
        # ResNet18 fc layer: (fc): Linear(in_features=512, out_features=1000, bias=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # NOTE: CIFAR-10 images are 32x32. ResNet is designed for 224x224.
        # Ideally, we should modify the first conv layer to handle small images better,
        # but for a standard baseline using torchvision, we often just use it as is 
        # or remove the first maxpool.
        # For this robust benchmark, let's keep it simple (standard ResNet) first, 
        # but modifying conv1 is a common trick for CIFAR.
        # Let's Modify conv1 to avoid excessive downsampling for 32x32 images.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity() # Remove maxpool
        
    else:
        raise NotImplementedError(f"Model {model_name} not implemented yet.")
        
    return model
