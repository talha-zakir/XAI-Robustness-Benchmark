
import torch
import numpy as np
import matplotlib.pyplot as plt

def normalize_attribution(attr):
    """
    Normalizes attribution map to [0, 1] range.
    attr: Tensor or Array
    """
    if isinstance(attr, torch.Tensor):
        attr = attr.detach().cpu().numpy()
        
    # Standard min-max normalization
    # If using absolute values (often done for visualization)
    attr = np.abs(attr)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    return attr

def show_heatmap(attr, img=None, title="Attribution", alpha=0.5):
    """
    Display heatmap overlay.
    img: (H, W, C) float [0,1]
    attr: (H, W) or (1, H, W) normalized [0,1]
    """
    if len(attr.shape) == 3:
        attr = attr.squeeze(0)
        
    plt.figure()
    if img is not None:
        plt.imshow(img)
        
    plt.imshow(attr, cmap='jet', alpha=alpha)
    plt.title(title)
    plt.axis('off')
    return plt.gcf()
