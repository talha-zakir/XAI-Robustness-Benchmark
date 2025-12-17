
import torch
import numpy as np
import cv2
from abc import ABC, abstractmethod

class Shift(ABC):
    def __init__(self, name, severities=[1, 2, 3, 4, 5]):
        self.name = name
        self.severities = severities
    
    @abstractmethod
    def apply(self, x, severity):
        """
        Apply shift to input tensor x.
        x: torch.Tensor (C, H, W), normalized or unnormalized depending on requirement.
           For this benchmark, we assume x is a normalized tensor, so we might need to unnormalize -> corrupt -> normalize,
           OR we apply corruptions before normalization in the pipeline.
           
           DECISION: It is standard to apply corruptions on the raw image (0-255 or 0-1) BEFORE normalization.
           However, our loader returns normalized tensors.
           We will implement a helper to denormalize -> corrupt -> normalize.
        """
        pass

# Helper for denormalization (CIFAR-10 specific stats)
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

def denormalize(x):
    return x * STD + MEAN

def normalize(x):
    return (x - MEAN) / STD

def tensor_to_np(x):
    # x: (C, H, W) -> (H, W, C) numpy uint8
    x = denormalize(x)
    x = torch.clamp(x, 0, 1)
    x = x.permute(1, 2, 0).cpu().numpy()
    return (x * 255).astype(np.uint8)

def np_to_tensor(x):
    # x: (H, W, C) uint8 -> (C, H, W) normalized tensor
    x = x.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    return normalize(x)

class GaussianNoise(Shift):
    def __init__(self):
        super().__init__("Gaussian Noise")
        # std deviations for noise
        self.params = {
            1: 0.04, 2: 0.08, 3: 0.12, 4: 0.18, 5: 0.26
        }
        
    def apply(self, x, severity):
        # Noise fits best on float images [0,1] or normalized.
        # Let's apply on normalized directly for simplicity, or 0-1?
        # Standard benchmark (CIFAR-10-C) applies on unit 8 usually.
        # But simple additive noise is fine on tensor.
        sigma = self.params[severity]
        return x + torch.randn_like(x) * sigma

class GaussianBlur(Shift):
    def __init__(self):
        super().__init__("Gaussian Blur")
        # kernel sizes (must be odd)
        self.params = {
            1: 3, 2: 5, 3: 7, 4: 9, 5: 11
        }
        
    def apply(self, x, severity):
        k = self.params[severity]
        img = tensor_to_np(x)
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        return np_to_tensor(blurred)

class Brightness(Shift):
    def __init__(self):
        super().__init__("Brightness")
        # value to add in hsv
        self.params = {
            1: 10, 2: 20, 3: 30, 4: 40, 5: 50
        }
    
    def apply(self, x, severity):
        val = self.params[severity]
        img = tensor_to_np(x)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply shift
        v = cv2.add(v, val)
        
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return np_to_tensor(img)

class Contrast(Shift):
    def __init__(self):
        super().__init__("Contrast")
        self.params = {
            1: 0.8, 2: 0.6, 3: 0.4, 4: 0.3, 5: 0.2
        }
        
    def apply(self, x, severity):
        alpha = self.params[severity]
        img = tensor_to_np(x)
        # simplistic contrast: x = (x - mean) * alpha + mean
        # BUT standard robust bench uses PIL ImageEnhance or just scaling
        # Let's use simple scaling around 127
        img = img.astype(np.float32)
        mean = 127.0
        img = (img - mean) * alpha + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        return np_to_tensor(img)

# Registry
SHIFTS = {
    'gaussian_noise': GaussianNoise(),
    'gaussian_blur': GaussianBlur(),
    'brightness': Brightness(),
    'contrast': Contrast()
}
