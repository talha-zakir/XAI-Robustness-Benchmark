
import torch
from captum.attr import IntegratedGradients

class IGExplainer:
    def __init__(self, model):
        self.model = model
        self.ig = IntegratedGradients(model)
        
    def explain(self, x, target_class=None, steps=50):
        """
        x: (1, C, H, W)
        """
        self.model.eval()
        if target_class is None:
            output = self.model(x)
            target_class = output.argmax(dim=1).item()
            
        attr = self.ig.attribute(x, target=target_class, n_steps=steps)
        
        # IG returns (1, C, H, W). To get a 2D map, we usually sum across channels or take norm.
        # Common practice: sum(abs(attr), dim=1)
        attr_map = torch.sum(torch.abs(attr), dim=1, keepdim=True)
        
        return attr_map
