
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam

class GradCAMExplainer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gc = LayerGradCam(model, target_layer)
        
    def explain(self, x, target_class=None):
        """
        x: (1, C, H, W) input tensor
        target_class: int or None. If None, uses predicted class.
        """
        self.model.eval()
        if target_class is None:
            output = self.model(x)
            target_class = output.argmax(dim=1).item()
            
        # Captum expects tensor for target
        target = target_class 
        
        # attribute returns (1, C_featre, H, W). GradCAM usually upsamples to input size.
        # LayerGradCam returns the CAM upsampled to the layer output size, not input size usually?
        # Captum LayerGradCam: "The returned attributions are the result of upsampling the CAMs... to the input size" 
        # (Wait, check captum docs or assume behavior. Usually it needs attribute_to_layer_input=True or manual upsample)
        # Actually Captum's LayerGradCam returns attribution w.r.t layer.
        # But we want the heatmap.
        # Let's try basic attribution.
        
        attr = self.gc.attribute(x, target=target, relu_attributions=True)
        
        # Upsample to input size (32x32 for CIFAR)
        attr = F.interpolate(attr, size=(32, 32), mode='bilinear', align_corners=False)
        
        return attr
