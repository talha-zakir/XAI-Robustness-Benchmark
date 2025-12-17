
import argparse
import os
import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.datasets.cifar10 import load_cifar10
from src.models.build_model import build_model
from src.shifts.corruptions import SHIFTS, normalize, denormalize
from src.explainers.gradcam import GradCAMExplainer
from src.explainers.integrated_gradients import IGExplainer
from src.explainers.utils import normalize_attribution, show_heatmap
from src.metrics.similarity import compute_ssim, compute_cosine, compute_topk_overlap
from src.metrics.drift import compute_wasserstein
from src.metrics.decision import is_decision_consistent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs/runs/run_001')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to eval')
    parser.add_argument('--batch_size', type=int, default=1) # Keep 1 for simple explainer looping
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, 'figures/qualitative_panels')
    os.makedirs(fig_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model
    model = build_model('resnet18', num_classes=10)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Data (Test set)
    _, testloader, classes = load_cifar10(args.data_root, batch_size=1) # Shuffle False for reproducibility
    
    # Initialize Explainers
    gradcam = GradCAMExplainer(model, target_layer=model.layer4[1].conv2)
    ig = IGExplainer(model)
    
    explainers = {
        'gradcam': gradcam,
        'ig': ig
    }
    
    records = []
    
    # Select subset of indices
    # We iterate loader until we hit samples count
    
    print("Starting Benchmark...")
    
    sample_count = 0
    pbar = tqdm(total=args.samples * len(SHIFTS) * 5) # approx progress
    
    # Pre-fetch samples to ensure we use same samples for all shifts
    # (Actually simpler to loop samples, then loop shifts per sample)
    
    for i, (img, label) in enumerate(testloader):
        if sample_count >= args.samples:
            break
        
        img = img.to(device)
        label = label.item()
        
        # 1. Clean Prediction & Explanation
        with torch.no_grad():
            clean_out = model(img)
            clean_pred = clean_out.argmax(dim=1).item()
        
        # Compute clean explanations
        clean_attrs = {}
        for name, explainer in explainers.items():
            # Target: predicted class (Mode A)
            attr = explainer.explain(img, target_class=clean_pred)
            clean_attrs[name] = normalize_attribution(attr) # Normalized [0,1] numpy
            
        # 2. Apply Shifts
        for shift_name, shift_obj in SHIFTS.items():
            for severity in getattr(shift_obj, 'severities', [1,2,3,4,5]): # Use method or default
                # Apply shift
                shifted_img = shift_obj.apply(img.clone().cpu().squeeze(0), severity).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    shift_out = model(shifted_img)
                    shift_pred = shift_out.argmax(dim=1).item()
                
                is_consistent = is_decision_consistent(clean_pred, shift_pred)
                
                # Explain
                for name, explainer in explainers.items():
                    # Target: Explain the NEW prediction? Or original?
                    # "Mode A: explain the predicted class each time" -> shift_pred
                    shift_attr = explainer.explain(shifted_img, target_class=shift_pred)
                    shift_attr_norm = normalize_attribution(shift_attr)
                    
                    clean_attr_norm = clean_attrs[name]
                    
                    # Metrics
                    ssim_val = compute_ssim(clean_attr_norm, shift_attr_norm)
                    cosine_val = compute_cosine(clean_attr_norm, shift_attr_norm)
                    topk_val = compute_topk_overlap(clean_attr_norm, shift_attr_norm)
                    drift_val = compute_wasserstein(clean_attr_norm, shift_attr_norm)
                    
                    record = {
                        'sample_idx': i,
                        'true_label': label,
                        'clean_pred': clean_pred,
                        'shift_pred': shift_pred,
                        'is_consistent': is_consistent,
                        'shift': shift_name,
                        'severity': severity,
                        'explainer': name,
                        'ssim': ssim_val,
                        'cosine': cosine_val,
                        'topk': topk_val,
                        'drift': drift_val
                    }
                    records.append(record)
                    
                    # Save Qualitative (First 5 samples only, for severity 3 and 5)
                    if sample_count < 5 and severity in [3, 5] and name == 'gradcam':
                        # Save plot
                        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
                        
                        # Clean Image
                        clean_np = denormalize(img[0].cpu()).permute(1,2,0).numpy()
                        clean_np = np.clip(clean_np, 0, 1)
                        axs[0].imshow(clean_np)
                        axs[0].set_title(f"Clean (Pred: {classes[clean_pred]})")
                        axs[0].axis('off')
                        
                        # Clean Attr
                        axs[1].imshow(clean_attr_norm.squeeze(), cmap='jet')
                        axs[1].set_title("Clean Attr")
                        axs[1].axis('off')
                        
                        # Shifted Image
                        shift_np = denormalize(shifted_img[0].cpu()).permute(1,2,0).numpy()
                        shift_np = np.clip(shift_np, 0, 1)
                        axs[2].imshow(shift_np)
                        axs[2].set_title(f"{shift_name} {severity} (Pred: {classes[shift_pred]})")
                        axs[2].axis('off')
                        
                        # Shifted Attr
                        axs[3].imshow(shift_attr_norm.squeeze(), cmap='jet')
                        axs[3].set_title(f"SSIM: {ssim_val:.2f}")
                        axs[3].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_dir, f"sample_{i}_{shift_name}_sev{severity}_{name}.png"))
                        plt.close(fig)

                pbar.update(1)
        
        sample_count += 1
        
    pbar.close()
    
    # Save Results
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    # Summary
    print("\nSummary Results (Mean SSIM per severity):")
    print(df.groupby(['explainer', 'shift', 'severity'])['ssim'].mean())

if __name__ == '__main__':
    main()
