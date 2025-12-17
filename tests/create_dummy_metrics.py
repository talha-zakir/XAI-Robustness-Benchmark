
import pandas as pd
import numpy as np
import os

def main():
    os.makedirs('outputs/runs/dummy_run', exist_ok=True)
    
    records = []
    shifts = ['gaussian_noise', 'gaussian_blur']
    explainers = ['gradcam', 'ig']
    
    for i in range(10): # 10 samples
        for shift in shifts:
            for severity in [1, 2, 3, 4, 5]:
                for explainer in explainers:
                    # Synthetic data
                    # Acc drops with severity
                    acc_prob = max(0.1, 1.0 - severity * 0.15)
                    is_consistent = np.random.random() < acc_prob
                    
                    # SSIM drops faster
                    ssim = max(0, 1.0 - severity * 0.2 + np.random.normal(0, 0.05))
                    
                    records.append({
                        'sample_idx': i,
                        'true_label': 0,
                        'clean_pred': 0,
                        'shift_pred': 0 if is_consistent else 1,
                        'is_consistent': is_consistent,
                        'shift': shift,
                        'severity': severity,
                        'explainer': explainer,
                        'ssim': ssim,
                        'cosine': 0.8,
                        'topk': 0.5,
                        'drift': 0.1
                    })
                    
    df = pd.DataFrame(records)
    df.to_csv('outputs/runs/dummy_run/metrics.csv', index=False)
    print("Created dummy metrics.csv")

if __name__ == '__main__':
    main()
