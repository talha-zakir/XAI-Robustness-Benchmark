
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_acc_vs_similarity(df, output_path):
    """
    Plot Accuracy and Explanation Similarity vs Severity.
    df: DataFrame with cols [severity, acc, ssim, explainer]
    """
    sns.set_style("whitegrid")
    
    # We expect 'acc' to be in the DF, or we compute it from 'is_consistent' logic relative to clean?
    # In benchmark runner, we didn't explicitly store 'acc' per severity row in a way that aggregates easily for the plot 
    # unless we group by severity.
    # We will assume the input df is the raw metrics.csv.
    
    # Aggregation
    shifts = df['shift'].unique()
    explainers = df['explainer'].unique()
    
    for shift in shifts:
        plt.figure(figsize=(10, 6))
        
        # Filter by shift
        shift_df = df[df['shift'] == shift]
        
        for explainer in explainers:
            exp_df = shift_df[shift_df['explainer'] == explainer]
            
            # Group by severity
            grouped = exp_df.groupby('severity').agg({
                'is_consistent': 'mean', # This acts as Accuracy relative to clean prediction (Stability of prediction)
                                         # OR if we want Accuracy relative to GT, we need (shift_pred == true_label)
                                         # Our runner saves true_label, shift_pred.
                'ssim': 'mean'
            }).reset_index()
            
            # Compute actual accuracy
            # We need to re-calc accuracy from raw data
            # actually 'is_consistent' is consistency with clean. 
            # We might want robustness (acc on shifted).
            
            # Let's calculate Accuracy (Shift Pred == True Label)
            acc_series = exp_df.groupby('severity').apply(
                lambda x: (x['shift_pred'] == x['true_label']).mean()
            )
            
            # Plot Accuracy (Dashed)
            plt.plot(grouped['severity'], acc_series, 
                     linestyle='--', marker='o', alpha=0.7, 
                     label=f'Acc ({explainer})') # Acc is method-agnostic but lines might overlap
            
            # Plot SSIM (Solid)
            plt.plot(grouped['severity'], grouped['ssim'], 
                     linestyle='-', marker='s', linewidth=2, 
                     label=f'SSIM ({explainer})')
            
        plt.xlabel('Severity')
        plt.ylabel('Score')
        plt.title(f'Accuracy vs Explanation Stability: {shift}')
        plt.legend()
        plt.ylim(0, 1.05)
        
        if output_path:
            plt.savefig(os.path.join(output_path, f'acc_vs_sim_{shift}.png'))
            plt.close()
            
def plot_method_rankings(df, output_path):
    pass # Can implement later
