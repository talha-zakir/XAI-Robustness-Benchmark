
import argparse
import pandas as pd
import os
from src.viz.plots import plot_acc_vs_similarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Path to experiment run directory')
    args = parser.parse_args()
    
    csv_path = os.path.join(args.run_dir, 'metrics.csv')
    if not os.path.exists(csv_path):
        print("metrics.csv not found!")
        return
        
    df = pd.read_csv(csv_path)
    
    # Generate Plots
    fig_dir = os.path.join(args.run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    print("Generating Accuracy vs Stability plots...")
    plot_acc_vs_similarity(df, fig_dir)
    
    # Generate Summary Table
    print("\nRobustness Scores (Avg SSIM on Consistent Predictions):")
    consistent_df = df[df['is_consistent'] == True]
    robustness = consistent_df.groupby(['explainer', 'shift'])['ssim'].mean().unstack()
    print(robustness)
    
    # Save Summary
    summary_path = os.path.join(args.run_dir, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write("Robustness Ranking (Higher SSIM is better):\n")
        f.write(robustness.to_string())
    
    print(f"Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
