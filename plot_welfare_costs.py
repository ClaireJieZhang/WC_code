#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_welfare_costs(cache_dir, k_min, k_max, lambda_param):
    """Load welfare costs from all three pipelines."""
    lambda_str = f"lambda_{str(lambda_param).replace('.', '_')}"
    welfare_dir = os.path.join(cache_dir, "welfare_evaluation")
    
    results = {}
    
    # Load results from each pipeline
    pipeline_files = {
        'SF': f"samira_sf_all_k{k_min}_to_{k_max}_{lambda_str}_summary.csv",
        'FCBC': f"fcbc_all_k{k_min}_to_{k_max}_{lambda_str}_summary.csv",
        'WC': f"welfare_clustering_all_k{k_min}_to_{k_max}_{lambda_str}_summary.csv"
    }
    
    for method, filename in pipeline_files.items():
        filepath = os.path.join(welfare_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            results[method] = df
        else:
            print(f"Warning: No results found for {method} at {filepath}")
    
    return results

def plot_welfare_costs(results, k_min, k_max, lambda_param, save_path):
    """Plot welfare costs comparison for all methods."""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'SF': 'tab:blue',
        'FCBC': 'tab:orange',
        'WC': 'tab:green'
    }
    
    for method, df in results.items():
        if df is not None and not df.empty:
            plt.plot(df['k'], df['max_welfare_cost'], 
                    marker='o', label=method, color=colors[method])
            
            # Add value labels
            for k, cost in zip(df['k'], df['max_welfare_cost']):
                plt.annotate(f'{cost:.3f}', 
                           (k, cost), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center')
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Maximum Welfare Cost')
    plt.title(f'Welfare Cost Comparison (λ={lambda_param})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set x-axis ticks to be exactly at k values
    plt.xticks(range(k_min, k_max + 1))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_group_costs(results, k_min, k_max, lambda_param, save_path):
    """Plot per-group average costs for all methods."""
    # Create a new figure for group costs
    plt.figure(figsize=(15, 8))
    
    # Get all unique group columns
    group_cols = []
    for df in results.values():
        if df is not None and not df.empty:
            group_cols.extend([col for col in df.columns if col.startswith('group_') and col.endswith('_avg_cost')])
    group_cols = sorted(set(group_cols))
    
    # Number of groups
    n_groups = len(group_cols)
    
    # Create subplots for each k value
    k_values = range(k_min, k_max + 1)
    n_cols = min(3, len(k_values))
    n_rows = (len(k_values) + n_cols - 1) // n_cols
    
    for i, k in enumerate(k_values, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # Plot bars for each method and group
        x = np.arange(n_groups)
        width = 0.25  # Width of bars
        
        for j, (method, df) in enumerate(results.items()):
            if df is not None and not df.empty:
                k_data = df[df['k'] == k]
                if not k_data.empty:
                    costs = [k_data[col].iloc[0] for col in group_cols]
                    plt.bar(x + j*width, costs, width, label=method)
        
        plt.title(f'k={k}')
        if i % n_cols == 1:  # Only for leftmost plots
            plt.ylabel('Average Cost')
        plt.xticks(x + width, [col.split('_')[1] for col in group_cols], rotation=45)
    
    # Add a single legend for all subplots
    plt.figlegend(loc='center right')
    plt.suptitle(f'Per-Group Average Costs (λ={lambda_param})')
    
    # Save plot
    group_costs_path = save_path.replace('.png', '_group_costs.png')
    plt.tight_layout()
    plt.savefig(group_costs_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot welfare costs from evaluation results")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Directory containing cached data and welfare_evaluation results")
    parser.add_argument("--k_min", type=int, required=True,
                       help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, required=True,
                       help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, required=True,
                       help="Lambda parameter to plot")
    parser.add_argument("--output", type=str, default="welfare_costs_comparison.png",
                       help="Output file path for the plot")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only try to create directory if there is a directory path
        os.makedirs(output_dir, exist_ok=True)
    
    # Load results from all pipelines
    results = load_welfare_costs(
        cache_dir=args.cache_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        lambda_param=args.lambda_param
    )
    
    if not results:
        print("No results found for any pipeline")
        return
    
    # Plot welfare costs comparison
    plot_welfare_costs(
        results=results,
        k_min=args.k_min,
        k_max=args.k_max,
        lambda_param=args.lambda_param,
        save_path=args.output
    )
    print(f"Welfare costs plot saved to: {args.output}")
    
    # Plot per-group costs
    plot_group_costs(
        results=results,
        k_min=args.k_min,
        k_max=args.k_max,
        lambda_param=args.lambda_param,
        save_path=args.output
    )
    group_costs_path = args.output.replace('.png', '_group_costs.png')
    print(f"Group costs plot saved to: {group_costs_path}")

if __name__ == "__main__":
    main() 