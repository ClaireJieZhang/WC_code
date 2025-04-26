import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_welfare_costs(cache_dir, k_range):
    """
    Load welfare costs for all approaches for a range of k values.
    Returns a dictionary of DataFrames, one for each approach.
    """
    results = {}
    
    # Load FCBC results
    fcbc_results = []
    for k in k_range:
        file_path = os.path.join(cache_dir, "welfare_evaluation", f"fcbc_k{k}_welfare_costs.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            fcbc_results.append(df)
    if fcbc_results:
        results['FCBC'] = pd.concat(fcbc_results, ignore_index=True)
    
    # Load Welfare Clustering results
    wc_results = []
    for k in k_range:
        file_path = os.path.join(cache_dir, "welfare_evaluation", f"welfare_clustering_k{k}_welfare_costs.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            wc_results.append(df)
    if wc_results:
        results['Welfare Clustering'] = pd.concat(wc_results, ignore_index=True)
    
    # Load Samira's SF results
    sf_results = []
    for k in k_range:
        file_path = os.path.join(cache_dir, "welfare_evaluation", f"samira_sf_k{k}_welfare_costs.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            sf_results.append(df)
    if sf_results:
        results['Samira SF'] = pd.concat(sf_results, ignore_index=True)
    
    return results

def plot_welfare_costs(cache_dir, k_range, lambda_param=0.5):
    """
    Plot welfare costs for different approaches.
    """
    # Load results
    results = load_welfare_costs(cache_dir, k_range)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot each approach
    for approach, df in results.items():
        # Sort by k value
        df = df.sort_values('k')
        plt.plot(df['k'], df['max_welfare_cost'], 
                marker='o', label=approach, linewidth=2)
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Welfare Cost')
    plt.title(f'Welfare Costs for Different Approaches (λ={lambda_param})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    output_dir = os.path.join(cache_dir, "welfare_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'welfare_costs_comparison_k{min(k_range)}_to_{max(k_range)}.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot welfare costs for different approaches")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, default=8, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight between distance and fairness costs")
    
    args = parser.parse_args()
    
    k_range = range(args.k_min, args.k_max + 1)
    plot_welfare_costs(args.cache_dir, k_range, args.lambda_param) 