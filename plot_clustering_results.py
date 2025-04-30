import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from matplotlib.patches import Patch
import seaborn as sns

def load_data(cache_dir):
    """Load the data matrix and group labels."""
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Load group names if available
    group_names_file = os.path.join(cache_dir, "group_names.txt")
    if os.path.exists(group_names_file):
        with open(group_names_file, "r") as f:
            group_names = [line.strip() for line in f.readlines()]
    else:
        unique_groups = sorted(np.unique(group_labels))
        group_names = [f"Group {g}" for g in unique_groups]
    
    return data_matrix, group_labels, group_names

def load_sf_results(cache_dir, k):
    """Load Samira's SF results for a specific k."""
    results_dir = os.path.join(cache_dir, "sf_results")
    if not os.path.exists(results_dir):
        return None
    
    # Look for detailed_results_k{k}.json file
    json_file = os.path.join(results_dir, f"detailed_results_k{k}.json")
    if not os.path.exists(json_file):
        return None
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Extract centers and assignment from the JSON structure
    if 'data' not in results:
        return None
    
    data = results['data']
    if 'centers' not in data or 'assignment' not in data:
        return None
    
    return {
        'centers': np.array(data['centers']),
        'assignment': np.array(data['assignment'])
    }

def load_fcbc_results(cache_dir, k):
    """Load FCBC results for a specific k."""
    results_dir = os.path.join(cache_dir, "fcbc_results")
    if not os.path.exists(results_dir):
        return None
    
    # Look for CSV files with k in the name
    csv_files = [f for f in os.listdir(results_dir) if f.startswith(f'fcbc_k{k}') and f.endswith('.csv')]
    if not csv_files:
        return None
    
    # Use the first matching file
    csv_path = os.path.join(results_dir, csv_files[0])
    data = pd.read_csv(csv_path)
    
    if len(data) == 0:
        return None
    
    # Get the row with minimum fair_cost
    best_result = data.loc[data['fair_cost'].idxmin()]
    
    # Extract centers and assignment
    centers = np.array(eval(best_result['centers']))
    assignment = np.array(eval(best_result['assignment']))
    
    return {
        'centers': centers,
        'assignment': assignment
    }

def load_wc_results(cache_dir, k, lambda_param):
    """Load Welfare Clustering results for a specific k and lambda."""
    results_dir = os.path.join(cache_dir, "welfare_clustering_results")
    if not os.path.exists(results_dir):
        return None
    
    # Format lambda value with underscore instead of decimal point
    lambda_str = f"{lambda_param:.3f}".replace('.', '_')
    json_file = os.path.join(results_dir, f"detailed_results_k{k}_lambda_{lambda_str}.json")
    
    if not os.path.exists(json_file):
        return None
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    k_str = str(k)
    if k_str not in results['results'] or 'fair' not in results['results'][k_str]:
        return None
    
    fair_results = results['results'][k_str]['fair']
    centers = np.array(fair_results['centers'])
    assignment = np.array(fair_results['assignment'])
    
    # Convert binary assignment matrix to indices if needed
    if len(assignment.shape) > 1:
        assignment = np.argmax(assignment, axis=1)
    
    return {
        'centers': centers,
        'assignment': assignment
    }

def plot_clustering(data, group_labels, centers, assignment, group_names, title, ax):
    """Plot clustering results with centers and group colors."""
    # Create color palette for groups
    unique_groups = sorted(np.unique(group_labels))
    group_colors = sns.color_palette("husl", n_colors=len(unique_groups))
    group_color_dict = dict(zip(unique_groups, group_colors))
    
    # Plot points colored by group
    for group in unique_groups:
        mask = group_labels == group
        ax.scatter(data[mask, 0], data[mask, 1], 
                  c=[group_color_dict[group]], 
                  alpha=0.6, 
                  label=group_names[group])
    
    # Plot cluster centers
    ax.scatter(centers[:, 0], centers[:, 1], 
              c='black', marker='*', s=200, 
              label='Centers')
    
    # Draw lines from points to their assigned centers
    for i in range(len(data)):
        center_idx = assignment[i]
        ax.plot([data[i, 0], centers[center_idx, 0]], 
                [data[i, 1], centers[center_idx, 1]], 
                'gray', alpha=0.1)
    
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_all_results(cache_dir, k, lambda_param):
    """Plot clustering results from all three pipelines for a specific k."""
    # Load data
    data_matrix, group_labels, group_names = load_data(cache_dir)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Load and plot SF results
    sf_results = load_sf_results(cache_dir, k)
    if sf_results:
        plot_clustering(data_matrix, group_labels, 
                       sf_results['centers'], sf_results['assignment'],
                       group_names, f"Socially Fair (k={k})", axes[0])
    else:
        axes[0].text(0.5, 0.5, "No SF results available", 
                    ha='center', va='center')
    
    # Load and plot FCBC results
    fcbc_results = load_fcbc_results(cache_dir, k)
    if fcbc_results:
        plot_clustering(data_matrix, group_labels,
                       fcbc_results['centers'], fcbc_results['assignment'],
                       group_names, f"FCBC (k={k})", axes[1])
    else:
        axes[1].text(0.5, 0.5, "No FCBC results available",
                    ha='center', va='center')
    
    # Load and plot WC results
    wc_results = load_wc_results(cache_dir, k, lambda_param)
    if wc_results:
        plot_clustering(data_matrix, group_labels,
                       wc_results['centers'], wc_results['assignment'],
                       group_names, f"Welfare Clustering (k={k}, λ={lambda_param})", axes[2])
    else:
        axes[2].text(0.5, 0.5, "No WC results available",
                    ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(cache_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    lambda_str = f"lambda_{str(lambda_param).replace('.', '_')}"
    plt.savefig(os.path.join(plots_dir, f"clustering_comparison_k{k}_{lambda_str}.png"),
                bbox_inches='tight', dpi=300)
    print(f"Plot saved as clustering_comparison_k{k}_{lambda_str}.png")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot clustering results from all pipelines")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Directory containing data and results")
    parser.add_argument("--k_min", type=int, required=True,
                       help="Minimum number of clusters to plot")
    parser.add_argument("--k_max", type=int, required=True,
                       help="Maximum number of clusters to plot")
    parser.add_argument("--lambda_param", type=float, required=True,
                       help="Lambda parameter to plot")
    
    args = parser.parse_args()
    
    for k in range(args.k_min, args.k_max + 1):
        print(f"\nPlotting results for k={k}")
        plot_all_results(args.cache_dir, k, args.lambda_param)

if __name__ == "__main__":
    main() 