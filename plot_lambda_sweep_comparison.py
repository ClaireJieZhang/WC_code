import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle
import configparser
import ast
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost_with_slack

def plot_wc_clusters(data_matrix, centers, assignment, group_labels, lambda_val, k, output_dir):
    """
    Plot WC clusters in 2D for a specific lambda value.
    Points are colored by group labels and connected to their centers by lines.
    
    Args:
        data_matrix: numpy array of shape (n_points, 2) - 2D points
        centers: numpy array of shape (k, 2) - 2D centers
        assignment: numpy array of shape (n_points, k) containing one-hot cluster assignments
        group_labels: numpy array of shape (n_points,) containing group labels
        lambda_val: lambda value used for this clustering
        k: number of clusters
        output_dir: directory to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get unique group labels
    unique_groups = np.unique(group_labels)
    group_colors = ['red', 'blue']  # One color per group
    
    # First plot all points colored by group
    for group_idx, group in enumerate(unique_groups):
        # Get points in this group
        group_mask = group_labels == group
        group_points = data_matrix[group_mask]  # Shape: (n_group_points, 2)
        
        # Plot all points for this group
        plt.scatter(group_points[:, 0], group_points[:, 1],
                   color=group_colors[group_idx], marker='o',
                   label=f'Group {group}',
                   alpha=0.6)
    
    # Then plot cluster assignments with lines
    for cluster in range(k):
        # Get points assigned to this cluster
        cluster_mask = assignment[:, cluster] == 1
        cluster_points = data_matrix[cluster_mask]
        
        # Plot lines from points to their center
        for point in cluster_points:
            plt.plot([point[0], centers[cluster, 0]], 
                    [point[1], centers[cluster, 1]],
                    'k-', alpha=0.2, linewidth=0.5)
    
    # Plot centers
    for i in range(k):
        plt.scatter(centers[i, 0], centers[i, 1],
                   marker='*', s=200, color='black',
                   label=f'Center {i}' if i == 0 else "")
    
    plt.title(f'Welfare Clustering Results (k={k}, λ={lambda_val:.3f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'wc_clusters_k{k}_lambda_{lambda_val:.3f}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def load_alpha_beta_from_config(config_file):
    """Load alpha and beta parameters from the config file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Parse alpha and beta from config
    alpha = ast.literal_eval(config['params']['alpha'])
    beta = ast.literal_eval(config['params']['beta'])
    
    return alpha, beta

def load_welfare_clustering_results(results_dir, k, lambda_values):
    """Load all welfare clustering results for different lambda values"""
    print(f"\nLooking for WC results in: {results_dir}")
    results = []
    
    # Load data matrix and group labels for plotting
    data_matrix = np.load(os.path.join(os.path.dirname(results_dir), "data_matrix.npy"))
    group_labels = np.load(os.path.join(os.path.dirname(results_dir), "group_labels.npy"))
    
    # Load alpha and beta from config
    config_file = "configs/welfare_clustering_config.ini"
    alpha, beta = load_alpha_beta_from_config(config_file)
    
    for lambda_val in lambda_values:
        # Format lambda value with underscore instead of decimal point for filename
        lambda_str = f"{lambda_val:.3f}".replace('.', '_')
        
        # Look for both CSV and JSON files
        csv_file = os.path.join(results_dir, f"welfare_clustering_k{k}_lambda_{lambda_str}.csv")
        json_file = os.path.join(results_dir, f"detailed_results_k{k}_lambda_{lambda_str}.json")
        
        if os.path.exists(csv_file) and os.path.exists(json_file):
            print(f"Found files for lambda = {lambda_val:.3f}")
            
            # Load summary from CSV
            data = pd.read_csv(csv_file)
            
            # Load detailed results from JSON
            with open(json_file, 'r') as f:
                detailed_data = json.load(f)
            
            # Extract centers and assignment from the nested structure
            centers = np.array(detailed_data['results'][str(k)]['fair']['centers'])
            # Convert one-hot assignment to cluster indices
            one_hot_assignment = np.array(detailed_data['results'][str(k)]['fair']['assignment'])
            assignment = np.argmax(one_hot_assignment, axis=1)  # Shape: (n_points,)
            
            # Calculate welfare cost using the same function as SF and FCBC
            max_welfare_cost, group_costs, cluster_stats, group_distance_costs, group_fairness_costs = evaluate_welfare_cost_with_slack(
                centers=centers,
                assignment=assignment,
                points=data_matrix,
                group_labels=group_labels,
                lambda_param=lambda_val,
                alpha=alpha,
                beta=beta
            )
            
            # Plot clusters for this lambda value
            plot_wc_clusters(data_matrix, centers, one_hot_assignment, group_labels, lambda_val, k, results_dir)
            
            result_dict = {
                'lambda': lambda_val,
                'max_welfare_cost': max_welfare_cost,
                'group_costs': group_costs,
                'group_distance_costs': group_distance_costs,
                'group_fairness_costs': group_fairness_costs
            }
            results.append(result_dict)
            print(f"Loaded lambda: {lambda_val:.3f}, cost: {result_dict['max_welfare_cost']:.3f}")
        else:
            print(f"Warning: Missing files for lambda = {lambda_val:.3f}")
    
    if not results:
        print(f"No results found in {results_dir} for k={k}")
        print("Available files:", os.listdir(results_dir))
        return None
        
    print(f"Total results loaded: {len(results)}")
    sorted_results = sorted(results, key=lambda x: x['lambda'])
    print("Sorted lambda values:", [r['lambda'] for r in sorted_results])
    return sorted_results

def load_sf_results(results_dir, k, lambda_values):
    """Load Samira Fair results and evaluate welfare costs for all lambda values"""
    try:
        # Look for results in the sf_results directory
        sf_dir = os.path.join(results_dir, 'sf_results')
        if os.path.exists(sf_dir):
            # Load data matrix and group labels
            data_matrix = np.load(os.path.join(results_dir, "data_matrix.npy"))
            group_labels = np.load(os.path.join(results_dir, "group_labels.npy"))
            
            # Load alpha and beta from config
            config_file = "configs/welfare_clustering_config.ini"
            alpha, beta = load_alpha_beta_from_config(config_file)
            
            # Look for the detailed results file for this k
            detailed_file = os.path.join(sf_dir, f"detailed_results_k{k}.json")
            if os.path.exists(detailed_file):
                print(f"Loading SF results from {detailed_file}")
                with open(detailed_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract centers and assignment
                centers = np.array(data['data']['centers'])
                assignment = np.array(data['data']['assignment'])
                
                # Calculate welfare cost for each lambda value
                results = {}
                for lambda_val in lambda_values:
                    max_welfare_cost, group_costs, cluster_stats, group_distance_costs, group_fairness_costs = evaluate_welfare_cost_with_slack(
                        centers=centers,
                        assignment=assignment,
                        points=data_matrix,
                        group_labels=group_labels,
                        lambda_param=lambda_val,
                        alpha=alpha,
                        beta=beta
                    )
                    
                    results[lambda_val] = {
                        'max_welfare_cost': max_welfare_cost,
                        'group_costs': group_costs,
                        'group_distance_costs': group_distance_costs,
                        'group_fairness_costs': group_fairness_costs
                    }
                
                return results
            else:
                print(f"Warning: No detailed results file found for k={k} in SF results")
                print("Available files:", os.listdir(sf_dir))
                return None
        
        print(f"Warning: Could not find SF results in {sf_dir}")
        return None
    except Exception as e:
        print(f"Warning: Error loading SF results: {e}")
        return None

def load_fcbc_results(results_dir, k, lambda_values):
    """Load FCBC results and evaluate welfare costs for all lambda values"""
    try:
        fcbc_dir = os.path.join(results_dir, 'fcbc_results')
        if os.path.exists(fcbc_dir):
            # Load data matrix and group labels
            data_matrix = np.load(os.path.join(results_dir, "data_matrix.npy"))
            group_labels = np.load(os.path.join(results_dir, "group_labels.npy"))
            
            # Load alpha and beta from config
            config_file = "configs/welfare_clustering_config.ini"
            alpha, beta = load_alpha_beta_from_config(config_file)
            
            # Look for CSV files with k in the name
            csv_files = [f for f in os.listdir(fcbc_dir) if f.startswith(f'fcbc_k{k}') and f.endswith('.csv')]
            if csv_files:
                # Use the first matching file
                csv_path = os.path.join(fcbc_dir, csv_files[0])
                print(f"Loading FCBC results from {csv_path}")
                data = pd.read_csv(csv_path)
                
                if len(data) == 0:
                    print(f"Warning: Empty CSV file found at {csv_path}")
                    return None
                
                try:
                    # Calculate welfare cost for each row in the CSV
                    results = {}
                    for _, row in data.iterrows():
                        centers = np.array(eval(row['centers']))
                        assignment = np.array(eval(row['assignment']), dtype=int)
                        
                        # Calculate welfare cost for each lambda value
                        for lambda_val in lambda_values:
                            max_welfare_cost, group_costs, cluster_stats, group_distance_costs, group_fairness_costs = evaluate_welfare_cost_with_slack(
                                centers=centers,
                                assignment=assignment,
                                points=data_matrix,
                                group_labels=group_labels,
                                lambda_param=lambda_val,
                                alpha=alpha,
                                beta=beta
                            )
                            
                            # Debug print
                            print(f"\nFCBC Results for lambda={lambda_val:.3f}:")
                            print(f"Group distance costs: {group_distance_costs}")
                            print(f"Max welfare cost: {max_welfare_cost}")
                            
                            # Store all results for this lambda value
                            results[lambda_val] = {
                                'max_welfare_cost': max_welfare_cost,
                                'group_costs': group_costs,
                                'group_distance_costs': group_distance_costs,
                                'group_fairness_costs': group_fairness_costs
                            }
                    
                    return results
                except Exception as e:
                    print(f"Error parsing FCBC results: {e}")
                    print("First row of data:")
                    print(data.iloc[0])
                    return None
            else:
                print(f"Warning: No CSV files found for k={k} in FCBC results")
                print("Available files:", os.listdir(fcbc_dir))
                return None
        
        print(f"Warning: Could not find FCBC results in {fcbc_dir}")
        return None
    except Exception as e:
        print(f"Warning: Error loading FCBC results: {e}")
        return None

def plot_lambda_sweep_comparison(cache_dir, k, lambda_start=0.0, lambda_end=0.05, lambda_step=0.001, output_dir=None):
    """
    Plot comparison of welfare costs across different lambda values for all methods.
    
    Args:
        cache_dir: Base directory containing results
        k: Number of clusters
        lambda_start: Starting value for lambda sweep
        lambda_end: Ending value for lambda sweep
        lambda_step: Step size for lambda sweep
        output_dir: Directory to save the plot (default: cache_dir)
    """
    print(f"\nLoading results from: {cache_dir}")
    
    # Define the lambda values to match the sweep experiment
    lambda_values = np.arange(lambda_start, lambda_end + lambda_step, lambda_step)
    
    # Load results for each method
    wc_results = load_welfare_clustering_results(cache_dir, k, lambda_values)
    print(f"WC results type: {type(wc_results)}")
    if wc_results:
        print(f"Number of WC results: {len(wc_results)}")
        print("\nWelfare Clustering Results:")
        for result in wc_results:
            print(f"Lambda: {result['lambda']:.3f}, Cost: {result['max_welfare_cost']:.3f}")
    else:
        print("No WC results found, cannot proceed")
        return
    
    # Load SF and FCBC results from the parent directory
    parent_dir = os.path.dirname(cache_dir)
    sf_results = load_sf_results(parent_dir, k, lambda_values)
    fcbc_results = load_fcbc_results(parent_dir, k, lambda_values)
    
    # Create figure with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))
    
    # Plot each method
    if wc_results:
        lambdas = [float(result['lambda']) for result in wc_results]
        total_costs = [float(result['max_welfare_cost']) for result in wc_results]
        # Take maximum of group costs - group_distance_costs is a dictionary
        distance_costs = [float(max(result['group_distance_costs'].values())) for result in wc_results]
        fairness_costs = [float(max(result['group_fairness_costs'].values())) for result in wc_results]
        cost_ratios = [float(d/f) if f != 0 else float('inf') for d, f in zip(distance_costs, fairness_costs)]
        
        # Plot total welfare cost
        ax1.plot(lambdas, total_costs, 'o-', label='Welfare Clustering', color='blue', zorder=3)
        ax1.scatter(lambdas, total_costs, color='blue', s=100, zorder=3)
        
        # Plot distance cost
        ax2.plot(lambdas, distance_costs, 'o-', label='Welfare Clustering', color='blue', zorder=3)
        ax2.scatter(lambdas, distance_costs, color='blue', s=100, zorder=3)
        
        # Plot fairness violation
        ax3.plot(lambdas, fairness_costs, 'o-', label='Welfare Clustering', color='blue', zorder=3)
        ax3.scatter(lambdas, fairness_costs, color='blue', s=100, zorder=3)
        
        # Plot cost ratio
        ax4.plot(lambdas, cost_ratios, 'o-', label='Welfare Clustering', color='blue', zorder=3)
        ax4.scatter(lambdas, cost_ratios, color='blue', s=100, zorder=3)
    
    if sf_results:
        lambdas = lambda_values  # Use the same lambda values as the sweep
        total_costs = [float(sf_results[lam]['max_welfare_cost']) for lam in lambdas]
        # Take maximum of group costs - group_distance_costs is a dictionary
        distance_costs = [float(max(sf_results[lam]['group_distance_costs'].values())) for lam in lambdas]
        fairness_costs = [float(max(sf_results[lam]['group_fairness_costs'].values())) for lam in lambdas]
        cost_ratios = [float(d/f) if f != 0 else float('inf') for d, f in zip(distance_costs, fairness_costs)]
        
        # Plot total welfare cost
        ax1.plot(lambdas, total_costs, 'o-', label='Socially Fair', color='red', zorder=1)
        ax1.scatter(lambdas, total_costs, color='red', s=100, zorder=1)
        
        # Plot distance cost
        ax2.plot(lambdas, distance_costs, 'o-', label='Socially Fair', color='red', zorder=1)
        ax2.scatter(lambdas, distance_costs, color='red', s=100, zorder=1)
        
        # Plot fairness violation
        ax3.plot(lambdas, fairness_costs, 'o-', label='Socially Fair', color='red', zorder=1)
        ax3.scatter(lambdas, fairness_costs, color='red', s=100, zorder=1)
        
        # Plot cost ratio
        ax4.plot(lambdas, cost_ratios, 'o-', label='Socially Fair', color='red', zorder=1)
        ax4.scatter(lambdas, cost_ratios, color='red', s=100, zorder=1)
    
    if fcbc_results:
        lambdas = lambda_values  # Use the same lambda values as the sweep
        total_costs = [float(fcbc_results[lam]['max_welfare_cost']) for lam in lambdas]
        # Take maximum of group costs - group_distance_costs is a dictionary
        distance_costs = [float(max(fcbc_results[lam]['group_distance_costs'].values())) for lam in lambdas]
        fairness_costs = [float(max(fcbc_results[lam]['group_fairness_costs'].values())) for lam in lambdas]
        cost_ratios = [float(d/f) if f != 0 else float('inf') for d, f in zip(distance_costs, fairness_costs)]
        
        # Plot total welfare cost
        ax1.plot(lambdas, total_costs, 'o-', label='FCBC', color='green', zorder=2)
        ax1.scatter(lambdas, total_costs, color='green', s=100, zorder=2)
        
        # Plot distance cost
        ax2.plot(lambdas, distance_costs, 'o-', label='FCBC', color='green', zorder=2)
        ax2.scatter(lambdas, distance_costs, color='green', s=100, zorder=2)
        
        # Plot fairness violation
        ax3.plot(lambdas, fairness_costs, 'o-', label='FCBC', color='green', zorder=2)
        ax3.scatter(lambdas, fairness_costs, color='green', s=100, zorder=2)
        
        # Plot cost ratio
        ax4.plot(lambdas, cost_ratios, 'o-', label='FCBC', color='green', zorder=2)
        ax4.scatter(lambdas, cost_ratios, color='green', s=100, zorder=2)
    
    # Set up each subplot
    ax1.set_xlabel('λ')
    ax1.set_ylabel('Maximum Welfare Cost')
    ax1.set_title(f'Comparison of Total Welfare Costs (k={k})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('λ')
    ax2.set_ylabel('Maximum Group Distance Cost')
    ax2.set_title(f'Comparison of Maximum Group Distance Costs (k={k})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('λ')
    ax3.set_ylabel('Maximum Group Fairness Cost')
    ax3.set_title(f'Comparison of Maximum Group Fairness Costs (k={k})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('λ')
    ax4.set_ylabel('Distance Cost / Fairness Violation')
    ax4.set_title(f'Ratio of Maximum Distance Cost to Maximum Fairness Cost (k={k})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # Set y-axis to log scale since ratios can vary by orders of magnitude
    ax4.set_yscale('log')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = cache_dir
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'lambda_sweep_comparison_k{k}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Also save the data
    data = {
        'lambda_values': [float(lam) for lam in lambda_values],
        'welfare_clustering': {
            'total_costs': [float(cost) for cost in total_costs] if wc_results else None,
            'distance_costs': [float(cost) for cost in distance_costs] if wc_results else None,
            'fairness_costs': [float(cost) for cost in fairness_costs] if wc_results else None,
            'cost_ratios': [float(ratio) for ratio in cost_ratios] if wc_results else None
        },
        'socially_fair': {
            'total_costs': [float(sf_results[lam]['max_welfare_cost']) for lam in lambda_values] if sf_results else None,
            'distance_costs': [float(max(sf_results[lam]['group_distance_costs'].values())) for lam in lambda_values] if sf_results else None,
            'fairness_costs': [float(max(sf_results[lam]['group_fairness_costs'].values())) for lam in lambda_values] if sf_results else None,
            'cost_ratios': [float(d/f) if f != 0 else float('inf') for d, f in zip(
                [float(max(sf_results[lam]['group_distance_costs'].values())) for lam in lambda_values] if sf_results else [],
                [float(max(sf_results[lam]['group_fairness_costs'].values())) for lam in lambda_values] if sf_results else []
            )] if sf_results else None
        },
        'fcbc': {
            'total_costs': [float(fcbc_results[lam]['max_welfare_cost']) for lam in lambda_values] if fcbc_results else None,
            'distance_costs': [float(max(fcbc_results[lam]['group_distance_costs'].values())) for lam in lambda_values] if fcbc_results else None,
            'fairness_costs': [float(max(fcbc_results[lam]['group_fairness_costs'].values())) for lam in lambda_values] if fcbc_results else None,
            'cost_ratios': [float(d/f) if f != 0 else float('inf') for d, f in zip(
                [float(max(fcbc_results[lam]['group_distance_costs'].values())) for lam in lambda_values] if fcbc_results else [],
                [float(max(fcbc_results[lam]['group_fairness_costs'].values())) for lam in lambda_values] if fcbc_results else []
            )] if fcbc_results else None
        }
    }
        
    with open(os.path.join(output_dir, f'lambda_sweep_comparison_k{k}.json'), 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print("Available results plotted:")
    if wc_results:
        print("- Welfare Clustering")
    if sf_results:
        print("- Socially Fair")
    if fcbc_results:
        print("- FCBC")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='cache',
                       help='Base directory containing all results')
    parser.add_argument('--k', type=int, default=4,
                       help='Number of clusters')
    parser.add_argument('--output_dir', type=str, default='cache/plots',
                       help='Directory to save plots')
    parser.add_argument('--lambda_start', type=float, default=0.0,
                       help='Starting value for lambda sweep')
    parser.add_argument('--lambda_end', type=float, default=0.05,
                       help='Ending value for lambda sweep')
    parser.add_argument('--lambda_step', type=float, default=0.001,
                       help='Step size for lambda sweep')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Call the plotting function
    plot_lambda_sweep_comparison(
        os.path.join(args.cache_dir, 'cache_synthetic_noisy', 'welfare_clustering_results'),
        args.k,
        args.lambda_start,
        args.lambda_end,
        args.lambda_step,
        args.output_dir
    )

if __name__ == '__main__':
    main() 