import os
import numpy as np
import pandas as pd
import json
import configparser
import ast
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost_with_slack

def load_welfare_clustering_results(cache_dir, k):
    """Load Welfare Clustering results from the most recent JSON file."""
    results_dir = os.path.join(cache_dir, "welfare_clustering_results")
    json_file = os.path.join(results_dir, f"detailed_results_k{k}.json")
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"No results file found for k={k}")
        
    with open(json_file, 'r') as f:
        results = json.load(f)
        
    # The results are nested under results[str(k)]['fair']
    k_str = str(k)
    if k_str not in results['results']:
        raise ValueError(f"Results file for k={k} does not contain results for this k value")
        
    fair_results = results['results'][k_str]['fair']
    if 'centers' not in fair_results or 'assignment' not in fair_results:
        raise ValueError(f"Results file for k={k} does not contain required 'centers' and 'assignment' fields")
        
    return {
        'centers': fair_results['centers'],
        'assignment': np.array(fair_results['assignment'])
    }

def load_alpha_beta_from_config(config_file):
    """Load alpha and beta parameters from the config file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Parse alpha and beta from config
    alpha = ast.literal_eval(config['params']['alpha'])
    beta = ast.literal_eval(config['params']['beta'])
    
    return alpha, beta

def evaluate_welfare_clustering_costs(cache_dir, k, lambda_param):
    """Evaluate Welfare Clustering costs for a specific k value."""
    # Load data
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Load alpha and beta from config
    config_file = "configs/welfare_clustering_config.ini"
    alpha, beta = load_alpha_beta_from_config(config_file)
    
    # Load results
    results = load_welfare_clustering_results(cache_dir, k)
    centers = np.array(results['centers'])
    assignment = results['assignment']
    
    # Convert binary assignment matrix to cluster indices
    if len(assignment.shape) > 1:  # If it's a binary matrix
        assignment = np.argmax(assignment, axis=1)
    
    # Print dimensions for debugging
    print(f"\nData dimensions for k={k}:")
    print(f"data_matrix shape: {data_matrix.shape}")
    print(f"group_labels shape: {group_labels.shape}")
    print(f"centers shape: {centers.shape}")
    print(f"assignment shape: {assignment.shape}")
    print(f"assignment min: {assignment.min()}, max: {assignment.max()}")
    print(f"\nAlpha: {alpha}")
    print(f"Beta: {beta}")
    
    # Calculate welfare cost with slack
    max_welfare_cost, group_costs = evaluate_welfare_cost_with_slack(
        centers=centers,
        assignment=assignment,
        points=data_matrix,
        group_labels=group_labels,
        lambda_param=lambda_param,
        alpha=alpha,
        beta=beta
    )
    
    # Create result row
    result_row = {
        'k': k,
        'lambda_param': lambda_param,
        'max_welfare_cost': max_welfare_cost,
        'group_costs': group_costs,
        'alpha': str(alpha),
        'beta': str(beta),
        'runtime': None
    }
    
    return result_row

def evaluate_welfare_clustering_costs_range(cache_dir, k_min, k_max, lambda_param):
    """Evaluate Welfare Clustering costs for a range of k values."""
    print(f"Evaluating Welfare Clustering costs for k from {k_min} to {k_max}")
    
    # Initialize list to store results
    all_results = []
    
    # Evaluate each k value
    for k in range(k_min, k_max + 1):
        try:
            result_row = evaluate_welfare_clustering_costs(cache_dir, k, lambda_param)
            all_results.append(result_row)
            print(f"Welfare cost for k={k}: {result_row['max_welfare_cost']:.4f}")
        except Exception as e:
            print(f"Error evaluating k={k}: {str(e)}")
    
    # Convert to DataFrame
    df_welfare = pd.DataFrame(all_results)
    
    # Save results
    results_dir = os.path.join(cache_dir, "welfare_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save combined results
    output_file = os.path.join(results_dir, f"welfare_clustering_all_k{k_min}_to_{k_max}_welfare_costs.csv")
    df_welfare.to_csv(output_file, index=False)
    
    print(f"\nAll results saved to: {output_file}")
    
    return df_welfare

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate welfare costs for Welfare Clustering results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, required=True, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, required=True, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, required=True, help="Weight between distance and fairness costs")
    
    args = parser.parse_args()
    
    evaluate_welfare_clustering_costs_range(args.cache_dir, args.k_min, args.k_max, args.lambda_param) 