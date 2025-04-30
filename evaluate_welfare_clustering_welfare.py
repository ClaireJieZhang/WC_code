import os
import numpy as np
import pandas as pd
import json
import configparser
import ast
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost_with_slack

def load_welfare_clustering_results(cache_dir, k, lambda_param):
    """Load Welfare Clustering results from the JSON file with matching lambda."""
    results_dir = os.path.join(cache_dir, "welfare_clustering_results")
    lambda_str = f"lambda_{str(lambda_param).replace('.', '_')}"
    json_file = os.path.join(results_dir, f"detailed_results_k{k}_{lambda_str}.json")
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"No results file found for k={k} and lambda={lambda_param}")
        
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Verify lambda value matches
    if abs(results.get('lambda_param', 0) - lambda_param) > 1e-6:
        raise ValueError(f"Lambda mismatch in results file. Expected {lambda_param}, found {results.get('lambda_param')}")
        
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

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    else:
        return obj

def evaluate_welfare_clustering_costs(cache_dir, k, lambda_param):
    """Evaluate Welfare Clustering costs for a specific k value."""
    # Load data
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Load alpha and beta from config
    config_file = "configs/welfare_clustering_config.ini"
    alpha, beta = load_alpha_beta_from_config(config_file)
    
    # Load results
    results = load_welfare_clustering_results(cache_dir, k, lambda_param)
    centers = np.array(results['centers'])
    assignment = results['assignment']
    
    # Convert binary assignment matrix to indices if needed
    if len(assignment.shape) > 1:  # If it's a binary matrix
        assignment = np.argmax(assignment, axis=1)
    
    # Print dimensions for debugging
    print(f"\nData dimensions for k={k}, lambda={lambda_param}:")
    print(f"data_matrix shape: {data_matrix.shape}")
    print(f"group_labels shape: {group_labels.shape}")
    print(f"centers shape: {centers.shape}")
    print(f"assignment shape: {assignment.shape}")
    print(f"assignment min: {assignment.min()}, max: {assignment.max()}")
    print(f"\nAlpha: {alpha}")
    print(f"Beta: {beta}")
    
    # Calculate welfare cost with slack
    max_welfare_cost, group_costs, cluster_stats, group_distance_costs = evaluate_welfare_cost_with_slack(
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
        'group_distance_costs': group_distance_costs,
        'alpha': str(alpha),
        'beta': str(beta),
        'runtime': None,
        'expected_proportions': cluster_stats['expected_proportions'],
        'cluster_stats': cluster_stats['clusters']
    }
    
    return result_row

def evaluate_welfare_clustering_costs_range(cache_dir, k_min, k_max, lambda_param):
    """Evaluate Welfare Clustering costs for a range of k values."""
    print(f"Evaluating Welfare Clustering costs for k from {k_min} to {k_max} with lambda={lambda_param}")
    
    # Initialize list to store results
    all_results = []
    
    # Evaluate each k value
    for k in range(k_min, k_max + 1):
        try:
            result_row = evaluate_welfare_clustering_costs(cache_dir, k, lambda_param)
            all_results.append(result_row)
            print(f"Welfare cost for k={k}: {result_row['max_welfare_cost']:.4f}")
            
            # Print group proportions
            print("\nCluster statistics:")
            print("Expected proportions:", result_row['expected_proportions'])
            for cluster_id, stats in result_row['cluster_stats'].items():
                print(f"\nCluster {cluster_id} (size={stats['size']}):")
                print("  Proportions:", stats['group_proportions'])
                print("  Violations:", stats['violations'])
                
        except Exception as e:
            print(f"Error evaluating k={k}: {str(e)}")
    
    # Create welfare_evaluation directory if it doesn't exist
    results_dir = os.path.join(cache_dir, "welfare_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    if not all_results:
        print("No results were successfully evaluated")
        return None
    
    # Convert numpy types to Python native types for JSON serialization
    serializable_results = convert_to_serializable(all_results)
    
    # Include lambda in output filenames
    lambda_str = f"lambda_{str(lambda_param).replace('.', '_')}"
    
    # Save full results with cluster statistics
    output_file = os.path.join(results_dir, f"welfare_clustering_all_k{k_min}_to_{k_max}_{lambda_str}_detailed.json")
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary metrics to CSV
    summary_results = []
    for result in all_results:
        summary_row = {
            'k': int(result['k']),
            'lambda_param': float(result['lambda_param']),
            'max_welfare_cost': float(result['max_welfare_cost']),
            'runtime': float(result['runtime']) if result['runtime'] is not None else None
        }
        # Add group costs
        for group, cost in result['group_costs'].items():
            summary_row[f'group_{group}_cost'] = float(cost)
        # Add group distance costs
        for group, dist_cost in result['group_distance_costs'].items():
            summary_row[f'group_{group}_distance_cost'] = float(dist_cost)
        
        # Add average costs per group
        for group in result['group_costs'].keys():
            group_costs = []
            for cluster_id, stats in result['cluster_stats'].items():
                if group in stats['group_proportions']:
                    cluster_size = stats['size']
                    group_prop = stats['group_proportions'][group]
                    group_costs.append(group_prop * cluster_size)
            avg_cost = sum(group_costs) / len(group_costs) if group_costs else 0
            summary_row[f'group_{group}_avg_cost'] = float(avg_cost)
        
        summary_results.append(summary_row)
    
    df_summary = pd.DataFrame(summary_results)
    summary_file = os.path.join(results_dir, f"welfare_clustering_all_k{k_min}_to_{k_max}_{lambda_str}_summary.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"Summary results saved to: {summary_file}")
    
    return df_summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate welfare costs for Welfare Clustering results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, required=True, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, required=True, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, required=True, help="Lambda parameter to evaluate")
    
    args = parser.parse_args()
    
    evaluate_welfare_clustering_costs_range(args.cache_dir, args.k_min, args.k_max, args.lambda_param) 