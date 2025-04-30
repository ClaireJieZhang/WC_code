import os
import numpy as np
import pandas as pd
import pickle
import configparser
import ast
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost_with_slack
import json

def parse_numpy_array(array_str):
    """
    Parse a numpy array string into a numpy array.
    """
    # Remove 'array(' and ')' from the string
    array_str = array_str.strip('array()')
    
    # Use regex to find all numbers, including scientific notation
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    numbers = re.findall(pattern, array_str)
    
    # Convert all numbers to float
    numbers = [float(x) for x in numbers]
    
    # Reshape into 2D array if we have multiple rows
    if '],' in array_str:
        # Count number of rows by counting '],'
        num_rows = array_str.count('],') + 1
        # Reshape the array
        return np.array(numbers).reshape(num_rows, -1)
    else:
        return np.array(numbers)

def load_samira_sf_results(cache_dir, k):
    """
    Load Samira's SF results for a specific k value.
    """
    # Load data matrix and group labels from the data directory
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Find the results file in the sf_results directory
    results_dir = os.path.join(cache_dir, "sf_results")
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Look for detailed_results_k{k}.json file
    json_file = os.path.join(results_dir, f"detailed_results_k{k}.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"No Samira's SF results found for k={k} in {results_dir}")
    
    # Load the JSON file
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Extract the centers and assignment from the JSON structure
    # The structure is different from the PKL format, so we need to adapt
    if 'data' not in results:
        raise ValueError(f"Results file for k={k} does not contain 'data' field")
    
    data = results['data']
    if 'centers' not in data or 'assignment' not in data:
        raise ValueError(f"Results file for k={k} does not contain required 'centers' and 'assignment' fields")
    
    # Create a structure similar to what the original code expected
    k_results = {
        'fair': {
            'centers': data['centers'],
            'assignment': data['assignment'],
            'runtime': data.get('metrics', {}).get('runtime', None)
        }
    }
    
    return data_matrix, group_labels, k_results

def load_alpha_beta_from_config(config_file):
    """Load alpha and beta parameters from the config file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Parse alpha and beta from config
    alpha = ast.literal_eval(config['params']['alpha'])
    beta = ast.literal_eval(config['params']['beta'])
    
    return alpha, beta

def evaluate_samira_sf_costs(cache_dir, k, lambda_param=0.5):
    """
    Evaluate welfare costs for Samira's SF results for a specific k.
    """
    # Load data and results
    data_matrix, group_labels, k_results = load_samira_sf_results(cache_dir, k)
    
    # Load alpha and beta from config
    config_file = "configs/welfare_clustering_config.ini"  # Using same config as Welfare Clustering
    alpha, beta = load_alpha_beta_from_config(config_file)
    
    # Print dimensions for debugging
    print(f"\nData dimensions for k={k}:")
    print(f"data_matrix shape: {data_matrix.shape}")
    print(f"group_labels shape: {group_labels.shape}")
    print(f"\nAlpha: {alpha}")
    print(f"Beta: {beta}")
    
    # Initialize list to store results
    welfare_results = []
    
    # Only process 'fair' clustering results
    if 'fair' not in k_results:
        print(f"No 'fair' results found for k={k}")
        return welfare_results
    method_results = k_results['fair']
    centers = np.array(method_results['centers'])
    assignment = np.array(method_results['assignment'])
    
    # Print dimensions for debugging
    print(f"\nFair clustering:")
    print(f"centers shape: {centers.shape}")
    print(f"assignment shape: {assignment.shape}")
    print(f"assignment min: {assignment.min()}, max: {assignment.max()}")
    
    # Ensure assignment indices are valid
    if assignment.max() >= centers.shape[0]:
        print(f"Warning: Invalid assignment indices found. Max index {assignment.max()} >= number of clusters {centers.shape[0]}")
        # Clip assignment values to valid range
        assignment = np.clip(assignment, 0, centers.shape[0] - 1)
        print(f"After clipping: assignment min: {assignment.min()}, max: {assignment.max()}")
    
    # Calculate welfare cost with slack
    try:
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
            'method': 'fair',
            'lambda_param': lambda_param,
            'max_welfare_cost': max_welfare_cost,
            'group_costs': group_costs,
            'group_distance_costs': group_distance_costs,
            'runtime': method_results.get('runtime', None),
            'alpha': str(alpha),
            'beta': str(beta),
            'expected_proportions': cluster_stats['expected_proportions'],
            'cluster_stats': cluster_stats['clusters']
        }
        
        welfare_results.append(result_row)
        
        # Print group proportions for each cluster
        print(f"\nGroup proportions for fair clustering (k={k}):")
        print("Expected proportions:", cluster_stats['expected_proportions'])
        print("\nActual proportions by cluster:")
        for cluster_id, stats in cluster_stats['clusters'].items():
            print(f"Cluster {cluster_id} (size={stats['size']}):")
            print("  Proportions:", stats['group_proportions'])
            print("  Violations:", stats['violations'])
            
    except Exception as e:
        print(f"Error calculating welfare cost for fair clustering: {str(e)}")
    
    return welfare_results

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

def evaluate_samira_sf_costs_range(cache_dir, k_min, k_max, lambda_param=0.5):
    """
    Evaluate welfare costs for Samira's SF results for a range of k values.
    """
    print(f"Evaluating Samira's SF welfare costs for k from {k_min} to {k_max} with lambda={lambda_param}")
    
    # Initialize list to store all results
    all_results = []
    
    # Evaluate each k value
    for k in range(k_min, k_max + 1):
        try:
            k_results = evaluate_samira_sf_costs(cache_dir, k, lambda_param)
            all_results.extend(k_results)
            
            # Find minimum welfare cost for this k
            if k_results:  # Only process if we have results
                k_df = pd.DataFrame(k_results)
                min_cost_row = k_df.loc[k_df['max_welfare_cost'].idxmin()]
                print(f"\nMinimum welfare cost for k={k}: {min_cost_row['max_welfare_cost']:.4f} with {min_cost_row['method']} clustering")
        except Exception as e:
            print(f"Error evaluating k={k}: {str(e)}")
            continue
    
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
    output_file = os.path.join(results_dir, f"samira_sf_all_k{k_min}_to_{k_max}_{lambda_str}_detailed.json")
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary metrics to CSV
    summary_results = []
    for result in all_results:
        summary_row = {
            'k': int(result['k']),
            'method': result['method'],
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
    summary_file = os.path.join(results_dir, f"samira_sf_all_k{k_min}_to_{k_max}_{lambda_str}_summary.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"Summary results saved to: {summary_file}")
    
    return df_summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate welfare costs for Samira's SF results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, required=True, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, required=True, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, required=True, help="Lambda parameter to evaluate")
    
    args = parser.parse_args()
    
    evaluate_samira_sf_costs_range(args.cache_dir, args.k_min, args.k_max, args.lambda_param) 