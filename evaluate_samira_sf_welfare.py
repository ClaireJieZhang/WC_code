import os
import numpy as np
import pandas as pd
import pickle
import configparser
import ast
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost_with_slack

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
    # Load data matrix and group labels
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Find the most recent results file
    results_dir = os.path.join(cache_dir, "socially_fair_results")
    result_files = [f for f in os.listdir(results_dir) if f.startswith("sf_results_") and f.endswith(".pkl")]
    if not result_files:
        raise FileNotFoundError(f"No Samira's SF results found in {results_dir}")
    
    # Sort by timestamp in filename and get the most recent
    latest_file = sorted(result_files)[-1]
    with open(os.path.join(results_dir, latest_file), 'rb') as f:
        results = pickle.load(f)
    
    # Get results for this k value
    k_results = results['results'][str(k)]
    
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
    
    # Process standard and fair clustering results
    for method in ['standard', 'fair']:
        if method not in k_results:
            continue
            
        method_results = k_results[method]
        centers = np.array(method_results['centers'])
        assignment = np.array(method_results['assignment'])
        
        # Print dimensions for debugging
        print(f"\n{method.capitalize()} clustering:")
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
                'method': method,
                'lambda_param': lambda_param,
                'max_welfare_cost': max_welfare_cost,
                'group_costs': group_costs,
                'runtime': method_results.get('runtime', None),
                'alpha': str(alpha),
                'beta': str(beta)
            }
            
            welfare_results.append(result_row)
        except Exception as e:
            print(f"Error calculating welfare cost for {method} clustering: {str(e)}")
            continue
    
    return welfare_results

def evaluate_samira_sf_costs_range(cache_dir, k_min, k_max, lambda_param=0.5):
    """
    Evaluate welfare costs for Samira's SF results for a range of k values.
    """
    print(f"Evaluating Samira's SF welfare costs for k from {k_min} to {k_max}")
    
    # Initialize list to store all results
    all_results = []
    
    # Evaluate each k value
    for k in range(k_min, k_max + 1):
        try:
            k_results = evaluate_samira_sf_costs(cache_dir, k, lambda_param)
            all_results.extend(k_results)
            
            # Find minimum welfare cost for this k
            k_df = pd.DataFrame(k_results)
            min_cost_row = k_df.loc[k_df['max_welfare_cost'].idxmin()]
            print(f"\nMinimum welfare cost for k={k}: {min_cost_row['max_welfare_cost']:.4f} with {min_cost_row['method']} clustering")
        except Exception as e:
            print(f"Error evaluating k={k}: {str(e)}")
    
    # Convert to DataFrame
    df_welfare = pd.DataFrame(all_results)
    
    # Save results
    results_dir = os.path.join(cache_dir, "welfare_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save combined results
    output_file = os.path.join(results_dir, f"samira_sf_all_k{k_min}_to_{k_max}_welfare_costs.csv")
    df_welfare.to_csv(output_file, index=False)
    
    print(f"\nAll results saved to: {output_file}")
    
    return df_welfare

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate welfare costs for Samira's SF results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, required=True, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, required=True, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight between distance and fairness costs")
    
    args = parser.parse_args()
    
    evaluate_samira_sf_costs_range(args.cache_dir, args.k_min, args.k_max, args.lambda_param) 