import os
import numpy as np
import pandas as pd
import configparser
import ast
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost_with_slack

def load_fcbc_results(cache_dir, k):
    """
    Load FCBC results for a specific k value.
    """
    # Load data matrix and group labels
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Load FCBC results for this k
    fcbc_results_file = os.path.join(cache_dir, "fcbc_results", f"fcbc_k{k}_all_pof.csv")
    fcbc_results = pd.read_csv(fcbc_results_file)
    
    return data_matrix, group_labels, fcbc_results

def load_alpha_beta_from_config(config_file):
    """Load alpha and beta parameters from the config file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Parse alpha and beta from config
    alpha = ast.literal_eval(config['params']['alpha'])
    beta = ast.literal_eval(config['params']['beta'])
    
    return alpha, beta

def evaluate_fcbc_welfare_costs(cache_dir, k, lambda_param=0.5):
    """
    Evaluate welfare costs for FCBC results with varying POF values for a specific k.
    """
    # Load data and results
    data_matrix, group_labels, fcbc_results = load_fcbc_results(cache_dir, k)
    
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
    
    # Process each POF value
    for _, row in fcbc_results.iterrows():
        pof = row['pof']
        
        # Extract centers and assignment from the results
        centers = np.array(eval(row['centers']))  # Convert string representation to array
        assignment_str = row['assignment']
        print(f"\nRaw assignment string: {assignment_str[:200]}...")  # Print first 200 chars
        
        assignment = np.array(eval(assignment_str), dtype=int)  # Ensure integer type for indexing
        
        # Print dimensions and content for debugging
        print(f"\nPOF {pof}:")
        print(f"centers shape: {centers.shape}")
        print(f"assignment shape: {assignment.shape}")
        print(f"assignment min: {assignment.min()}, max: {assignment.max()}")
        print(f"unique assignments: {np.unique(assignment)}")
        print(f"first few assignments: {assignment[:10]}")
        
        # Verify assignment format
        if assignment.max() >= centers.shape[0]:
            raise ValueError(f"Invalid assignment: max value {assignment.max()} >= number of clusters {centers.shape[0]}")
        if assignment.min() < 0:
            raise ValueError(f"Invalid assignment: min value {assignment.min()} < 0")
        
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
            'pof': pof,
            'lambda_param': lambda_param,
            'max_welfare_cost': max_welfare_cost,
            'group_costs': group_costs,
            'fcbc_objective': row['fair_cost'],
            'runtime': row['runtime'],
            'alpha': str(alpha),
            'beta': str(beta)
        }
        
        welfare_results.append(result_row)
    
    return welfare_results

def evaluate_fcbc_welfare_costs_range(cache_dir, k_min, k_max, lambda_param=0.5):
    """
    Evaluate welfare costs for FCBC results for a range of k values.
    """
    print(f"Evaluating FCBC welfare costs for k from {k_min} to {k_max}")
    
    # Initialize list to store all results
    all_results = []
    
    # Evaluate each k value
    for k in range(k_min, k_max + 1):
        try:
            k_results = evaluate_fcbc_welfare_costs(cache_dir, k, lambda_param)
            all_results.extend(k_results)
            
            # Find minimum welfare cost for this k
            k_df = pd.DataFrame(k_results)
            min_cost_row = k_df.loc[k_df['max_welfare_cost'].idxmin()]
            print(f"\nMinimum welfare cost for k={k}: {min_cost_row['max_welfare_cost']:.4f} at POF={min_cost_row['pof']}")
        except Exception as e:
            print(f"Error evaluating k={k}: {str(e)}")
    
    # Convert to DataFrame
    df_welfare = pd.DataFrame(all_results)
    
    # Save results
    results_dir = os.path.join(cache_dir, "welfare_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save combined results
    output_file = os.path.join(results_dir, f"fcbc_all_k{k_min}_to_{k_max}_welfare_costs.csv")
    df_welfare.to_csv(output_file, index=False)
    
    print(f"\nAll results saved to: {output_file}")
    
    return df_welfare

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate welfare costs for FCBC results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, required=True, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, required=True, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight between distance and fairness costs")
    
    args = parser.parse_args()
    
    evaluate_fcbc_welfare_costs_range(args.cache_dir, args.k_min, args.k_max, args.lambda_param)
 