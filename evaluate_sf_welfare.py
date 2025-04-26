import os
import numpy as np
import pandas as pd
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost

def load_sf_results(cache_dir, k):
    """
    Load SF clustering results for a specific k value.
    """
    # Load data matrix and group labels
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Load SF results for this k
    sf_results_file = os.path.join(cache_dir, "sf_results", f"sf_k{k}.csv")
    sf_results = pd.read_csv(sf_results_file)
    
    return data_matrix, group_labels, sf_results

def evaluate_sf_welfare_costs(cache_dir, k, lambda_param=0.5):
    """
    Evaluate welfare costs for SF clustering results.
    """
    # Load data and results
    data_matrix, group_labels, sf_results = pd.read_csv(sf_results_file)
    
    # Print dimensions for debugging
    print(f"\nData dimensions:")
    print(f"data_matrix shape: {data_matrix.shape}")
    print(f"group_labels shape: {group_labels.shape}")
    
    # Initialize list to store results
    welfare_results = []
    
    # Process each row (there should be only one for SF)
    for _, row in sf_results.iterrows():
        # Extract centers and assignment from the results
        centers = np.array(eval(row['centers']))  # Convert string representation to array
        assignment = np.array(eval(row['assignment']), dtype=int)  # Ensure integer type for indexing
        
        # Print dimensions for debugging
        print(f"\nSF Clustering:")
        print(f"centers shape: {centers.shape}")
        print(f"assignment shape: {assignment.shape}")
        print(f"assignment min: {assignment.min()}, max: {assignment.max()}")
        
        # Calculate welfare cost
        welfare_metrics = evaluate_welfare_cost(
            centers=centers,
            assignment=assignment,
            points=data_matrix,
            group_labels=group_labels,
            lambda_param=lambda_param
        )
        
        # Create result row
        result_row = {
            'k': k,
            'lambda_param': lambda_param,
            'max_welfare_cost': welfare_metrics['max_welfare_cost'],
            'group_costs': welfare_metrics['group_costs'],
            'sf_objective': row['objective'],
            'runtime': row['runtime']
        }
        
        welfare_results.append(result_row)
    
    # Convert to DataFrame
    df_welfare = pd.DataFrame(welfare_results)
    
    # Save results
    results_dir = os.path.join(cache_dir, "welfare_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    output_file = os.path.join(results_dir, f"sf_k{k}_welfare_costs.csv")
    df_welfare.to_csv(output_file, index=False)
    
    print(f"\nResults for k={k}:")
    print(f"Welfare cost: {df_welfare['max_welfare_cost'].iloc[0]:.4f}")
    print(f"\nDetailed results saved to: {output_file}")
    
    return df_welfare

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate welfare costs for SF clustering results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight between distance and fairness costs")
    
    args = parser.parse_args()
    
    evaluate_sf_welfare_costs(args.cache_dir, args.k, args.lambda_param) 