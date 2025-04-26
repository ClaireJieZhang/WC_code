import os
import numpy as np
import pandas as pd
import json
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost

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

def evaluate_welfare_clustering_costs(cache_dir, k, lambda_param):
    """Evaluate Welfare Clustering costs for different lambda values."""
    # Load data
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Load results
    results = load_welfare_clustering_results(cache_dir, k)
    centers = np.array(results['centers'])
    assignment = results['assignment']
    
    # Convert binary assignment matrix to cluster indices
    if len(assignment.shape) > 1:  # If it's a binary matrix
        assignment = np.argmax(assignment, axis=1)
    
    # Print dimensions for debugging
    print(f"\nData dimensions:")
    print(f"data_matrix shape: {data_matrix.shape}")
    print(f"group_labels shape: {group_labels.shape}")
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
        'distance_costs': welfare_metrics.get('distance_costs', None),
        'fairness_costs': welfare_metrics.get('fairness_costs', None),
        'runtime': None
    }
    
    # Convert to DataFrame
    df_welfare = pd.DataFrame([result_row])
    
    # Save results
    results_dir = os.path.join(cache_dir, "welfare_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    output_file = os.path.join(results_dir, f"welfare_clustering_k{k}_welfare_costs.csv")
    df_welfare.to_csv(output_file, index=False)
    
    print(f"\nResults for k={k}:")
    print(f"Welfare cost: {df_welfare['max_welfare_cost'].iloc[0]:.4f}")
    print(f"\nDetailed results saved to: {output_file}")
    
    return df_welfare

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate welfare costs for Welfare Clustering results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--lambda_param", type=float, required=True, help="Weight between distance and fairness costs")
    
    args = parser.parse_args()
    
    evaluate_welfare_clustering_costs(args.cache_dir, args.k, args.lambda_param) 