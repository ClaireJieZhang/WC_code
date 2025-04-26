import os
import sys
import argparse
import numpy as np
import pandas as pd
import configparser
import json
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import from evaluation_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_welfare_clustering_for_k(cache_dir, config_file, k, lambda_param=0.5):
    """
    Run the Welfare_Clustering experiment for a specific k value.
    
    Args:
        cache_dir: str - Directory containing cached data
        config_file: str - Path to config file for Welfare_Clustering
        k: int - Number of clusters
        lambda_param: float - Lambda parameter for Welfare_Clustering
        
    Returns:
        dict - Results for this k value
    """
    print(f"Running Welfare_Clustering experiment for k={k}")
    
    # Load the cached data
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    df_clean = pd.read_csv(os.path.join(cache_dir, "df_clean.csv"))
    
    # Read group names
    with open(os.path.join(cache_dir, "group_names.txt"), "r") as f:
        group_names = [line.strip() for line in f.readlines()]
    
    # Update the config file with the current k value
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Set the number of clusters in the config
    if not config.has_section('clustering'):
        config.add_section('clustering')
    config['clustering']['num_clusters'] = str(k)
    
    # Ensure params section exists
    if not config.has_section('params'):
        config.add_section('params')
    
    # Set default alpha and beta if not present
    if 'alpha' not in config['params']:
        config['params']['alpha'] = str({0: lambda_param, 1: lambda_param})
    if 'beta' not in config['params']:
        config['params']['beta'] = str({0: lambda_param, 1: lambda_param})
    
    # Write the updated config to a temporary file
    temp_config_file = os.path.join(cache_dir, f"temp_config_k{k}.ini")
    with open(temp_config_file, 'w') as f:
        config.write(f)
    
    # Import the run_full_pipeline_with_loaded_data function
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Welfare_Clustering"))
    from main_wc import run_full_pipeline_with_loaded_data
    
    # Run the experiment
    results_dir = os.path.join(cache_dir, "welfare_clustering_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the experiment
    results = run_full_pipeline_with_loaded_data(
        df=df_clean,
        svar_all=group_labels,
        group_names=group_names,
        config_file=temp_config_file,
        dataset_name="cached_data",
        lambda_param=lambda_param
    )
    
    # Helper function to convert numpy arrays to lists
    def to_list(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    # Create a structured results dictionary
    structured_results = {
        'pipeline': 'Welfare_Clustering',
        'dataset': 'cached_data',
        'timestamp': pd.Timestamp.now().isoformat(),
        'data': {
            'points': data_matrix.tolist(),
            'group_labels': group_labels.tolist(),
            'group_names': group_names
        },
        'results': {
            str(k): {
                'standard': None,  # Welfare Clustering only does fair version
                'fair': {
                    'centers': to_list(results['centers']),
                    'assignment': to_list(results['assignment']),
                    'metrics': {
                        'objective': float(results.get('objective', 0.0)),
                        'runtime': results.get('runtime', None),
                        'group_costs': to_list(results.get('group_costs', None)),
                        'proportions_normalized': to_list(results.get('proportions_normalized', []))
                    }
                }
            }
        }
    }
    
    # Save detailed results to JSON
    detailed_file = os.path.join(results_dir, f"detailed_results_k{k}.json")
    with open(detailed_file, 'w') as f:
        json.dump(structured_results, f, indent=2)
    print(f"Detailed results for k={k} saved to: {detailed_file}")
    
    # Create a row for the results DataFrame (without the large arrays)
    result_row = {
        'k': k,
        'lambda_param': lambda_param,
        'objective': float(results.get('objective', 0.0)),
        'runtime': results.get('runtime', None),
        'num_points': len(data_matrix),
        'timestamp': pd.Timestamp.now().isoformat(),
        'alpha': config['params']['alpha'],
        'beta': config['params']['beta']
    }
    
    # Add group-specific metrics if available
    if 'group_costs' in results:
        for i, cost in enumerate(results['group_costs']):
            result_row[f'group_{i}_cost'] = float(cost)
    
    # Save individual results for this k
    df_k = pd.DataFrame([result_row])
    k_file = os.path.join(results_dir, f"welfare_clustering_k{k}.csv")
    df_k.to_csv(k_file, index=False)
    print(f"Results for k={k} saved to: {k_file}")
    
    # Clean up the temporary config file
    os.remove(temp_config_file)
    
    print(f"Welfare_Clustering experiment for k={k} completed")
    return result_row

def run_welfare_clustering_experiment(cache_dir, config_file, k_min=4, k_max=8, lambda_param=0.5):
    """
    Run the Welfare_Clustering experiment for multiple k values.
    
    Args:
        cache_dir: str - Directory containing cached data
        config_file: str - Path to config file for Welfare_Clustering
        k_min: int - Minimum number of clusters
        k_max: int - Maximum number of clusters
        lambda_param: float - Lambda parameter for Welfare_Clustering
    """
    print(f"Running Welfare_Clustering experiment with k from {k_min} to {k_max}")
    
    # Create a DataFrame to store all results
    all_results = []
    
    for k in range(k_min, k_max + 1):
        # Run the experiment for this k
        result_row = run_welfare_clustering_for_k(
            cache_dir=cache_dir,
            config_file=config_file,
            k=k,
            lambda_param=lambda_param
        )
        all_results.append(result_row)
    
    # Convert to DataFrame
    df_all = pd.DataFrame(all_results)
    
    # Save combined results
    results_dir = os.path.join(cache_dir, "welfare_clustering_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save combined results
    combined_file = os.path.join(results_dir, f"welfare_clustering_all_k{k_min}_to_{k_max}.csv")
    df_all.to_csv(combined_file, index=False)
    print(f"Combined results for all k values saved to: {combined_file}")

def main():
    parser = argparse.ArgumentParser(description="Run Welfare_Clustering experiment with cached data")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--config_file", type=str, default="configs/welfare_clustering_config.ini", help="Path to config file for Welfare_Clustering")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, default=8, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Lambda parameter for Welfare_Clustering")
    args = parser.parse_args()
    
    run_welfare_clustering_experiment(
        cache_dir=args.cache_dir,
        config_file=args.config_file,
        k_min=args.k_min,
        k_max=args.k_max,
        lambda_param=args.lambda_param
    )
    
    print("Welfare_Clustering experiment completed!")

if __name__ == "__main__":
    main() 