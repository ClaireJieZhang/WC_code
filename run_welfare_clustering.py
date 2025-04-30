import os
import sys
import argparse
import numpy as np
import pandas as pd
import configparser
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from evaluation_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_lambda_str(lambda_param):
    """Convert lambda parameter to string format for filenames"""
    return f"lambda_{'{:.3f}'.format(lambda_param).replace('.', '_')}"

def run_welfare_clustering_for_k(cache_dir, config_file, k, lambda_param=0.5, init_method="random", normalize_data_flag=True, upweight=1.0):
    """
    Run the Welfare_Clustering experiment for a specific k value.
    
    Args:
        cache_dir: str - Directory containing cached data
        config_file: str - Path to config file
        k: int - Number of clusters
        lambda_param: float - Lambda parameter for fairness
        init_method: str - Initialization method
        normalize_data_flag: bool - Whether to normalize data (default: True)
        upweight: float - Factor to upweight fairness violations (default: 1.0)
    """
    print(f"Running Welfare_Clustering experiment for k={k} with init_method={init_method} and lambda={lambda_param}")
    print(f"Data normalization is {'enabled' if normalize_data_flag else 'disabled'}")
    print(f"Fairness violation upweight factor: {upweight}")
    
    # Load cached data
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    print("\nDEBUG - Initial data loading:")
    print(f"Initial data_matrix shape: {data_matrix.shape}")
    print(f"Initial data_matrix range: {np.min(data_matrix):.3f} to {np.max(data_matrix):.3f}")
    print("First few rows of initial data_matrix:")
    print(data_matrix[:2])
    
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    df_clean = pd.read_csv(os.path.join(cache_dir, "df_clean.csv"))
    
    # Read group names
    with open(os.path.join(cache_dir, "group_names.txt"), "r") as f:
        group_names = [line.strip() for line in f.readlines()]
    
    # Update config
    config = configparser.ConfigParser()
    config.read(config_file)
    if not config.has_section('clustering'):
        config.add_section('clustering')
    config['clustering']['num_clusters'] = str(k)
    if not config.has_section('params'):
        config.add_section('params')
    if 'alpha' not in config['params']:
        config['params']['alpha'] = str({0: lambda_param, 1: lambda_param})
    if 'beta' not in config['params']:
        config['params']['beta'] = str({0: lambda_param, 1: lambda_param})
    
    # Add normalization flag to config
    if not config.has_section('data'):
        config.add_section('data')
    config['data']['normalize'] = str(normalize_data_flag)
    
    temp_config_file = os.path.join(cache_dir, f"temp_config_k{k}.ini")
    with open(temp_config_file, 'w') as f:
        config.write(f)
    
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Welfare_Clustering"))
    from main_wc import run_full_pipeline_with_loaded_data

    results_dir = os.path.join(cache_dir, "welfare_clustering_results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nDEBUG - Before running pipeline:")
    print(f"data_matrix shape: {data_matrix.shape}")
    print(f"data_matrix range: {np.min(data_matrix):.3f} to {np.max(data_matrix):.3f}")
    
    results = run_full_pipeline_with_loaded_data(
        df=df_clean,
        svar_all=group_labels,
        group_names=group_names,
        config_file=temp_config_file,
        dataset_name="cached_data",
        lambda_param=lambda_param,
        init_method=init_method,
        normalize_data_flag=normalize_data_flag,  # Pass the flag to the pipeline
        upweight=upweight  # Pass the upweight factor
    )
    
    print("\nDEBUG - After pipeline completion:")
    print(f"data_matrix range: {np.min(data_matrix):.3f} to {np.max(data_matrix):.3f}")
    
    # Debug: Print original results from pipeline
    print("\nDEBUG - Original results from pipeline:")
    print(f"Centers shape: {results['centers'].shape}")
    print(f"Centers min/max: {np.min(results['centers']):.3f}/{np.max(results['centers']):.3f}")
    print(f"Assignment shape: {results['assignment'].shape}")
    print(f"Assignment min/max: {np.min(results['assignment']):.3f}/{np.max(results['assignment']):.3f}")
    print("\nFirst few rows of centers:")
    print(results['centers'][:2])
    print("\nFirst few rows of assignment:")
    print(results['assignment'][:2])
    
    # Check if pipeline returned normalized data
    if 'normalized_data' in results:
        print("\nDEBUG - Pipeline returned normalized data:")
        print(f"Normalized data shape: {results['normalized_data'].shape}")
        print(f"Normalized data range: {np.min(results['normalized_data']):.3f} to {np.max(results['normalized_data']):.3f}")
    
    # Helper function to convert numpy arrays to lists
    def to_list(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    # Debug: Print data after to_list conversion
    centers_list = to_list(results['centers'])
    assignment_list = to_list(results['assignment'])
    print("\nDEBUG - After to_list conversion:")
    print(f"Centers type: {type(centers_list)}")
    print(f"Centers first few rows:")
    print(centers_list[:2])
    print(f"Assignment type: {type(assignment_list)}")
    print(f"Assignment first few rows:")
    print(assignment_list[:2])
    
    structured_results = {
        'pipeline': 'Welfare_Clustering',
        'dataset': 'cached_data',
        'timestamp': pd.Timestamp.now().isoformat(),
        'init_method': init_method,
        'lambda_param': lambda_param,
        'data': {
            'points': data_matrix.tolist(),
            'group_labels': group_labels.tolist(),
            'group_names': group_names
        },
        'results': {
            str(k): {
                'standard': None,
                'fair': {
                    'centers': centers_list,
                    'assignment': assignment_list,
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
    
    # Debug: Print data in structured_results before JSON conversion
    print("\nDEBUG - Data in structured_results before JSON:")
    print(f"Centers type: {type(structured_results['results'][str(k)]['fair']['centers'])}")
    print(f"Centers first few rows:")
    print(structured_results['results'][str(k)]['fair']['centers'][:2])
    print(f"Assignment type: {type(structured_results['results'][str(k)]['fair']['assignment'])}")
    print(f"Assignment first few rows:")
    print(structured_results['results'][str(k)]['fair']['assignment'][:2])
    
    # Include lambda in filenames with standardized formatting
    lambda_str = get_lambda_str(lambda_param)
    detailed_file = os.path.join(results_dir, f"detailed_results_k{k}_{lambda_str}.json")
    with open(detailed_file, 'w') as f:
        json.dump(structured_results, f, indent=2)
    print(f"Detailed results for k={k} saved to: {detailed_file}")
    
    # Debug: Immediately read back and check values
    print("\nDEBUG - After reading back from JSON:")
    with open(detailed_file, 'r') as f:
        loaded_results = json.load(f)
    
    loaded_centers = np.array(loaded_results['results'][str(k)]['fair']['centers'])
    loaded_assignment = np.array(loaded_results['results'][str(k)]['fair']['assignment'])
    print(f"Loaded centers shape: {loaded_centers.shape}")
    print(f"Loaded centers min/max: {np.min(loaded_centers):.3f}/{np.max(loaded_centers):.3f}")
    print("Loaded centers first few rows:")
    print(loaded_centers[:2])
    print(f"\nLoaded assignment shape: {loaded_assignment.shape}")
    print(f"Loaded assignment min/max: {np.min(loaded_assignment):.3f}/{np.max(loaded_assignment):.3f}")
    print("Loaded assignment first few rows:")
    print(loaded_assignment[:2])
    
    # Debug: Compare data ranges
    print("\nDEBUG - Data ranges comparison:")
    print(f"Original data matrix range: {np.min(data_matrix):.3f} to {np.max(data_matrix):.3f}")
    print(f"Original centers range: {np.min(results['centers']):.3f} to {np.max(results['centers']):.3f}")
    print(f"Loaded centers range: {np.min(loaded_centers):.3f} to {np.max(loaded_centers):.3f}")
    
    # Immediately read from the saved JSON file and create a verification plot
    # print("Verifying saved results by reading from JSON and creating a plot...")
    # with open(detailed_file, 'r') as f:
    #     loaded_results = json.load(f)
    
    # # Extract centers and assignment from the loaded JSON
    # k_str = str(k)
    # if k_str in loaded_results['results']:
    #     fair_results = loaded_results['results'][k_str]['fair']
    #     if 'centers' in fair_results and 'assignment' in fair_results:
    #         loaded_centers = np.array(fair_results['centers'])
    #         loaded_assignment = np.array(fair_results['assignment'])
            
    #         # Create a verification plot
    #         # Create output directory for verification plots
    #         verification_dir = os.path.join(cache_dir, "verification_plots")
    #         os.makedirs(verification_dir, exist_ok=True)
            
    #         # Plot for each pair of dimensions
    #         n_dims = data_matrix.shape[1]
    #         for i in range(n_dims):
    #             for j in range(i+1, n_dims):
    #                 plt.figure(figsize=(10, 8))
                    
    #                 # Plot data points colored by group
    #                 unique_groups = np.unique(group_labels)
    #                 for group_idx in unique_groups:
    #                     mask = group_labels == group_idx
    #                     plt.scatter(data_matrix[mask, i], data_matrix[mask, j], 
    #                                label=f'Group {group_idx}',
    #                                alpha=0.6)
                    
    #                 # Plot centers
    #                 plt.scatter(loaded_centers[:, i], loaded_centers[:, j], 
    #                            c='black', marker='*', s=200, 
    #                            label='Centers')
                    
    #                 # Draw lines from points to their assigned centers
    #                 for point_idx in range(len(data_matrix)):
    #                     if loaded_assignment.ndim == 2:  # If assignment is a matrix
    #                         center_idx = np.argmax(loaded_assignment[point_idx])  # Get the cluster with highest assignment
    #                     else:  # If assignment is a vector
    #                         center_idx = loaded_assignment[point_idx]
                    
    #                     plt.plot([data_matrix[point_idx, i], loaded_centers[center_idx, i]],
    #                             [data_matrix[point_idx, j], loaded_centers[center_idx, j]],
    #                             'gray', alpha=0.1)
                    
    #                 plt.title(f'Verification Plot (k={k}, λ={lambda_param})\nDimensions {i} vs {j}')
    #                 plt.xlabel(f'Dimension {i}')
    #                 plt.ylabel(f'Dimension {j}')
    #                 plt.legend()
    #                 plt.grid(True, linestyle='--', alpha=0.7)
                    
    #                 # Save verification plot
    #                 verification_file = os.path.join(verification_dir, f"verification_k{k}_{lambda_str}_dims_{i}_{j}.png")
    #                 plt.savefig(verification_file, bbox_inches='tight', dpi=300)
    #                 plt.close()
            
    #         print(f"Verification plots saved to: {verification_dir}")
    #         print(f"Loaded centers shape: {loaded_centers.shape}")
    #         print(f"Loaded assignment shape: {loaded_assignment.shape}")
    #     else:
    #         print("Warning: 'centers' or 'assignment' not found in loaded results")
    # else:
    #     print(f"Warning: Results for k={k} not found in loaded JSON")
    
    result_row = {
        'k': k,
        'lambda_param': lambda_param,
        'objective': float(results.get('objective', 0.0)),
        'runtime': results.get('runtime', None),
        'num_points': len(data_matrix),
        'timestamp': pd.Timestamp.now().isoformat(),
        'alpha': config['params']['alpha'],
        'beta': config['params']['beta'],
        'init_method': init_method
    }
    
    if 'group_costs' in results:
        for i, cost in enumerate(results['group_costs']):
            result_row[f'group_{i}_cost'] = float(cost)
    
    df_k = pd.DataFrame([result_row])
    k_file = os.path.join(results_dir, f"welfare_clustering_k{k}_{lambda_str}.csv")
    df_k.to_csv(k_file, index=False)
    print(f"Results for k={k} saved to: {k_file}")
    
    os.remove(temp_config_file)
    
    print(f"Welfare_Clustering experiment for k={k} completed")
    return result_row


def run_welfare_clustering_experiment(cache_dir, config_file, k_min=4, k_max=8, lambda_param=0.5, init_method="random", normalize_data_flag=True, upweight=1.0):
    """
    Run the Welfare_Clustering experiment for multiple k values.
    
    Args:
        cache_dir: str - Directory containing cached data
        config_file: str - Path to config file
        k_min: int - Minimum number of clusters
        k_max: int - Maximum number of clusters
        lambda_param: float - Lambda parameter for fairness
        init_method: str - Initialization method
        normalize_data_flag: bool - Whether to normalize data (default: True)
        upweight: float - Factor to upweight fairness violations (default: 1.0)
    """
    print(f"Running Welfare_Clustering experiment from k={k_min} to k={k_max} with init_method={init_method} and lambda={lambda_param}")
    print(f"Data normalization is {'enabled' if normalize_data_flag else 'disabled'}")
    print(f"Fairness violation upweight factor: {upweight}")
    
    all_results = []
    for k in range(k_min, k_max + 1):
        result_row = run_welfare_clustering_for_k(
            cache_dir=cache_dir,
            config_file=config_file,
            k=k,
            lambda_param=lambda_param,
            init_method=init_method,
            normalize_data_flag=normalize_data_flag,
            upweight=upweight
        )
        all_results.append(result_row)
    
    df_all = pd.DataFrame(all_results)
    results_dir = os.path.join(cache_dir, "welfare_clustering_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Include lambda in combined results filename
    lambda_str = get_lambda_str(lambda_param)
    combined_file = os.path.join(results_dir, f"welfare_clustering_all_k{k_min}_to_{k_max}_{lambda_str}.csv")
    df_all.to_csv(combined_file, index=False)
    print(f"Combined results saved to: {combined_file}")



def main():
    parser = argparse.ArgumentParser(description="Run Welfare_Clustering experiment with cached data")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--config_file", type=str, default="configs/welfare_clustering_config.ini", help="Path to config file for Welfare_Clustering")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, default=8, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Lambda parameter for Welfare_Clustering")
    parser.add_argument("--init_method", type=str, default="random", help="Initialization method (random, kmeans++, uniform_box, hardcoded)")
    parser.add_argument("--normalize_data", action="store_true", help="Whether to normalize data (default: True)")
    parser.add_argument("--no_normalize_data", action="store_false", dest="normalize_data", help="Disable data normalization")
    parser.add_argument("--upweight", type=float, default=1.0, help="Factor to upweight fairness violations (default: 1.0)")
    parser.set_defaults(normalize_data=True)
    args = parser.parse_args()

    run_welfare_clustering_experiment(
        cache_dir=args.cache_dir,
        config_file=args.config_file,
        k_min=args.k_min,
        k_max=args.k_max,
        lambda_param=args.lambda_param,
        init_method=args.init_method,
        normalize_data_flag=args.normalize_data,
        upweight=args.upweight
    )

    print("Welfare_Clustering experiment completed!")

if __name__ == "__main__":
    main()
    