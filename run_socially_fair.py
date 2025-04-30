import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import json

# Add the parent directory to the path so we can import from evaluation_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_socially_fair_experiment(cache_dir, k_min=4, k_max=8, num_iters=10, best_out_of=10,
                                  init_method="naive", normalize_data_flag=False):
    """
    Run the SociallyFair_Python experiment with cached data.

    Args:
        cache_dir: str - Directory containing cached data
        k_min: int - Minimum number of clusters
        k_max: int - Maximum number of clusters
        num_iters: int - Number of iterations for Lloyd's algorithm
        best_out_of: int - Number of random initializations to try
        init_method: str - Initialization method ('naive', 'uniform_box', 'kmeanspp', 'fixed_synthetic')
        normalize_data_flag: bool - Whether to normalize data
    """
    print(f"Running SociallyFair_Python experiment with k from {k_min} to {k_max}")
    
    # Load the cached data
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Read group names
    with open(os.path.join(cache_dir, "group_names.txt"), "r") as f:
        group_names = [line.strip() for line in f.readlines()]
    
    # Import the run_sf_pipeline_with_loaded_data function
    from SociallyFair_Python.main import run_sf_pipeline_with_loaded_data
    
    # Run the experiment
    results = run_sf_pipeline_with_loaded_data(
        data_all=data_matrix,
        svar_all=group_labels,
        dataset_name="cached_data",
        k_min=k_min,
        k_max=k_max,
        num_iters=num_iters,
        best_out_of=best_out_of,
        verbose=True,
        init_method=init_method,
        normalize_data_flag=normalize_data_flag
    )
    
    # Create results directory
    results_dir = os.path.join(cache_dir, "sf_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results for each k value
    for k, result in results.items():
        detailed_results = {
            'pipeline': 'SociallyFair_Python',
            'dataset': 'cached_data',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'k': k,
                'num_iters': num_iters,
                'best_out_of': best_out_of,
                'init_method': init_method,
                'normalize_data': normalize_data_flag
            },
            'data': {
                'points': data_matrix.tolist(),
                'group_labels': group_labels.tolist(),
                'group_names': group_names,
                'centers': result['centers_f'].tolist(),
                'assignment': result['assignment'].tolist(),
                'metrics': {
                    'runtime': result['runtime_f'],
                    'cost': result['cost_f'].tolist(),
                    'distance_cost': result.get('distance_cost', 0.0),
                    'fairness_cost': result.get('fairness_cost', 0.0)
                }
            }
        }
        
        # Save in standardized format
        detailed_file = os.path.join(results_dir, f"detailed_results_k{k}.json")
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"✅ Detailed results for k={k} saved to {detailed_file}")
    
    # Save summary results
    summary_results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'k_range': list(range(k_min, k_max + 1)),
            'num_iters': num_iters,
            'best_out_of': best_out_of,
            'init_method': init_method,
            'normalize_data': normalize_data_flag
        },
        'summary': {str(k): {
            'runtime': results[k]['runtime_f'],
            'cost': results[k]['cost_f'].tolist()
        } for k in results}
    }
    
    summary_file = os.path.join(results_dir, f"summary_results_k{k_min}_to_{k_max}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"✅ Summary results saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Run SociallyFair_Python experiment with cached data")

    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, default=8, help="Maximum number of clusters")
    parser.add_argument("--num_iters", type=int, default=10, help="Number of iterations for Lloyd's algorithm")
    parser.add_argument("--best_out_of", type=int, default=10, help="Number of random initializations")
    parser.add_argument("--init_method", type=str, default="naive",
                        choices=["naive", "uniform_box", "kmeanspp", "fixed_synthetic"],
                        help="Initialization method for centers")
    parser.add_argument("--normalize_data_flag", action="store_true",
                        help="If set, normalize data to zero mean and unit variance")

    args = parser.parse_args()
    
    run_socially_fair_experiment(
        cache_dir=args.cache_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        num_iters=args.num_iters,
        best_out_of=args.best_out_of,
        init_method=args.init_method,
        normalize_data_flag=args.normalize_data_flag
    )
    
    print("✅ SociallyFair_Python experiment completed!")

if __name__ == "__main__":
    main()
