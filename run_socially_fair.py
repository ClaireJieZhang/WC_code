import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Add the parent directory to the path so we can import from evaluation_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_socially_fair_experiment(cache_dir, k_min=4, k_max=8, num_iters=10, best_out_of=10):
    """
    Run the SociallyFair_Python experiment with cached data.
    
    Args:
        cache_dir: str - Directory containing cached data
        k_min: int - Minimum number of clusters
        k_max: int - Maximum number of clusters
        num_iters: int - Number of iterations for Lloyd's algorithm
        best_out_of: int - Number of random initializations to try
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
        verbose=True
    )
    
    # Save results
    results_dir = os.path.join(cache_dir, "socially_fair_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a structured results dictionary
    structured_results = {
        'pipeline': 'SociallyFair_Python',
        'dataset': 'cached_data',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'points': data_matrix.tolist(),
            'group_labels': group_labels.tolist(),
            'group_names': group_names
        },
        'results': {}
    }
    
    # Process each k value's results
    for k, result in results.items():
        # Only save the 'fair' results
        structured_results['results'][str(k)] = {
            'fair': {
                'centers': result['centers_f'].tolist(),
                'assignment': result['clustering_f'][0].tolist(),  # Take first assignment
                'runtime': result['runtime_f'],
                'cost': result['cost_f'].tolist()
            }
        }
    
    # Save to pickle file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"sf_results_{timestamp}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(structured_results, f)
    
    print(f"SociallyFair_Python results saved to {output_file}")
    
    # Also save results in text format for easy viewing
    txt_file = os.path.join(results_dir, "results.txt")
    with open(txt_file, 'w') as f:
        for k in range(k_min, k_max + 1):
            result = results[k]
            f.write(f"k={k}: {result}\n")
    
    print(f"Results also saved in text format to {txt_file}")

def main():
    parser = argparse.ArgumentParser(description="Run SociallyFair_Python experiment with cached data")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, default=8, help="Maximum number of clusters")
    parser.add_argument("--num_iters", type=int, default=10, help="Number of iterations for Lloyd's algorithm")
    parser.add_argument("--best_out_of", type=int, default=10, help="Number of random initializations to try")
    args = parser.parse_args()
    
    run_socially_fair_experiment(
        cache_dir=args.cache_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        num_iters=args.num_iters,
        best_out_of=args.best_out_of
    )
    
    print("SociallyFair_Python experiment completed!")

if __name__ == "__main__":
    main() 