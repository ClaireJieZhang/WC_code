import os
import sys
import argparse
import numpy as np
import pandas as pd

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
    
    # Save results to a file
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        for k, result in results.items():
            f.write(f"k={k}: {result}\n")
    
    print(f"SociallyFair_Python results saved to {results_dir}")

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