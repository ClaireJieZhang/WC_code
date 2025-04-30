import os
import sys
import argparse
import numpy as np
import pandas as pd
import timeit
from pathlib import Path

# Add the parent directory to the path so we can import from evaluation_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom FCBC utility
from fcbc_cached_util import run_fcbc_with_cached_data

def run_fcbc_experiment(cache_dir, k_min=4, k_max=8, pof_min=1.001, pof_max=1.1, pof_step=0.01, deltas=[0.1], normalize_data_flag=True):
    """
    Run FCBC experiment for multiple k values and POF values using cached data.
    
    Args:
        cache_dir: str - Directory containing cached data
        k_min: int - Minimum number of clusters
        k_max: int - Maximum number of clusters
        pof_min: float - Minimum POF value
        pof_max: float - Maximum POF value
        pof_step: float - Step size for POF values
        deltas: list - List of delta values
        normalize_data_flag: bool - Whether to normalize data (default: True)
    """
    print(f"Running FCBC experiment with k from {k_min} to {k_max} and POF from {pof_min} to {pof_max}")
    print(f"Data normalization is {'enabled' if normalize_data_flag else 'disabled'}")

    # Load the cached data
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))

    # Create POF values
    pof_values = np.linspace(
        start=pof_min,
        stop=pof_max,
        num=int((pof_max - pof_min) / pof_step) + 1,
        endpoint=True
    )

    # Create results directory
    results_dir = os.path.join(cache_dir, "fcbc_results")
    os.makedirs(results_dir, exist_ok=True)

    # Store all results across k and pof
    all_results = []

    for k in range(k_min, k_max + 1):
        k_results = []

        for pof in pof_values:
            # Run the experiment for this k and POF
            outputs = run_fcbc_with_cached_data(
                data_matrix=data_matrix,
                group_labels=group_labels,
                k=k,
                deltas=deltas,
                pof=pof,
                scaling=normalize_data_flag
            )

            # Each output corresponds to a different delta
            for output in outputs:
                # Convert centers and assignment to string format for CSV storage
                # centers is already a list of lists, so we can use str directly
                centers_str = str(output['centers'])
                
                # assignment is a flattened binary matrix of shape (num_points, num_clusters)
                # where each row has exactly one 1 indicating cluster assignment
                assignment = np.array(output['assignment'])
                num_points = len(data_matrix)
                num_clusters = k
                # Reshape back to binary matrix and find cluster indices
                assignment = assignment.reshape(num_points, num_clusters)
                print(f"First row of reshaped assignment matrix: {assignment[0]}")
                # Convert binary matrix to cluster indices (0 to k-1)
                assignment = np.argmax(assignment, axis=1).astype(int)
                assignment_str = str(assignment.tolist())
                
                result_row = {
                    'k': k,
                    'pof': pof,
                    'delta': output['delta'],
                    'util_value': output['util_objective'],
                    'util_lp': output['util_lp'],
                    'lp_iters': output["bs_iterations"],
                    'opt_index': output['opt_index'],
                    'epsilon': output["epsilon"],
                    'epsilon_set_size': output["epsilon set size "],
                    'min_clust_size': min(output["sizes"]),
                    'runtime': output["time"],
                    'fair_cost': output['objective'],
                    'colorblind_cost': output['unfair_score'],
                    'num_points': len(data_matrix),
                    'centers': centers_str,
                    'assignment': assignment_str
                }
                k_results.append(result_row)
                all_results.append(result_row)

        # Save results for this k value
        if k_results:
            df = pd.DataFrame(k_results)
            csv_file = os.path.join(results_dir, f'fcbc_k{k}.csv')
            df.to_csv(csv_file, index=False)
            print(f"✅ Results for k={k} saved to {csv_file}")

    # Save all results in one file
    if all_results:
        df_all = pd.DataFrame(all_results)
        csv_file = os.path.join(results_dir, f'fcbc_all_results.csv')
        df_all.to_csv(csv_file, index=False)
        print(f"✅ All results saved to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Run FCBC experiment with cached data")

    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, default=8, help="Maximum number of clusters")
    parser.add_argument("--pof_min", type=float, default=1.001, help="Minimum POF value")
    parser.add_argument("--pof_max", type=float, default=1.1, help="Maximum POF value")
    parser.add_argument("--pof_step", type=float, default=0.01, help="Step size for POF values")
    parser.add_argument("--deltas", type=str, default="0.1", help="Comma-separated list of delta values")
    parser.add_argument("--normalize_data", type=str, default="False", choices=["True", "False"],
                        help="Whether to normalize data (True/False)")

    args = parser.parse_args()
    
    # Convert deltas string to list of floats
    deltas = [float(x.strip()) for x in args.deltas.split(',')]
    
    # Convert normalize_data string to boolean
    normalize_data_flag = args.normalize_data.lower() == "true"
    
    run_fcbc_experiment(
        cache_dir=args.cache_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        pof_min=args.pof_min,
        pof_max=args.pof_max,
        pof_step=args.pof_step,
        deltas=deltas,
        normalize_data_flag=normalize_data_flag
    )
    
    print("✅ FCBC experiment completed!")

if __name__ == "__main__":
    main()
