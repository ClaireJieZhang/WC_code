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

def run_fcbc_experiment(cache_dir, k_min=4, k_max=8, pof_min=1.001, pof_max=1.1, pof_step=0.01, deltas=[0.1], lambda_param=0.5):
    """
    Run FCBC experiment for multiple k values and POF values using cached data.
    """
    print(f"Running FCBC experiment with k from {k_min} to {k_max} and POF from {pof_min} to {pof_max}")

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
                lambda_param=lambda_param
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
                    'lambda_param': lambda_param,
                    'num_points': len(data_matrix),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'centers': centers_str,
                    'assignment': assignment_str
                }
                k_results.append(result_row)
                all_results.append(result_row)

        # Save results for this k value
        df_k = pd.DataFrame(k_results)
        k_file = os.path.join(results_dir, f"fcbc_k{k}_all_pof.csv")
        df_k.to_csv(k_file, index=False)
        print(f"Results for k={k} with all POF values saved to: {k_file}")

    # Save combined results
    df_all = pd.DataFrame(all_results)
    combined_file = os.path.join(results_dir, f"fcbc_all_k{k_min}_to_{k_max}_pof{pof_min}_to_{pof_max}.csv")
    df_all.to_csv(combined_file, index=False)
    print(f"Combined results for all k and POF values saved to: {combined_file}")

def main():
    parser = argparse.ArgumentParser(description="Run FCBC experiment with cached data")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, default=8, help="Maximum number of clusters")
    parser.add_argument("--pof_min", type=float, default=1.001, help="Minimum POF value")
    parser.add_argument("--pof_max", type=float, default=1.1, help="Maximum POF value")
    parser.add_argument("--pof_step", type=float, default=0.01, help="Step size for POF values")
    parser.add_argument("--deltas", type=str, default="0.1", help="Comma-separated list of delta values")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Lambda parameter for welfare cost calculation")
    args = parser.parse_args()

    # Parse deltas
    deltas = [float(d.strip()) for d in args.deltas.split(",")]

    run_fcbc_experiment(
        cache_dir=args.cache_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        pof_min=args.pof_min,
        pof_max=args.pof_max,
        pof_step=args.pof_step,
        deltas=deltas,
        lambda_param=args.lambda_param
    )

    print("FCBC experiment completed!")

if __name__ == "__main__":
    main()
