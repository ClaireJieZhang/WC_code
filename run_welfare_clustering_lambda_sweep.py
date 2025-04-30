import os
import sys
import numpy as np
import configparser
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_welfare_clustering import run_welfare_clustering_for_k

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Get clustering parameters
    k = config.getint('clustering', 'num_clusters')
    
    # Get alpha and beta parameters
    alpha = eval(config.get('params', 'alpha'))
    beta = eval(config.get('params', 'beta'))
    
    return k, alpha, beta

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='cache/cache_synthetic_noisy',
                       help='Directory containing cached data')
    parser.add_argument('--config_path', type=str, default='configs/welfare_clustering_config.ini',
                       help='Path to welfare clustering config file')
    parser.add_argument('--output_dir', type=str, default='cache/welfare_clustering_results',
                       help='Directory to save results')
    parser.add_argument('--upweight', type=float, default=1.0,
                       help='Factor to upweight fairness violations')
    parser.add_argument('--lambda_start', type=float, default=0.0,
                       help='Starting value for lambda sweep')
    parser.add_argument('--lambda_end', type=float, default=0.05,
                       help='Ending value for lambda sweep')
    parser.add_argument('--lambda_step', type=float, default=0.001,
                       help='Step size for lambda sweep')
    args = parser.parse_args()

    # Load config
    k, alpha, beta = load_config(args.config_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate lambda values with specified parameters
    lambda_values = np.arange(args.lambda_start, args.lambda_end + args.lambda_step, args.lambda_step)
    
    # Run welfare clustering for each lambda
    for lambda_param in lambda_values:
        print(f"\nRunning welfare clustering with lambda = {lambda_param:.3f}")
        
        # Run clustering using run_welfare_clustering_for_k
        result_row = run_welfare_clustering_for_k(
            cache_dir=args.cache_dir,
            config_file=args.config_path,
            k=k,
            lambda_param=lambda_param,
            init_method="kmeanspp",  # Changed from "kmeans++" to "kmeanspp"
            normalize_data_flag=False,  # As requested, no normalization
            upweight=args.upweight  # Pass the upweight parameter
        )
        
        print(f"Completed run for lambda = {lambda_param:.3f}")
        print(f"Objective value: {result_row['objective']}")

if __name__ == '__main__':
    main() 