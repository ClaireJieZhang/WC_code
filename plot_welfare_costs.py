#!/usr/bin/env python3
import argparse
import os
from evaluation_utils.plotting import plot_welfare_costs_from_csv

def main():
    parser = argparse.ArgumentParser(description="Plot welfare costs from evaluation results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k_min", type=int, required=True, help="Minimum number of clusters")
    parser.add_argument("--k_max", type=int, required=True, help="Maximum number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight between distance and fairness costs")
    parser.add_argument("--output", type=str, default="welfare_costs_comparison.png", help="Output file path for the plot")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only try to create directory if there is a directory path
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot welfare costs
    plot_welfare_costs_from_csv(
        cache_dir=args.cache_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        lambda_param=args.lambda_param,
        save_path=args.output
    )
    
    print(f"Plot saved to: {args.output}")

if __name__ == "__main__":
    main() 