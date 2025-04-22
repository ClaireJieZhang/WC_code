#!/usr/bin/env python3
"""
Script to compare welfare costs across different pipelines.
"""

import os
import argparse
import numpy as np
from plotting import load_and_plot_results
import sys

def main():
    parser = argparse.ArgumentParser(description="Run comparison between clustering pipelines.")
    parser.add_argument("--samira_results", type=str, help="Path to Samira_SF_Python results")
    parser.add_argument("--welfare_results", type=str, help="Path to Welfare_Clustering results")
    parser.add_argument("--fair_results", type=str, help="Path to Fair-Clustering-Under-Bounded-Cost results")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Directory to save comparison results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress logs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    result_files = {}
    if args.samira_results:
        result_files['Samira_SF_Python'] = args.samira_results
    if args.welfare_results:
        result_files['Welfare_Clustering'] = args.welfare_results
    if args.fair_results:
        result_files['Fair-Clustering-Under-Bounded-Cost'] = args.fair_results
    
    if not result_files:
        print("Error: No result files provided. Use --samira_results, --welfare_results, and/or --fair_results.")
        sys.exit(1)
    
    # Generate output filename
    output_file = os.path.join(
        args.output_dir, 
        f"welfare_costs_{args.dataset}_k{args.k}_lambda{args.lambda_param}_p{args.p}.png"
    )
    
    # Load and plot results
    try:
        load_and_plot_results(
            result_files=result_files,
            dataset_name=args.dataset,
            k=args.k,
            lambda_param=args.lambda_param,
            p=args.p,
            save_path=output_file
        )
        print(f"Comparison plot saved to: {output_file}")
    except Exception as e:
        print(f"Error generating plot: {e}")

if __name__ == "__main__":
    main() 