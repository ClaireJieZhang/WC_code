#!/usr/bin/env python3
"""
Script to run all three clustering pipelines with the same data and standardize the results.
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import configparser
from data_utils import load_data_from_config, normalize_data
from standardize_results import standardize_and_save_results

def read_list(value):
    """Helper function to read comma-separated lists from config files."""
    return [item.strip() for item in value.split(',')]

def run_samira_pipeline(config, k_min, k_max, num_iters, best_out_of, verbose=False):
    """Run the Samira_SF_Python pipeline."""
    dataset_name = config['DEFAULT']['dataset_name']
    
    # Get the absolute path to the Samira_SF_Python directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samira_dir = os.path.join(base_dir, "Samira_SF_Python")
    
    cmd = [
        "python", os.path.join(samira_dir, "main.py"),
        "--dataset", dataset_name,
        "--k_min", str(k_min),
        "--k_max", str(k_max),
        "--iters", str(num_iters),
        "--best_out_of", str(best_out_of)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running Samira_SF_Python pipeline on {dataset_name}...")
    subprocess.run(cmd, check=True)
    
    # Return the path to the results file
    return os.path.join(samira_dir, "results", f"{dataset_name}_k{k_min}-{k_max}_results.pkl")

def run_welfare_pipeline(config, k, lambda_param, max_points=None):
    """Run the Welfare_Clustering pipeline."""
    dataset_name = config['DEFAULT']['dataset_name']
    dataset_section = config[dataset_name]
    
    # Get the absolute path to the workspace and Welfare_Clustering directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    welfare_dir = os.path.join(base_dir, "Welfare_Clustering")
    
    # Create a temporary config file for this run
    config_content = f"""[DEFAULT]
dataset_name = {dataset_name}

[{dataset_name}]
dataset_name = {dataset_name}
csv_file = {dataset_section.get('csv_file', 'data/' + dataset_name + '.csv')}
separator = {dataset_section.get('separator', ',')}
columns = {dataset_section.get('columns', '')}
text_columns = {dataset_section.get('text_columns', '')}
fairness_variable = {dataset_section.get('fairness_variable', '')}
variable_of_interest = {dataset_section.get('variable_of_interest', '')}
sex_conditions = {dataset_section.get('sex_conditions', '')}
sex_group_names = {dataset_section.get('sex_group_names', '')}
max_points = {max_points if max_points else dataset_section.get('max_points', '10000')}

[clustering]
clustering_method = socially_fair_kmeans
num_clusters = {k}

[params]
alpha = {{0: {lambda_param}, 1: {lambda_param}}}
beta  = {{0: {lambda_param}, 1: {lambda_param}}}
"""
    
    os.makedirs(os.path.join(welfare_dir, "config"), exist_ok=True)
    temp_config_path = os.path.join(welfare_dir, "config", "temp_config.ini")
    
    with open(temp_config_path, "w") as f:
        f.write(config_content)
    
    # Run the pipeline from the Welfare_Clustering directory
    cmd = [
        "python", "main_wc.py",
        f"{dataset_name}_{dataset_section.get('fairness_variable', '')}"
    ]
    
    print(f"Running Welfare_Clustering pipeline on {dataset_name}...")
    subprocess.run(cmd, check=True, cwd=welfare_dir)
    
    # Return the path to the results file
    return os.path.join(welfare_dir, "results", f"{dataset_name}_{dataset_section.get('fairness_variable', '')}_results.json")

def run_fcbc_pipeline(config, k, lambda_param, max_points=None):
    """Run the Fair-Clustering-Under-Bounded-Cost pipeline."""
    dataset_name = config['DEFAULT']['dataset_name']
    dataset_section = config[dataset_name]
    
    # Create a temporary config file for this run
    config_content = f"""[{dataset_name}_{dataset_section.get('fairness_variable', '')}]
csv_file = {dataset_section.get('csv_file', 'data/' + dataset_name + '.csv')}
separator = {dataset_section.get('separator', ',')}
columns = {dataset_section.get('columns', '')}
text_columns = {dataset_section.get('text_columns', '')}
variable_of_interest = {dataset_section.get('variable_of_interest', '')}
fairness_variable = {dataset_section.get('fairness_variable', '')}
race_conditions = {dataset_section.get('race_conditions', '')}
sex_conditions = {dataset_section.get('sex_conditions', '')}
"""
    
    os.makedirs("Fair-Clustering-Under-Bounded-Cost/config", exist_ok=True)
    temp_config_path = "Fair-Clustering-Under-Bounded-Cost/config/temp_config.ini"
    
    with open(temp_config_path, "w") as f:
        f.write(config_content)
    
    # Run the pipeline
    cmd = [
        "python", "Fair-Clustering-Under-Bounded-Cost/main.py",
        "--config", temp_config_path,
        "--dataset", f"{dataset_name}_{dataset_section.get('fairness_variable', '')}",
        "--k", str(k),
        "--lambda", str(lambda_param)
    ]
    
    if max_points:
        cmd.extend(["--max_points", str(max_points)])
    
    print(f"Running Fair-Clustering-Under-Bounded-Cost pipeline on {dataset_name}...")
    subprocess.run(cmd, check=True)
    
    # Return the path to the results file
    return f"Fair-Clustering-Under-Bounded-Cost/results/{dataset_name}_{dataset_section.get('fairness_variable', '')}_k{k}_results.pkl"

def main():
    parser = argparse.ArgumentParser(description="Run all clustering pipelines with the same data.")
    parser.add_argument("--config", type=str, required=True, help="Path to the common configuration file")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight between distance and fairness costs")
    parser.add_argument("--max_points", type=int, help="Maximum number of points to use")
    parser.add_argument("--output_dir", type=str, default="standardized_results", help="Directory to save standardized results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress logs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the configuration
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(args.config)
    
    dataset_name = config['DEFAULT']['dataset_name']
    
    # Load the data once to ensure consistency
    print(f"Loading {dataset_name} dataset...")
    data, group_labels, group_names = load_data_from_config(args.config, dataset_name, max_points=args.max_points)
    
    # Run each pipeline
    samira_results = run_samira_pipeline(config, k_min=args.k, k_max=args.k, num_iters=10, best_out_of=10, verbose=args.verbose)
    welfare_results = run_welfare_pipeline(config, k=args.k, lambda_param=args.lambda_param, max_points=args.max_points)
    fcbc_results = run_fcbc_pipeline(config, k=args.k, lambda_param=args.lambda_param, max_points=args.max_points)
    
    # Standardize results
    print("\nStandardizing results...")
    
    # Samira_SF_Python
    standardize_and_save_results(
        pipeline_name="Samira_SF_Python",
        result_file=samira_results,
        output_file=os.path.join(args.output_dir, f"{dataset_name}_samira_standardized.json"),
        data=data,
        group_labels=group_labels,
        lambda_param=args.lambda_param,
        p=2
    )
    
    # Welfare_Clustering
    standardize_and_save_results(
        pipeline_name="Welfare_Clustering",
        result_file=welfare_results,
        output_file=os.path.join(args.output_dir, f"{dataset_name}_welfare_standardized.json"),
        data=data,
        group_labels=group_labels,
        lambda_param=args.lambda_param,
        p=2
    )
    
    # Fair-Clustering-Under-Bounded-Cost
    standardize_and_save_results(
        pipeline_name="Fair-Clustering-Under-Bounded-Cost",
        result_file=fcbc_results,
        output_file=os.path.join(args.output_dir, f"{dataset_name}_fcbc_standardized.json"),
        data=data,
        group_labels=group_labels,
        lambda_param=args.lambda_param,
        p=2
    )
    
    print(f"\n✅ All pipelines completed and results standardized in {args.output_dir}/")

if __name__ == "__main__":
    main() 