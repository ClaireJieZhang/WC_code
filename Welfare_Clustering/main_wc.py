import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import OrdinalEncoder
from fairlearn.datasets import fetch_adult
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from lp_solver import run_connector_and_solve_lp
from min_cost_rounding import rounding_wrapper
from clustering_utils import give_rand_centers, lloyd, comp_cost  # assumed helper functions
import configparser
from util.configutil import read_list
from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key)
from util.results_util import save_clustering_results
from collections import defaultdict
import sys

def normalize_data(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    return (X - X_mean) / (X_std + 1e-8)







def load_data(config_file, dataset_name, max_points=None):
    """
    Load and preprocess data for clustering.
    
    Args:
        config_file: str - Path to config file
        dataset_name: str - Name of dataset section in config
        max_points: int, optional - Maximum number of points to use
        
    Returns:
        df: pandas DataFrame - Preprocessed features
        color_flag_array: numpy array - Group labels
        group_names: list - Names of the groups
    """
    # === Load config ===
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # === Step 1: Load raw data
    df = read_data(config, dataset_name)

    # === Step 2: Subsample if needed
    if max_points and len(df) > max_points:
        df = df.head(max_points)

    # === Step 3: Evaluate group membership using condition logic
    variable_of_interest = config[dataset_name].getlist("fairness_variable")
    assert len(variable_of_interest) == 1
    variable = variable_of_interest[0]

    bucket_conditions = config[dataset_name].getlist(f"{variable}_conditions")

    color_flag_array = np.zeros(len(df), dtype=int)
    group_counts = [0 for _ in bucket_conditions]

    print("[DEBUG] About to run eval() on variable:", variable)
    print("[DEBUG] Sample value to eval:", df.iloc[0][variable])
    print("[DEBUG] Conditions:", bucket_conditions)
    
    for i, row in df.iterrows():
        for bucket_idx, bucket in enumerate(bucket_conditions):
            try:
                if eval(bucket)(row[variable]):
                    color_flag_array[i] = bucket_idx
                    group_counts[bucket_idx] += 1
                    break
            except Exception as e:
                print(f"⚠️ Error evaluating condition on row {i}: {row[variable]} → {e}")

    # === Step 4: Clean the data (e.g., encode strings)
    df, _ = clean_data(df, config, dataset_name)

    # === Step 5: Filter to selected clustering columns
    selected_columns = config[dataset_name].getlist("columns")
    df = df[selected_columns]

    # === Step 6: Group names (optional override from config)
    group_names_key = f"{variable}_group_names"
    if group_names_key in config[dataset_name]:
        group_names = config[dataset_name].getlist(group_names_key)
    else:
        group_names = [f"Group {i}" for i in range(len(bucket_conditions))]

    print(f"[INFO] Group counts: {dict(enumerate(group_counts))}")
    return df, color_flag_array, group_names




def socially_fair_kmeans(data_normalized, svar_all, k, num_iters=10, best_out_of=5):
    #np.random.seed(42)
    rand_centers, _, _ = give_rand_centers(
        data_normalized, data_normalized, data_normalized, k, best_out_of
    )

    print("data_normalized shape:", data_normalized.shape)
    print("svar_all shape:", svar_all.shape)
    print("unique groups:", np.unique(svar_all, return_counts=True))


    centers_f, clustering_f, runtime_f = lloyd(
        data_normalized, svar_all, k, num_iters, best_out_of, rand_centers, is_fair=1, verbose=False
    )
    return centers_f


def run_full_pipeline(config_file, dataset_name, lambda_param=0.5, max_points=None):

    # === Load config ===
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    k = int(config["clustering"]["num_clusters"])
    alpha_val = ast.literal_eval(config["params"]["alpha"])
    beta_val = ast.literal_eval(config["params"]["beta"])

    # === Step 1: Load data via FCBC-style logic ===
    df, svar_all, group_names = load_data(config_file, dataset_name, max_points=max_points)
    data_normalized = normalize_data(df.values)

    #+++++++++++++++++++++check data proportion+++++++++++++++++++++++++++++++#
    unique, counts = np.unique(svar_all, return_counts=True)
    total = len(svar_all)

    print("\n[Diagnostics] Global group ratios:")
    for u, c in zip(unique, counts):
        print(f"Group {u}: count={c}, proportion={c/total:.3f}")

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


    # === Step 2: Socially fair clustering to get centers ===
    centers = socially_fair_kmeans(data_normalized, svar_all, k=k)

    # === Step 3: Distance matrix ===
    distance_matrix = cdist(data_normalized, centers, metric='sqeuclidean')

    # === Step 4: Setup alpha, beta per group (gracefully handle scalar or dict input) ===
    unique_groups = np.unique(svar_all)

    if isinstance(alpha_val, dict):
        alpha = alpha_val
    else:
        alpha = {h: alpha_val for h in unique_groups}

    if isinstance(beta_val, dict):
        beta = beta_val
    else:
        beta = {h: beta_val for h in unique_groups}

    # === Step 5: LP + Rounding ===
    df_normalized = pd.DataFrame(data_normalized)
    lp_result = run_connector_and_solve_lp(df_normalized, svar_all, centers, alpha, beta, lambda_param)
    lp_assignment = lp_result['x_frac']
    lp_objective = lp_result['z']

    final_result = rounding_wrapper(
        lp_assignment=lp_assignment,
        distance_matrix=distance_matrix,
        color_labels=svar_all,
        num_clusters=centers.shape[0],
        num_colors=len(unique_groups),
        lp_objective=lp_objective,
        df=data_normalized,
        centers=centers
    )
    
    # Add group names to the result
    final_result['group_names'] = group_names

    return final_result



if __name__ == "__main__":
  
    # === Load outer (controller) config file ===
    outer_config_file = "config/my_experiment.ini"
    outer_config = configparser.ConfigParser(converters={'list': read_list})
    outer_config.read(outer_config_file)

    # === Get which experiment to run from command line ===
    config_str = sys.argv[1] if len(sys.argv) > 1 else "adult_sex"

    # === Extract fields from outer config section ===
    dataset_name = outer_config[config_str]["dataset"]
    config_file = outer_config[config_str]["config_file"]
    lambda_param = float(outer_config[config_str].get("lambda_param", 0.5))
    max_points = outer_config[config_str].getint("max_points", fallback=None)

    # === Call the actual pipeline ===
    result = run_full_pipeline(
        config_file=config_file,
        dataset_name=dataset_name,
        lambda_param=lambda_param,
        max_points=max_points
    )

    print("Rounded clustering objective:", result['objective'])
    print("Group proportions (normalized):", result['proportions_normalized'])

    # === Save results ===
    config_params = {
        "dataset": dataset_name,
        "config_file": config_file,
        "lambda_param": lambda_param,
        "max_points": max_points
    }
    
    save_clustering_results(
        result=result,
        config_params=config_params,
        dataset_name=dataset_name
    )