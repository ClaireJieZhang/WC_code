import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import OrdinalEncoder
from fairlearn.datasets import fetch_adult
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from lp_solver import run_connector_and_solve_lp
from min_cost_rounding import rounding_wrapper
from clustering_utils import (
    give_rand_centers, 
    lloyd, 
    comp_cost, 
    kmeans_plus_plus_initialization,
    uniform_box_initialization,
    hardcoded_initialization
)

import configparser
from util.configutil import read_list
from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key)
from util.results_util import save_clustering_results
from collections import defaultdict
import sys
from evaluation_utils.data_utils import load_and_prepare_data
import os


def normalize_data(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    return (X - X_mean) / (X_std + 1e-8)







def compute_group_costs_post_lp(x_frac, data_points, centers, group_labels, lambda_param, upweight=1.0):
    distance_matrix = cdist(data_points, centers, metric="sqeuclidean")
    n, k = x_frac.shape
    colors = np.unique(group_labels)
    costs_by_group = {}

    for h in colors:
        indices_h = np.where(group_labels == h)[0]
        group_size = len(indices_h)

        # Distance term
        dist_sum = 0.0
        for j in indices_h:
            for i in range(k):
                dist_sum += lambda_param * x_frac[j, i] * distance_matrix[j, i]
        dist_term = dist_sum / group_size

        # Fairness term: compute proportional assignment to each center
        center_totals = x_frac.sum(axis=0)  # shape (k,)
        center_group = x_frac[indices_h].sum(axis=0)  # group h mass at each center
        deviation = 0.0
        r_h = group_size / n

        for i in range(k):
            total = center_totals[i]
            group_mass = center_group[i]
            if total > 1e-6:
                prop = group_mass / total
                deviation += abs(prop - r_h)
        dev_term = ((1 - lambda_param) * upweight * deviation) / group_size

        costs_by_group[h] = {
            "distance_cost": dist_term,
            "fairness_cost": dev_term,
            "total_cost": dist_term + dev_term
        }

    return costs_by_group


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



def socially_fair_kmeans(data_normalized, svar_all, k, num_iters=10, best_out_of=5, init_method="random"):
    """
    Socially fair k-means clustering.

    Args:
        data_normalized: (n, d) array
        svar_all: (n,) group labels
        k: number of clusters
        num_iters: number of Lloyd updates
        best_out_of: number of random initializations (ignored if init_method != 'random')
        init_method: 'random', 'kmeans++', 'uniform_box', or 'hardcoded'

    Returns:
        centers_f: (k, d) final centers
    """

    print("\nDEBUG - socially_fair_kmeans:")
    print(f"data_normalized shape: {data_normalized.shape}")
    print(f"svar_all shape: {svar_all.shape}")
    print(f"k: {k}")
    print(f"init_method: {init_method}")

    # For non-random initialization methods, we only need one initialization
    actual_best_out_of = 1 if init_method != "random" else best_out_of

    if init_method == "random":
        rand_centers, _, _ = give_rand_centers(
            data_normalized, data_normalized, data_normalized, k, actual_best_out_of
        )
    elif init_method == "kmeanspp":
        rand_centers = [kmeans_plus_plus_initialization(data_normalized, k)]
    elif init_method == "uniform_box":
        rand_centers = [uniform_box_initialization(data_normalized, k)]
    elif init_method == "hardcoded":
        rand_centers = [hardcoded_initialization(data_normalized, k)]
    else:
        raise ValueError(f"Unknown init_method {init_method}")

    print(f"rand_centers type: {type(rand_centers)}")
    print(f"rand_centers length: {len(rand_centers)}")
    if len(rand_centers) > 0:
        print(f"rand_centers[0] shape: {rand_centers[0].shape}")

    print("data_normalized shape:", data_normalized.shape)
    print("svar_all shape:", svar_all.shape)
    print("unique groups:", np.unique(svar_all, return_counts=True))

    centers_f, clustering_f, runtime_f = lloyd(
        data_normalized, svar_all, k, num_iters, actual_best_out_of, rand_centers, is_fair=1, verbose=False
    )
    return centers_f



def run_full_pipeline(config_file, dataset_name, lambda_param=0.5, max_points=None, init_method="random"):
    """
    Welfare Clustering full pipeline (loads and normalizes data, performs SF clustering, LP solve, rounding).

    Args:
        config_file: str - path to config file
        dataset_name: str - section name
        lambda_param: float - LP fairness parameter
        max_points: int or None - optional subsampling
        init_method: str - initialization method for clustering ('random', 'kmeans++', etc.)

    Returns:
        final_result: dict containing clustering and assignment results
    """

    # === Load config ===
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    k = int(config["clustering"]["num_clusters"])
    alpha_val = ast.literal_eval(config["params"]["alpha"])
    beta_val = ast.literal_eval(config["params"]["beta"])

    # === Step 1: Load and normalize data ===
    df, svar_all, group_names = load_data(config_file, dataset_name, max_points=max_points)
    data_normalized = normalize_data(df.values)

    # +++ Diagnostics on group proportions +++
    unique, counts = np.unique(svar_all, return_counts=True)
    total = len(svar_all)

    print("\n[Diagnostics] Global group ratios:")
    for u, c in zip(unique, counts):
        print(f"Group {u}: count={c}, proportion={c/total:.3f}")
    # ++++++++++++++++++++++++++++++++++++++++++

    # === Step 2: Socially fair clustering to get centers ===
    centers = socially_fair_kmeans(data_normalized, svar_all, k=k, init_method=init_method)

    # === Step 3: Compute distance matrix ===
    distance_matrix = cdist(data_normalized, centers, metric='sqeuclidean')

    # === Step 4: Setup alpha, beta per gro



def run_full_pipeline_with_loaded_data(df, svar_all, group_names, config_file, dataset_name, lambda_param=0.5, init_method="random", normalize_data_flag=True, upweight=1.0):
    """
    Welfare Clustering pipeline using pre-loaded and pre-cleaned data.

    Args:
        df: pd.DataFrame — cleaned and scaled feature matrix
        svar_all: np.ndarray — group labels (0/1)
        group_names: list — string names of each group
        config_file: str — path to .ini config
        dataset_name: str — section name
        lambda_param: float — LP lambda weight
        init_method: str — initialization method
        normalize_data_flag: bool - whether to normalize data (default: True)
        upweight: float - factor to upweight fairness violations (default: 1.0)

    Returns:
        result: dict containing final rounded result
    """
  
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    k = int(config["clustering"]["num_clusters"])
    alpha_val = ast.literal_eval(config["params"]["alpha"])
    beta_val = ast.literal_eval(config["params"]["beta"])

    # Normalize data if flag is True
    if normalize_data_flag:
        print("Normalizing data...")
        data_normalized = normalize_data(df.values)
    else:
        print("Using unnormalized data...")
        data_normalized = df.values.copy()

    # Diagnostics
    unique, counts = np.unique(svar_all, return_counts=True)
    total = len(svar_all)
    print("\n[Diagnostics] Global group ratios:")
    for u, c in zip(unique, counts):
        print(f"Group {u}: count={c}, proportion={c/total:.3f}")

    print(f"init_method: {init_method}")
    # Socially fair clustering
    centers = socially_fair_kmeans(data_normalized, svar_all, k=k, init_method=init_method)
    print("\n[Diagnostics] Initial centers from SF k-means:")
    print("Center coordinates:")
    for i, center in enumerate(centers):
        print(f"Center {i}: {center}")
    
    distance_matrix = cdist(data_normalized, centers, metric='sqeuclidean')

    # Run LP solver with upweight factor
    lp_result = run_connector_and_solve_lp(
        df=data_normalized,
        svar_all=svar_all,
        centers=centers,
        alpha=alpha_val,
        beta=beta_val,
        lambda_param=lambda_param,
        upweight=upweight  # Pass the upweight factor
    )

    # Save LP result to file
    # 
    from plot_lp_fractional import plot_lp_fractional_solution

    # After getting LP solution
    #plot_lp_fractional_solution(
    #    x_frac=lp_result['x_frac'],
    #    group_labels=svar_all,
    #    centers=centers,
    #    output_dir="results",
    #    title="My LP Solution",
    #    lambda_param=lambda_param,
    #    data_points=data_normalized  # Pass the data points for clustering plot
    #)

    lp_assignment = lp_result['x_frac']
    lp_objective = lp_result['z']

    # Debug: Print LP result
    group_costs = compute_group_costs_post_lp(
        x_frac=lp_result["x_frac"],
        data_points=data_normalized,
        centers=centers,
        group_labels=svar_all,
        lambda_param=lambda_param,
        upweight=upweight
    )

    for h, costs in group_costs.items():
        print(f"[Group {h}]")
        print(f"  Distance Cost (λ term):   {costs['distance_cost']:.6f}")
        print(f"  Fairness Cost:            {costs['fairness_cost']:.6f}")
        print(f"  Total Cost:               {costs['total_cost']:.6f}")
        x_frac = lp_result["x_frac"]
        z_val = lp_result["z"]

    print(f"\n=== LP RESULT ===")
    print(f"Objective upper bound z = {z_val:.4f}")
    print(f"Assignment matrix shape: {x_frac.shape}")
    print(f"First all rows of fractional assignments:\n{x_frac[:]}")
    cluster_mass = lp_assignment.sum(axis=0)  # shape (k,)
    print(f"\nCluster mass (sum of x_frac over all points): {cluster_mass}")


    print("\n[Diagnostics] Centers after LP solving:")
    print("Center coordinates:")
    for i, center in enumerate(centers):
        print(f"Center {i}: {center}")

    final_result = rounding_wrapper(
        lp_assignment=lp_assignment,
        distance_matrix=distance_matrix,
        color_labels=svar_all,
        num_clusters=centers.shape[0],
        num_colors=len(np.unique(svar_all)),
        lp_objective=lp_objective,
        df=data_normalized,
        centers=centers
    )
    
    # Add centers to final results
    final_result['centers'] = centers  # Use original centers
    final_result['group_names'] = group_names
    
    # Now check if centers changed during the pipeline
    print("\n[Diagnostics] Centers in final result:")
    print("Center coordinates:")
    for i, center in enumerate(final_result['centers']):
        print(f"Center {i}: {center}")
    
    # Verify centers haven't changed
    if not np.allclose(centers, final_result['centers'], rtol=1e-10, atol=1e-10):
        print("\n⚠️ WARNING: Centers have changed during the pipeline!")
        print("Differences in center coordinates:")
        for i in range(len(centers)):
            diff = centers[i] - final_result['centers'][i]
            if not np.allclose(centers[i], final_result['centers'][i], rtol=1e-10, atol=1e-10):
                print(f"Center {i} differences:")
                print(f"  Original: {centers[i]}")
                print(f"  Final:    {final_result['centers'][i]}")
                print(f"  Diff:     {diff}")
    else:
        print("\n✓ Centers remained exactly unchanged throughout the pipeline")
    
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
    use_preloaded_data = outer_config[config_str].getboolean("use_preloaded_data", fallback=False)
    init_method = outer_config[config_str].get("init_method", fallback="random")  # <=== NEW

    # === Choose pipeline execution path ===
    if use_preloaded_data:
        data_matrix, group_labels, df_clean, group_names = load_and_prepare_data(
            config_file=config_file,
            dataset_name=dataset_name,
            max_points=max_points
        )
        result = run_full_pipeline_with_loaded_data(
            df=df_clean,
            svar_all=group_labels,
            group_names=group_names,
            config_file=config_file,
            dataset_name=dataset_name,
            lambda_param=lambda_param,
            init_method=init_method,   # <=== NEW
            normalize_data_flag=True
        )
    else:
        result = run_full_pipeline(
            config_file=config_file,
            dataset_name=dataset_name,
            lambda_param=lambda_param,
            max_points=max_points,
            init_method=init_method   # <=== NEW
        )

    print("Rounded clustering objective:", result['objective'])
    print("Group proportions (normalized):", result['proportions_normalized'])

    config_params = {
        "dataset": dataset_name,
        "config_file": config_file,
        "lambda_param": lambda_param,
        "max_points": max_points,
        "init_method": init_method   # <=== Optional but nice to save it
    }

    save_clustering_results(
        result=result,
        config_params=config_params,
        dataset_name=dataset_name
    )
