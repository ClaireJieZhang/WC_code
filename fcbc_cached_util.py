import os
import sys
import time
import numpy as np
import pandas as pd
import copy

# Add FCBC code path
fcbc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fair-Clustering-Under-Bounded-Cost")
sys.path.append(fcbc_path)

# Import FCBC modules
from cplex_fair_assignment_lp_solver_util import fair_partial_assignment_util
from util.clusteringutil import vanilla_clustering
from util.pof_util import get_viol_value_two_color, get_util_value_two_color
from util.utilhelpers import find_proprtions_two_color_deter


def run_fcbc_with_cached_data(
    data_matrix,
    group_labels,
    k,
    deltas,
    pof,
    lambda_param=0.5,
    epsilon=1/(2**7),
    fairness_variable_name="group",
    scaling=True,
    clustering_method="kmeans"
):
    """
    Run FCBC pipeline on cached data.

    Args:
        data_matrix (np.ndarray): Feature matrix
        group_labels (np.ndarray): Group labels for fairness
        k (int): Number of clusters
        deltas (list of float): List of delta values
        pof (float): Price of Fairness parameter
        lambda_param (float): (Unused currently, reserved)
        epsilon (float): Epsilon for LP solving
        fairness_variable_name (str): Name of fairness variable
        scaling (bool): Whether to standardize data
        clustering_method (str): "kmeans" or "kmedian"

    Returns:
        List of output dicts (one per delta)
    """
    print(f"Running FCBC with cached data for k={k}, POF={pof}")

    lower_bound = 0
    p_acc = 1.0
    two_color_util = True

    df = pd.DataFrame(data_matrix)

    fairness_vars = [fairness_variable_name]

    color_flag = {fairness_vars[0]: list(group_labels)}

    representation = {}
    unique_labels = np.unique(group_labels)
    color_proportions = {}
    for label in unique_labels:
        color_proportions[int(label)] = np.sum(group_labels == label) / len(group_labels)
    representation[fairness_vars[0]] = color_proportions

    (_, color_proportions), = representation.items()

    if scaling:
        df = (df - df.mean()) / df.std()

    t1 = time.monotonic()
    initial_score, pred, cluster_centers = vanilla_clustering(df, k, clustering_method)
    t2 = time.monotonic()
    cluster_time = t2 - t1
    print(f"Clustering time: {cluster_time:.4f} seconds")

    sizes = [0 for _ in range(k)]
    for p in pred:
        sizes[p] += 1

    dataset_ratio = {
        fairness_vars[0]: {
            int(label): np.sum(group_labels == label) / len(group_labels)
            for label in unique_labels
        }
    }

    results = []

    for delta in deltas:
        alpha, beta = {}, {}
        if two_color_util:
            a_val, b_val = 1 + delta, 1 - delta
        else:
            a_val, b_val = 1 / (1 - delta), 1 - delta

        for var, bucket_dict in representation.items():
            alpha[var] = {k: a_val * bucket_dict[k] for k in bucket_dict.keys()}
            beta[var] = {k: b_val * bucket_dict[k] for k in bucket_dict.keys()}

        fp_color_flag = {fairness_vars[0]: color_flag[fairness_vars[0]]}
        fp_alpha = {fairness_vars[0]: alpha[fairness_vars[0]]}
        fp_beta = {fairness_vars[0]: beta[fairness_vars[0]]}

        alpha_orig = copy.deepcopy(fp_alpha)
        beta_orig = copy.deepcopy(fp_beta)

        num_colors = len(representation[fairness_vars[0]])

        t1 = time.monotonic()
        res, nf_time, r_min, col_min, r_max, col_max = fair_partial_assignment_util(
            df, cluster_centers, initial_score, delta, color_proportions,
            fp_alpha, fp_beta, fp_color_flag, clustering_method, num_colors,
            lower_bound, epsilon, pof
        )
        t2 = time.monotonic()
        lp_time = t2 - t1

        output = {}

        output["num_clusters"] = k
        output["partial_success"] = res["partial_success"]
        output["dataset_distribution"] = dataset_ratio
        output["prob_proportions"] = representation
        output["alpha"] = alpha_orig
        output["beta"] = beta_orig
        output["unfair_score"] = initial_score
        output["objective"] = res["objective"]
        output["partial_fair_score"] = res["partial_objective"]
        output["sizes"] = sizes
        output["attributes"] = {
            fairness_vars[0]: {
                i: [j for j, l in enumerate(color_flag[fairness_vars[0]]) if l == i]
                for i in np.unique(group_labels)
            }
        }
        output["centers"] = [list(center) for center in cluster_centers]
        output["points"] = [list(point) for point in df.values]
        output["assignment"] = res["assignment"]
        output["partial_assignment"] = res["partial_assignment"]
        output["name"] = "cached_data"
        output["clustering_method"] = clustering_method
        output["scaling"] = scaling
        output["delta"] = delta
        output["time"] = lp_time
        output["cluster_time"] = cluster_time
        output["partial_proportions"] = res["partial_proportions"]
        output["proportions"] = res["proportions"]
        output["partial_proportions_normalized"] = res["partial_proportions_normalized"]
        output["proportions_normalized"] = res["proportions_normalized"]
        output["Cluster_Size_Lower_Bound"] = lower_bound
        output["p_acc"] = p_acc
        output["nf_time"] = nf_time

        viol_upper, viol_lower = get_viol_value_two_color(
            np.reshape(output['proportions_normalized'], (2, -1)),
            alpha_orig,
            beta_orig
        )
        util_objective = get_util_value_two_color(
            np.reshape(output['proportions_normalized'], (2, -1)),
            alpha_orig,
            beta_orig
        )

        rounded_prop, _, _ = find_proprtions_two_color_deter(
            np.reshape(res["assignment"], (-1, k)),
            2, color_flag[fairness_vars[0]], k
        )
        lp_prop, _, _ = find_proprtions_two_color_deter(
            np.reshape(res["partial_assignment"], (-1, k)),
            2, color_flag[fairness_vars[0]], k
        )

        output["util_objective"] = util_objective
        output["bs_iterations"] = res["bs_iterations"]
        output["epsilon"] = epsilon
        output["epsilon set size "] = 1/epsilon
        output["alpha_pof"] = pof
        output["upper_violations"] = viol_upper.ravel().tolist()
        output["lower_violations"] = viol_lower.ravel().tolist()
        output["opt_index"] = res["opt_index"]
        output["util_lp"] = res["util_lp"]
        output["color_flag"] = color_flag[fairness_vars[0]]

        results.append(output)

    return results
