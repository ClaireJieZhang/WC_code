import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial

# Add the parent directory to the path so we can import from FCBC
fcbc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fair-Clustering-Under-Bounded-Cost")
sys.path.append(fcbc_path)

# Import necessary FCBC modules
from cplex_fair_assignment_lp_solver_util import fair_partial_assignment_util
from util.clusteringutil import vanilla_clustering
from util.probutil import form_class_prob_vector, sample_colors, create_prob_vecs, sample_colors_ml_model
from util.pof_util import relax_alpha_viol, relax_beta_viol, get_color_with_min_proportion, relax_util_viol, alpha_beta_array, get_viol_value_two_color, get_util_value_two_color
from util.utilhelpers import get_clust_sizes, max_Viol_multi_color, x_for_colorBlind, max_RatioViol_multi_color, find_proprtions_two_color_deter 


def run_fcbc_with_cached_data(data_matrix, group_labels, k, deltas, pof, lambda_param=0.5, epsilon=1/(2**7)):
    """
    Run FCBC with cached data, avoiding unnecessary parsing.
    
    Args:
        data_matrix: numpy array - Preprocessed data matrix
        group_labels: numpy array - Group labels
        k: int - Number of clusters
        deltas: list - Delta values for FCBC
        pof: float - POF value
        lambda_param: float - Lambda parameter for welfare cost calculation
        epsilon: float - Epsilon value for FCBC
        
    Returns:
        dict - Results for this k and POF
    """
    print(f"Running FCBC with cached data for k={k}, POF={pof}")
    
    # Set up FCBC parameters
    lower_bound = 0
    p_acc = 1.0
    ml_model_flag = False
    two_color_util = True
    
    # Create a DataFrame from the data matrix
    df = pd.DataFrame(data_matrix)
    
    # Create color flag dictionary
    color_flag = {'fairness_variable': group_labels}
    
    # Calculate representation
    representation = {}
    unique_labels = np.unique(group_labels)
    
    # Calculate proportions for each color
    color_proportions = {}
    for label in unique_labels:
        color_proportions[int(label)] = np.sum(group_labels == label) / len(group_labels)
    
    representation['fairness_variable'] = color_proportions
    
    # Get the color proportions for the first (and only) variable
    (_, color_proportions), = representation.items()
    
    # Scale data if needed (using the same scaling as in FCBC)
    scaling = True
    if scaling:
        df = (df - df.mean()) / df.std()
    
    # Use vanilla clustering as the base method
    clustering_method = "kmeans"
    
    # Run vanilla clustering
    t1 = time.monotonic()
    initial_score, pred, cluster_centers = vanilla_clustering(df, k, clustering_method)
    t2 = time.monotonic()
    cluster_time = t2 - t1
    print("Clustering time: {}".format(cluster_time))
    
    # Calculate cluster sizes
    sizes = [0 for _ in range(k)]
    for p in pred:
        sizes[p] += 1
    
    # Calculate dataset ratio
    dataset_ratio = {}
    for attr, color_dict in representation.items():
        dataset_ratio[attr] = {int(color): proportion for color, proportion in color_dict.items()}
    
    # Set up fairness variables
    fairness_vars = ['fairness_variable']
    
    # Process each delta value
    for delta in deltas:
        # Calculate alpha and beta values
        alpha, beta = {}, {}
        if two_color_util:
            a_val, b_val = 1 + delta, 1 - delta
        else:
            a_val, b_val = 1 / (1 - delta), 1 - delta
        
        for var, bucket_dict in representation.items():
            alpha[var] = {k: a_val * representation[var][k] for k in bucket_dict.keys()}
            beta[var] = {k: b_val * representation[var][k] for k in bucket_dict.keys()}
        
        # Get the first (and only) variable's values
        fp_color_flag = color_flag['fairness_variable']
        fp_alpha = alpha['fairness_variable']
        fp_beta = beta['fairness_variable']
        
        # Make copies for later use
        alpha_orig = fp_alpha.copy()
        beta_orig = fp_beta.copy()
        
        # Get the number of colors
        num_colors = max(color_proportions.keys()) + 1
        
        # Run fair partial assignment
        t1 = time.monotonic()
        res, nf_time, r_min, col_min, r_max, col_max = fair_partial_assignment_util(
            df, cluster_centers, initial_score, delta, color_proportions, 
            fp_alpha, fp_beta, fp_color_flag, clustering_method, num_colors, 
            lower_bound, epsilon, pof
        )
        t2 = time.monotonic()
        lp_time = t2 - t1
        
        # Create output dictionary
        output = {}
        
        # Basic information
        output["num_clusters"] = k
        output["partial_success"] = res["partial_success"]
        output["dataset_distribution"] = dataset_ratio
        output["prob_proportions"] = representation
        output["alpha"] = alpha
        output["beta"] = beta
        output["unfair_score"] = initial_score
        output["objective"] = res["objective"]
        output["partial_fair_score"] = res["partial_objective"]
        output["sizes"] = sizes
        output["attributes"] = {var: {i: [j for j, l in enumerate(color_flag[var]) if l == i] for i in np.unique(color_flag[var])} for var in fairness_vars}
        
        # Clustering results
        output["centers"] = [list(center) for center in cluster_centers]
        output["points"] = [list(point) for point in df.values]
        output["assignment"] = res["assignment"]
        output["partial_assignment"] = res["partial_assignment"]
        
        # Metadata
        output["name"] = "cached_data"
        output["clustering_method"] = clustering_method
        output["scaling"] = scaling
        output["delta"] = delta
        output["time"] = lp_time
        output["cluster_time"] = cluster_time
        
        # Proportions
        output['partial_proportions'] = res['partial_proportions']
        output['proportions'] = res['proportions']
        output['partial_proportions_normalized'] = res['partial_proportions_normalized']
        output['proportions_normalized'] = res['proportions_normalized']
        
        # Additional parameters
        output['Cluster_Size_Lower_Bound'] = lower_bound
        output['p_acc'] = p_acc
        output['nf_time'] = nf_time
        
        # Calculate violations and utility
        viol_upper, viol_lower = get_viol_value_two_color(
            np.reshape(output['proportions_normalized'], (2, -1)), 
            alpha_orig, beta_orig
        )
        util_objective = get_util_value_two_color(
            np.reshape(output['proportions_normalized'], (2, -1)), 
            alpha_orig, beta_orig
        )
        
        # Calculate rounded proportions
        rounded_prop, _, _ = find_proprtions_two_color_deter(
            np.reshape(res["assignment"], (-1, k)), 
            2, fp_color_flag, k
        )
        lp_prop, _, _ = find_proprtions_two_color_deter(
            np.reshape(res["partial_assignment"], (-1, k)), 
            2, fp_color_flag, k
        )
        
        # Add final metrics
        output['util_objective'] = util_objective
        output["bs_iterations"] = res['bs_iterations']
        output["epsilon"] = epsilon
        output["epsilon set size "] = 1/epsilon
        output["alpha_pof"] = pof
        output['upper_violations'] = viol_upper.ravel().tolist()
        output['lower_violations'] = viol_lower.ravel().tolist()
        output['opt_index'] = res['opt_index']
        output['util_lp'] = res['util_lp']
        output['color_flag'] = fp_color_flag
        
        # Return the output
        return output 