import numpy as np
import pandas as pd
from scipy.io import loadmat
import argparse
from collections import defaultdict
from fairlearn.datasets import fetch_adult
from sklearn.preprocessing import OrdinalEncoder
import os, pickle
import sys


# Add the parent directory to the path so we can import from evaluation_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def give_rand_centers(data, data_pf, data_n, k, best_out_of, seed=None):
    """
    Generate multiple sets of random initial centers for three datasets.

    Parameters:
    - data: numpy array (n, d)
    - data_pf: numpy array (n, d) - fair PCA
    - data_n: numpy array (n, dp) - normalized only
    - k: number of cluster centers
    - best_out_of: how many random sets to generate
    - seed: optional int for reproducibility

    Returns:
    - rand_centers: array (best_out_of, k, d)
    - rand_centers_pf: array (best_out_of, k, d)
    - rand_centers_n: array (best_out_of, k, dp)
    """
    if seed is not None:
        np.random.seed(seed)

    n, d = data.shape
    dp = data_n.shape[1]

    rand_centers = np.zeros((best_out_of, k, d))
    rand_centers_pf = np.zeros((best_out_of, k, d))
    rand_centers_n = np.zeros((best_out_of, k, dp))

    for i in range(best_out_of):
        perm = np.random.permutation(n)[:k]
        rand_centers[i] = data[perm]
        rand_centers_pf[i] = data_pf[perm]
        rand_centers_n[i] = data_n[perm]

    return rand_centers, rand_centers_pf, rand_centers_n






def comp_cost(data, svar, k, clustering, is_fair):
    """
    Computes per-group clustering costs for fair or standard setting.
    Uses cluster centers from find_centers and cost from kmeans_cost_s_c.
    """
    costs = np.zeros(2)

    if is_fair == 0:
        g1 = (svar == 0)
        g2 = (svar == 1)

        size1 = np.sum(g1)
        size2 = np.sum(g2)

        if size1 == 0 or size2 == 0:
            print("⚠️ One of the groups has no points. Returning NaN costs.")
            return np.array([np.nan, np.nan])

        centers = find_centers(data, svar, k, clustering, is_fair)

        costs[0] = kmeans_cost_s_c(data[g1], clustering[g1], centers, flag=1) / (size1 + 1e-8)
        costs[1] = kmeans_cost_s_c(data[g2], clustering[g2], centers, flag=1) / (size2 + 1e-8)

    else:
        data1, data2 = data
        clustering1, clustering2 = clustering
        size1 = data1.shape[0]
        size2 = data2.shape[0]

        if size1 == 0 or size2 == 0:
            print("⚠️ One of the fair groups has no points. Returning NaN costs.")
            return np.array([np.nan, np.nan])

        centers = find_centers(data, svar, k, clustering, is_fair)

        costs[0] = kmeans_cost_s_c(data1, clustering1, centers, flag=1) / (size1 + 1e-8)
        costs[1] = kmeans_cost_s_c(data2, clustering2, centers, flag=1) / (size2 + 1e-8)

    return costs






def find_centers(data, svar, k, clustering, is_fair):
    """
    Compute cluster centers, optionally with fairness interpolation.

    Parameters:
    - data: if is_fair == 0, ndarray (n, d); if 1, list of two arrays [data1, data2]
    - svar: group labels or ignored if is_fair == 1
    - k: number of clusters
    - clustering: array (is_fair == 0) or list of two arrays (is_fair == 1)
    - is_fair: 0 or 1

    Returns:
    - centers: ndarray of shape (k, d)
    """
    if is_fair == 0:
        n, d = data.shape
        centers = np.full((k, d), np.inf)
        for i in range(k):
            members = (clustering == i)
            if np.sum(members) > 0:
                centers[i] = np.mean(data[members], axis=0)
        return centers

    else:
        data1, data2 = data
        clustering1, clustering2 = clustering
        n1, d = data1.shape
        n2 = data2.shape[0]

        muA = np.full((k, d), np.inf)
        muB = np.zeros((k, d))
        alphaA = np.zeros(k)
        alphaB = np.zeros(k)
        l = np.zeros(k)
        deltaA = 0.0
        deltaB = 0.0

        for i in range(k):
            cluster_data1 = data1[clustering1 == i]
            cluster_data2 = data2[clustering2 == i]
            alphaA[i] = len(cluster_data1) / n1 if n1 else 0
            alphaB[i] = len(cluster_data2) / n2 if n2 else 0

            if alphaA[i] + alphaB[i] != 0:
                if alphaA[i] == 0 or alphaB[i] == 0:
                    combined = np.vstack([cluster_data1, cluster_data2])
                    mu = np.mean(combined, axis=0)
                    muA[i] = mu
                    muB[i] = mu
                else:
                    muA[i] = np.mean(cluster_data1, axis=0)
                    muB[i] = np.mean(cluster_data2, axis=0)

                l[i] = np.linalg.norm(muA[i] - muB[i])

                if len(cluster_data1) > 0:
                    deltaA += np.sum((cluster_data1 - muA[i])**2)
                if len(cluster_data2) > 0:
                    deltaB += np.sum((cluster_data2 - muB[i])**2)

        deltaA /= n1
        deltaB /= n2

        # Directly call global b_search
        cost, x = b_search(deltaA, deltaB, alphaA, alphaB, l)

        centers = np.full((k, d), np.inf)
        for i in range(k):
            if l[i] == 0:
                centers[i] = muA[i]
            else:
                centers[i] = ((l[i] - x[i]) * muA[i] + x[i] * muB[i]) / l[i]

        return centers





import time

def lloyd(data, svar, k, num_iters, best_out_of, rand_centers, is_fair, verbose=False):
    """
    Run Fair-Lloyd or standard Lloyd algorithm with multiple initializations.

    Parameters:
    - data: if is_fair == 0: ndarray; if 1: list of two arrays [group1_data, group2_data]
    - svar: sensitive attribute values (1 or 2), shape (n,)
    - k: number of clusters
    - num_iters: number of Lloyd steps per run
    - best_out_of: how many random initializations to try
    - rand_centers: array of shape (best_out_of, k, d)
    - is_fair: 0 (unfair) or 1 (fair)
    - verbose: whether to print progress and warnings

    Returns:
    - centers: ndarray of shape (k, d)
    - clustering: cluster assignments (array or list of arrays)
    - runtime: average runtime across initializations
    """
    min_cost = None
    total_runtime = 0
    centers = None
    clustering = None

    # Separate data by group if fair
    if is_fair:
        data_sep = [data[svar == 0], data[svar == 1]]
        data_temp = np.vstack(data_sep)
        ns = [len(data_sep[0]), len(data_sep[1])]
        data_input = data_sep
    else:
        data_temp = data
        ns = [1, 1]
        data_input = data

    for i in range(best_out_of):
        if verbose:
            print(f"Initialization {i + 1}/{best_out_of}")
        current_centers = rand_centers[i]

        if verbose:
            print(f"[DEBUG] rand_centers[{i}] shape = {current_centers.shape}")
            print(f"[DEBUG] rand_centers[{i}]:\n{current_centers}")


        start_time = time.time()
        for j in range(num_iters):
            final = (j == num_iters - 1)
            current_clustering = find_clustering(data_temp, ns, current_centers, final, is_fair)
            if is_fair == 0:
                unique, counts = np.unique(current_clustering, return_counts=True)
                print(f"[Clustering] Cluster counts: {dict(zip(unique, counts))}")
            else:
                all_clusters = np.concatenate(current_clustering)
                unique, counts = np.unique(all_clusters, return_counts=True)
                print(f"[Fair Clustering] Cluster counts: {dict(zip(unique, counts))}")


            if not final:
                current_centers = find_centers(data_input, svar, k, current_clustering, is_fair)

        runtime = time.time() - start_time
        total_runtime += runtime

        current_cost = comp_cost(data_input, svar, k, current_clustering, is_fair)

        if not np.isnan(current_cost).any():
            if min_cost is None or np.sum(current_cost) < np.sum(min_cost):
                min_cost = current_cost
                centers = current_centers
                clustering = current_clustering

    if clustering is None:
        if verbose:
            print("⚠️ No initialization produced a valid clustering. Using last available.")
        clustering = current_clustering
        centers = current_centers

    avg_runtime = total_runtime / best_out_of
    return centers, clustering, avg_runtime






def find_clustering(data, ns, centers, is_last, is_fair):
    """
    Assign points to clusters based on closest center.
    Prevents empty clusters if is_last == False.

    Parameters:
    - data: (n, d) array of all points
    - ns: [n1, n2] sizes of group1 and group2 (used only if is_fair == 1)
    - centers: (k, d) array of cluster centers
    - is_last: True if final iteration (no need to fix empty clusters)
    - is_fair: 0 or 1

    Returns:
    - cluster_idx: if is_fair == 0, (n,) array;
                   if is_fair == 1, [array of n1 assignments, array of n2 assignments]
    """
    k = centers.shape[0]
    n = data.shape[0]

    # Compute distances (squared Euclidean)
    dists = np.zeros((k, n))
    for i in range(k):
        dists[i] = np.sum((data - centers[i]) ** 2, axis=1)
    cluster_temp = np.argmin(dists, axis=0)

    if is_fair == 0:
        cluster_idx = cluster_temp.copy()

        # Fix empty clusters if not last iteration
        if not is_last:
            clus_num = np.array([np.sum(cluster_idx == i) for i in range(k)])
            for i in range(k):
                if clus_num[i] == 0:
                    sorted_indices = np.argsort(np.sum((data - centers[i]) ** 2, axis=1))
                    for j in sorted_indices:
                        temp = cluster_idx[j]
                        if clus_num[temp] > 1:
                            clus_num[i] += 1
                            clus_num[temp] -= 1
                            cluster_idx[j] = i
                            break
        return cluster_idx

    else:
        n1 = ns[0]
        cluster_idx1 = cluster_temp[:n1].copy()
        cluster_idx2 = cluster_temp[n1:].copy()

        if not is_last:
            clus_num = np.array([np.sum(cluster_temp == i) for i in range(k)])
            sorted_indices = np.argsort(np.sum((data - centers[i]) ** 2, axis=1))

            for i in range(k):
                if clus_num[i] == 0:
                    for j in sorted_indices:
                        if j < n1:
                            temp = cluster_idx1[j]
                            tempi = 1
                        else:
                            temp = cluster_idx2[j - n1]
                            tempi = 2

                        if clus_num[temp] > 1:
                            clus_num[i] += 1
                            clus_num[temp] -= 1
                            if tempi == 1:
                                cluster_idx1[j] = i
                            else:
                                cluster_idx2[j - n1] = i
                            break

        return [cluster_idx1, cluster_idx2]








def normalize_data(X):
    """
    Normalize data to zero mean and unit variance.
    Removes columns with zero standard deviation.

    Parameters:
    - X: numpy array of shape (n_samples, n_features)

    Returns:
    - X_norm: normalized data with constant features removed
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    # Avoid divide-by-zero: only keep columns with non-zero std
    keep = stds != 0
    X_centered = X - means
    X_scaled = X_centered[:, keep] / stds[keep]
    


    return X_scaled




  

def kmeans_cost_s_c(data, clustering, centers, flag=0):
    """
    Compute k-means clustering cost given data, clustering, and centers.

    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - clustering: 1D array of length n_samples, with cluster indices (0-indexed)
    - centers: numpy array of shape (k, n_features)
    - flag: 
        - 0: sum of squared distances to centers
        - 1: uses formula based on intra-cluster variance + center offset

    Returns:
    - cost: float, total clustering cost
    """
    k = centers.shape[0]
    cost = 0.0

    for i in range(k):
        mask = (clustering == i)

        if np.sum(mask) == 0:
            print(f"⚠️ Cluster {i} is empty. Skipping it.")
            continue  # ✅ safely skip empty clusters

        cluster_points = data[mask]
        cluster_size = cluster_points.shape[0]

        if flag == 0:
            diffs = cluster_points - centers[i]
            cluster_cost = np.sum(np.square(diffs))
        else:
            cluster_mean = np.mean(cluster_points, axis=0)
            intra_var = np.sum(np.square(cluster_points - cluster_mean))
            center_offset = cluster_size * np.linalg.norm(cluster_mean - centers[i])**2
            cluster_cost = intra_var + center_offset

        cost += cluster_cost

    return cost





def b_search(deltaA, deltaB, alphaA, alphaB, l, tol=1e-10, max_iter=64):
    """
    Binary search over gamma to balance fairness cost between two groups.

    Parameters:
    - deltaA, deltaB: intra-cluster variances for group A and B
    - alphaA, alphaB: cluster-wise group proportions (length k)
    - l: vector of distances between group means (length k)
    - tol: stopping tolerance
    - max_iter: maximum number of iterations

    Returns:
    - cost: final max(f, g)
    - x: vector of interpolation weights for group A
    """
    gamma = 0.5
    gamma_low = 1.0
    gamma_high = 0.0

    for _ in range(max_iter):
        denom = gamma * alphaA + (1 - gamma) * alphaB
        x = (1 - gamma) * alphaB * l / denom

        f = deltaA + np.sum(alphaA * x**2)
        g = deltaB + np.sum(alphaB * (l - x)**2)

        cost = max(f, g)

        if abs(f - g) < tol:
            break
        elif f > g:
            gamma_high = gamma
            gamma = (gamma + gamma_low) / 2
        else:
            gamma_low = gamma
            gamma = (gamma + gamma_high) / 2

    return cost, x




def naive_local_load_data(dataset_name):
    """
    Load dataset features and sensitive attribute for fairness-aware clustering.

    Parameters:
    - dataset_name: one of {'credit', 'adult', 'LFW', 'compasWB'}

    Returns:
    - data_all: numpy array of shape (n_samples, n_features)
    - svar_all: numpy array of sensitive attributes (values: 1 or 2)
    - group_names: list of group name strings
    """
    if dataset_name == 'credit':
        data_all = np.loadtxt('../Data/credit/credit_degree.csv', delimiter=',', skiprows=2, usecols=range(1, None))
        svar_temp = np.loadtxt('../Data/credit/educationAttribute.csv')
        svar_all = pre_process_education_vector(svar_temp).astype(int)
        group_names = ['Higher Education', 'Lower Education']

    elif dataset_name == 'adult':
        # Use the common data loader for adult dataset
        data_all, svar_all, group_names = load_adult_data()
        
        print("First 5 rows of data_all:")
        print(data_all[:5])
        
        print("Any NaNs in data_all?", np.isnan(data_all).any())  # should be False

    elif dataset_name == 'LFW':
        mat_data = loadmat('../Data/LFW/LFW.mat')
        data_all = mat_data['data']
        svar_all = mat_data['sensitive'].flatten()
        group_names = ['Female', 'Male']

    elif dataset_name == 'compasWB':
        mat_data = loadmat('../Data/compas/compas-data.mat')
        race = mat_data['svarRace'].flatten()
        mask = (race == 1) | (race == 3)
        data_all = mat_data['dataCompas'][mask]
        svar_all = ((race[mask] - 1) // 2 + 1).astype(int)
        race_names = [str(r[0]) if isinstance(r, np.ndarray) else r for r in mat_data['raceNames'][0]]
        group_names = [race_names[0], race_names[2]]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return data_all, svar_all, group_names




 

def pre_process_education_vector(vec):
    """
    Maps education levels to binary sensitive attribute:
    - 1 or 2 → 1
    - else   → 2
    """
    vec = np.array(vec).flatten()
    sensitive = np.where((vec == 1) | (vec == 2), 1, 2)
    return sensitive






def run_pipeline(dataset_name, k_min, k_max, num_iters, best_out_of, verbose=False):
    np.random.seed(12345)
    data_all, svar_all, group_names = naive_local_load_data(dataset_name)
    data_normalized = normalize_data(data_all)
    print("First 5 normalized rows:")
    print(data_normalized[:5])


    print("Any NaNs in data_normalized?", np.isnan(data_normalized).any())


    results = defaultdict(dict)

    for k in range(k_min, k_max + 1):
        if verbose:
            print(f"\n=== k = {k} ===")
        rand_centers, _, _ = give_rand_centers(
            data_normalized, data_normalized, data_normalized, k, best_out_of
        )

        # Unfair Lloyd
        centers, clustering, runtime = lloyd(
            data_normalized, svar_all, k, num_iters, best_out_of, rand_centers, is_fair=0, verbose=verbose
        )

        cost = comp_cost(data_normalized, svar_all, k, clustering, is_fair=0)

        results[k]['centers'] = centers
        results[k]['clustering'] = clustering
        results[k]['runtime'] = runtime
        results[k]['cost'] = cost

        # Fair Lloyd
        centers_f, clustering_f, runtime_f = lloyd(
            data_normalized, svar_all, k, num_iters, best_out_of, rand_centers, is_fair=1, verbose=verbose
        )

        cost_f = comp_cost(
            [data_normalized[svar_all == 0], data_normalized[svar_all == 1]],
            svar_all, k, clustering_f, is_fair=1
        )

        results[k]['centers_f'] = centers_f
        results[k]['clustering_f'] = clustering_f
        results[k]['runtime_f'] = runtime_f
        results[k]['cost_f'] = cost_f

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{dataset_name}_k{k_min}-{k_max}_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✅ Results saved to: {out_path}")
    return results


def run_sf_pipeline_with_loaded_data(
    data_all,
    svar_all,
    dataset_name,
    k_min=4,
    k_max=15,
    num_iters=10,
    best_out_of=10,
    verbose=False
):
    """
    Run Fair-Lloyd pipeline using pre-loaded data and group labels.
    Saves results as in original run_pipeline.
    """

    print("[DEBUG] unique group labels:", np.unique(svar_all, return_counts=True))
    print("[DEBUG] data_all shape:", data_all.shape)
    print("[DEBUG] svar_all dtype:", svar_all.dtype)
    print("[DEBUG] svar_all unique values:", np.unique(svar_all))

    data_normalized = normalize_data(data_all)
    results = defaultdict(dict)

    for k in range(k_min, k_max + 1):
        if verbose:
            print(f"\n=== k = {k} ===")

        rand_centers, _, _ = give_rand_centers(
            data_normalized, data_normalized, data_normalized, k, best_out_of
        )

        centers, clustering, runtime = lloyd(
            data_normalized, svar_all, k, num_iters, best_out_of, rand_centers, is_fair=0, verbose=verbose
        )
        cost = comp_cost(data_normalized, svar_all, k, clustering, is_fair=0)

        centers_f, clustering_f, runtime_f = lloyd(
            data_normalized, svar_all, k, num_iters, best_out_of, rand_centers, is_fair=1, verbose=verbose
        )
        cost_f = comp_cost(
            [data_normalized[svar_all == 0], data_normalized[svar_all == 1]],
            svar_all, k, clustering_f, is_fair=1
        )

        results[k]['centers'] = centers
        results[k]['clustering'] = clustering
        results[k]['runtime'] = runtime
        results[k]['cost'] = cost

        results[k]['centers_f'] = centers_f
        results[k]['clustering_f'] = clustering_f
        results[k]['runtime_f'] = runtime_f
        results[k]['cost_f'] = cost_f

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{dataset_name}_k{k_min}-{k_max}_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✅ Results saved to: {out_path}")
    return results


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Run Fair-Lloyd clustering pipeline.")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "credit", "LFW", "compasWB"],
                        help="Dataset name to use.")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of clusters.")
    parser.add_argument("--k_max", type=int, default=15, help="Maximum number of clusters.")
    parser.add_argument("--iters", type=int, default=10, help="Number of Lloyd iterations.")
    parser.add_argument("--best_out_of", type=int, default=10, help="Number of random initializations to try.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress logs.")
    parser.add_argument("--use_preloaded_data", action="store_true",
                        help="Use shared data loader from evaluation_utils instead of internal dataset loading.")

    args = parser.parse_args()

    if args.use_preloaded_data:
        config_file = "configs/standard_run.ini"
        data_matrix, group_labels, _, _ = load_data(config_file, dataset_name=args.dataset)

        results = run_sf_pipeline_with_loaded_data(
            data_all=data_matrix,
            svar_all=group_labels,
            dataset_name=args.dataset,
            k_min=args.k_min,
            k_max=args.k_max,
            num_iters=args.iters,
            best_out_of=args.best_out_of,
            verbose=args.verbose
        )
    else:
        results = run_pipeline(
            dataset_name=args.dataset,
            k_min=args.k_min,
            k_max=args.k_max,
            num_iters=args.iters,
            best_out_of=args.best_out_of,
            verbose=args.verbose
        )
