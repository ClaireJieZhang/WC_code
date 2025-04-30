import numpy as np
import pandas as pd
from scipy.io import loadmat
import argparse
from collections import defaultdict
from fairlearn.datasets import fetch_adult
from sklearn.preprocessing import OrdinalEncoder
import os, pickle
from scipy.spatial.distance import cdist


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
        svar = svar - np.min(svar)
        if not np.array_equal(np.unique(svar), [0, 1]):
            raise ValueError(f"Expected svar to contain exactly 2 groups, got: {np.unique(svar)}")

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



def hardcoded_initialization(X, k, R=5, r=1, D=10, normalize=True):
    """
    Hardcoded initialization for synthetic dataset with two regions (upper/lower).

    Args:
        X: (n, d) array — input data
        k: int — number of clusters (must be 4 for this synthetic setup)
        R: spread between red/blue centers in upper region
        r: spread between red/blue centers in lower region
        D: vertical separation between regions
        normalize: whether to normalize centers based on dataset X

    Returns:
        centers: (k, d) array of initialized cluster centers
    """
    if k != 4:
        raise ValueError("Hardcoded synthetic initialization expects k=4.")

    # Define hard-coded centers
    centers = np.array([
        [-R, D],  # Red cluster in Region 1 (upper)
        [R, D],   # Blue cluster in Region 1 (upper)
        [-r, 0],  # Red cluster in Region 2 (lower)
        [r, 0]    # Blue cluster in Region 2 (lower)
    ])  # Shape (4, 2)

    if normalize:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        keep = stds != 0  # Only normalize dimensions with nonzero std

        centers_norm = np.zeros_like(centers)
        centers_norm[:, keep] = (centers[:, keep] - means[keep]) / stds[keep]
        return centers_norm
    else:
        return centers


def uniform_box_initialization(X, k):
    """
    Initialize centers uniformly inside the bounding box of the data.

    Args:
        X: (n, d) numpy array
        k: number of clusters

    Returns:
        centers: (k, d) numpy array
    """
    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centers = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
    return centers


def kmeans_plus_plus_initialization(X, k, num_trials=10):
    """
    k-means++ initialization with multiple trials to select the best centers.

    Args:
        X: (n, d) numpy array
        k: number of clusters
        num_trials: number of times to run k-means++ initialization

    Returns:
        centers: (k, d) numpy array - best set of centers found
    """
    n, d = X.shape
    best_centers = None
    best_cost = float('inf')

    for _ in range(num_trials):
        # Initialize centers for this trial
        centers = np.zeros((k, d))
        indices = []

        # 1. Choose the first center randomly
        first_idx = np.random.choice(n)
        centers[0] = X[first_idx]
        indices.append(first_idx)

        # 2. Choose the next centers
        for i in range(1, k):
            dists = np.min(
                np.linalg.norm(X[:, np.newaxis] - centers[np.newaxis, :i], axis=2) ** 2,
                axis=1
            )
            probs = dists / np.sum(dists)
            next_idx = np.random.choice(n, p=probs)
            centers[i] = X[next_idx]
            indices.append(next_idx)

        # Calculate initial cost for this set of centers
        distances = np.min(
            np.linalg.norm(X[:, np.newaxis] - centers[np.newaxis, :], axis=2) ** 2,
            axis=1
        )
        cost = np.sum(distances)

        # Update best centers if this is better
        if cost < best_cost:
            best_cost = cost
            best_centers = centers.copy()

    return best_centers






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
    import time
    min_cost = None
    total_runtime = 0
    centers = None
    clustering = None

    # === Normalize group labels to 0 and 1 ===
    if is_fair:
        svar = svar - np.min(svar)
        if not np.array_equal(np.unique(svar), [0, 1]):
            raise ValueError(f"Expected svar to contain exactly 2 groups, got: {np.unique(svar)}")

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

        start_time = time.time()
        for j in range(num_iters):
            final = (j == num_iters - 1)
            current_clustering = find_clustering(data_temp, ns, current_centers, final, is_fair, verbose)

            if verbose:
                if is_fair == 0:
                    unique, counts = np.unique(current_clustering, return_counts=True)
                    print(f"[Standard Clustering] Cluster counts: {dict(zip(unique, counts))}")
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






def find_clustering(data, ns, centers, is_last, is_fair, verbose=False):
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

        if verbose:
            print("Assigned clusters (standard):", np.unique(cluster_idx, return_counts=True))

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

        if verbose:
            print("Assigned clusters (fair):")
            print("  Group 1:", np.unique(cluster_idx1, return_counts=True))
            print("  Group 2:", np.unique(cluster_idx2, return_counts=True))

        if not is_last:
            clus_num = np.array([np.sum(cluster_temp == i) for i in range(k)])

            for i in range(k):
                if clus_num[i] == 0:
                    sorted_indices = np.argsort(np.sum((data - centers[i]) ** 2, axis=1))
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
    
    print(np.std(X_scaled, axis=0))

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



 

def pre_process_education_vector(vec):
    """
    Maps education levels to binary sensitive attribute:
    - 1 or 2 → 1
    - else   → 2
    """
    vec = np.array(vec).flatten()
    sensitive = np.where((vec == 1) | (vec == 2), 1, 2)
    return sensitive





def calculate_welfare_cost(centers, assignment, points, group_labels, lambda_param=0.5, p=2):
    """
    Calculate D_h for each group and find the maximum.
    
    Args:
        centers: numpy array of shape (k, d) - cluster centers
        assignment: numpy array of shape (n,) - cluster assignments for each point
        points: numpy array of shape (n, d) - data points
        group_labels: numpy array of shape (n,) - group labels for each point
        lambda_param: float in [0,1] - weight between distance and fairness costs
        p: int in {1,2} - 1 for k-median, 2 for k-means
        
    Returns:
        dict containing:
        - max_welfare_cost: maximum D_h across all groups
        - group_costs: dictionary mapping each group to its D_h value
    """
    unique_groups = np.unique(group_labels)
    num_clusters = len(centers)
    
    # Calculate group sizes and proportions
    group_sizes = {h: np.sum(group_labels == h) for h in unique_groups}
    total_points = len(points)
    group_proportions = {h: size/total_points for h, size in group_sizes.items()}
    
    # Calculate distance costs for each point
    distances = cdist(points, centers, metric='sqeuclidean' if p == 2 else 'cityblock')
    point_distances = np.array([distances[i, assignment[i]] for i in range(len(points))])
    
    # Calculate D_h for each group
    group_costs = {}
    for h in unique_groups:
        # Get points in this group
        group_mask = (group_labels == h)
        group_distances = point_distances[group_mask]
        
        # Calculate distance cost component (normalized by group size)
        distance_cost = np.sum(group_distances) / group_sizes[h]
        
        # Calculate fairness cost component
        fairness_cost = 0
        for i in range(num_clusters):
            # Points in cluster i
            cluster_mask = (assignment == i)
            cluster_size = np.sum(cluster_mask)
            if cluster_size == 0:
                continue
                
            # Points of group h in cluster i
            group_in_cluster = np.sum(cluster_mask & group_mask)
            actual_proportion = group_in_cluster / cluster_size
            
            # Calculate violation
            violation = abs(group_proportions[h] - actual_proportion)
            fairness_cost += cluster_size * violation
        
        # Calculate D_h for this group
        D_h = (lambda_param * distance_cost + 
               (1 - lambda_param) * fairness_cost)
        
        group_costs[h] = D_h
    
    max_welfare_cost = max(group_costs.values())
    
    return {
        'max_welfare_cost': max_welfare_cost,
        'group_costs': group_costs
    }
