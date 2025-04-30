import numpy as np
from scipy.spatial.distance import cdist

def evaluate_welfare_cost(centers, assignment, points, group_labels, lambda_param=0.5, p=2):
    """
    Evaluate the welfare cost (D_h) for a given clustering solution.
    This function can be used to evaluate results from any pipeline.
    
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

def evaluate_welfare_cost_with_slack(centers, assignment, points, group_labels, lambda_param, alpha, beta, p=2):
    """
    Extended version of evaluate_welfare_cost.
    Accepts asymmetric group-specific slack (alpha, beta) to define fairness violation.

    Args:
        centers: (k, d) array of cluster centers
        assignment: (n,) array of cluster assignments
        points: (n, d) array of data points
        group_labels: (n,) array of group labels
        lambda_param: float in [0, 1], balance between distance and fairness
        alpha: dict mapping group label -> alpha value (positive slack)
        beta: dict mapping group label -> beta value (negative slack)
        p: 1 (L1) or 2 (L2) distance
    Returns:
        max_welfare_cost: float, maximum D_h across groups
        group_costs: dict mapping group label -> D_h
        cluster_stats: dict containing group proportions and violations for each cluster
        group_distance_costs: dict mapping group label -> average distance cost
        group_fairness_costs: dict mapping group label -> fairness violation cost
    """
    unique_groups = np.unique(group_labels)
    total_points = len(points)
    group_sizes = {h: np.sum(group_labels == h) for h in unique_groups}
    group_proportions = {h: size / total_points for h, size in group_sizes.items()}

    distances = cdist(points, centers, metric='sqeuclidean' if p == 2 else 'cityblock')
    point_distances = np.array([distances[i, assignment[i]] for i in range(len(points))])

    group_costs = {}
    group_distance_costs = {}
    group_fairness_costs = {}
    cluster_stats = {
        'expected_proportions': group_proportions,
        'clusters': {}
    }

    for h in unique_groups:
        group_mask = (group_labels == h)
        
        # Distance cost - sum of distances divided by group size
        group_distances = point_distances[group_mask]
        distance_cost = np.sum(group_distances) / group_sizes[h]
        group_distance_costs[h] = float(distance_cost)

        # Fairness cost with slack
        fairness_violation = 0.0
        for cluster_id in range(len(centers)):
            cluster_mask = (assignment == cluster_id)
            cluster_size = np.sum(cluster_mask)
            if cluster_size == 0:
                continue  # skip empty clusters

            group_in_cluster = np.sum((group_labels == h) & cluster_mask)
            actual_ratio = group_in_cluster / cluster_size
            expected_ratio = group_proportions[h]

            lower_bound = expected_ratio - beta[h]
            upper_bound = expected_ratio + alpha[h]

            if lower_bound <= actual_ratio <= upper_bound:
                violation = 0.0
            else:
                violation = min(abs(lower_bound - actual_ratio), abs(upper_bound - actual_ratio))

            # Scale violation by cluster size and normalize by group size
            fairness_violation += violation * cluster_size / group_sizes[h]

            # Store cluster statistics
            if cluster_id not in cluster_stats['clusters']:
                cluster_stats['clusters'][cluster_id] = {
                    'size': int(cluster_size),
                    'group_proportions': {},
                    'violations': {}
                }
            cluster_stats['clusters'][cluster_id]['group_proportions'][h] = float(actual_ratio)
            cluster_stats['clusters'][cluster_id]['violations'][h] = float(violation)

        # Store raw fairness violation cost
        group_fairness_costs[h] = float(fairness_violation)

        # Combined cost
        D_h = lambda_param * distance_cost + (1 - lambda_param) * fairness_violation
        group_costs[h] = D_h

    max_welfare_cost = max(group_costs.values())
    return max_welfare_cost, group_costs, cluster_stats, group_distance_costs, group_fairness_costs
