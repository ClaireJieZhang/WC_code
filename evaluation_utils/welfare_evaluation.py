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