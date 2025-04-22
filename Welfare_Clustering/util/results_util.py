import json
import datetime
import os
import numpy as np
from clustering_utils import calculate_welfare_cost

def save_clustering_results(result, config_params, dataset_name, output_dir="results"):
    """
    Save clustering results in a standardized format.
    
    Args:
        result (dict): The clustering results from run_full_pipeline
        config_params (dict): Configuration parameters used
        dataset_name (str): Name of the dataset
        output_dir (str): Directory to save results in
    """
    # Create timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    
    # Create output directory structure
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Calculate welfare cost
    welfare_metrics = calculate_welfare_cost(
        centers=result['centers'],
        assignment=result['assignment'],
        points=result['df'],
        group_labels=result['color_labels'],
        lambda_param=config_params.get('lambda_param', 0.5)
    )
    
    # Prepare output dictionary
    output = {
        # Configuration
        "config": config_params,
        "dataset": dataset_name,
        "timestamp": timestamp,
        "pipeline": "Welfare_Clustering",
        
        # Results
        "objective": result['objective'],
        "proportions_normalized": result['proportions_normalized'].tolist(),
        "centers": [center.tolist() for center in result['centers']],
        "assignment": result['assignment'].tolist(),
        
        # Group information
        "group_names": result['group_names'],  # Only save the actual group names
        
        # Welfare metrics
        "welfare_cost": {
            "max_cost": welfare_metrics['max_welfare_cost'],
            "group_costs": welfare_metrics['group_costs'],
            "distance_costs": welfare_metrics['distance_costs'],
            "fairness_costs": welfare_metrics['fairness_costs']
        },
        
        # Additional metrics
        "lp_objective": result.get('lp_objective', None),
        "group_sizes": result.get('group_sizes', None),
        "cluster_sizes": result.get('cluster_sizes', None)
    }
    
    # Save to JSON file
    filename = f"wc_results_{timestamp}.json"
    filepath = os.path.join(dataset_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {filepath}")
    return filepath 