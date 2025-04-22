import numpy as np
import matplotlib.pyplot as plt
import json
import os
from .welfare_evaluation import evaluate_welfare_cost

# Pipeline colors
PIPELINE_COLORS = {
    'Socially_Fair_Python': 'blue',
    'Welfare_Clustering': 'green',
    'Fair-Clustering-Under-Bounded-Cost': 'red'
}

# Pipeline markers
PIPELINE_MARKERS = {
    'Socially_Fair_Python': 'o',
    'Welfare_Clustering': 's',
    'Fair-Clustering-Under-Bounded-Cost': '^'
}

def plot_welfare_costs_comparison(result_files, dataset_name, k, lambda_param, p, save_path=None):
    """
    Plot welfare costs comparison across different pipelines.
    """
    plt.figure(figsize=(10, 6))
    
    for pipeline_name, result_file in result_files.items():
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        k_results = results['results'][str(k)]
        
        # Plot standard clustering results
        if k_results['standard'] is not None:
            plt.scatter(
                [k], 
                [k_results['standard']['metrics']['welfare_cost']],
                label=f'{pipeline_name} (Standard)',
                marker=PIPELINE_MARKERS.get(pipeline_name, 'o'),
                color=PIPELINE_COLORS.get(pipeline_name, 'gray')
            )
        
        # Plot fair clustering results
        if k_results['fair'] is not None:
            plt.scatter(
                [k], 
                [k_results['fair']['metrics']['welfare_cost']],
                label=f'{pipeline_name} (Fair)',
                marker=PIPELINE_MARKERS.get(pipeline_name, 's'),
                color=PIPELINE_COLORS.get(pipeline_name, 'gray')
            )
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Welfare Cost')
    plt.title(f'Welfare Costs Comparison (λ={lambda_param}, p={p})')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def load_and_plot_results(result_files, dataset_name, k, lambda_param=0.5, p=2, save_path=None):
    """
    Load results from multiple pipeline output files and plot welfare costs.
    
    Args:
        result_files: dict - keys are pipeline names, values are paths to result files
        dataset_name: str - name of the dataset
        k: int - number of clusters
        lambda_param: float - weight between distance and fairness costs
        p: int - distance metric parameter (1 for k-median, 2 for k-means)
        save_path: str - path to save the plot (optional)
    """
    pipeline_results = {}
    
    for pipeline_name, result_file in result_files.items():
        # Load results from file
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Extract necessary components
        pipeline_results[pipeline_name] = {
            'centers': np.array(results['centers']),
            'assignment': np.array(results['assignment']),
            'points': np.array(results['points']),
            'group_labels': np.array(results['group_labels'])
        }
    
    # Plot welfare costs
    plot_welfare_costs_comparison(
        result_files=pipeline_results,
        dataset_name=dataset_name,
        k=k,
        lambda_param=lambda_param,
        p=p,
        save_path=save_path
    )
    
    return pipeline_results

# Example usage
if __name__ == "__main__":
    # Example of how to use the plotting functions
    print("Example of how to use the plotting functions:")
    print("\n1. Direct plotting with results:")
    print("pipeline_results = {")
    print("    'Samira_SF_Python': {'centers': centers1, 'assignment': assignment1, 'points': points, 'group_labels': group_labels},")
    print("    'Welfare_Clustering': {'centers': centers2, 'assignment': assignment2, 'points': points, 'group_labels': group_labels},")
    print("    'Fair-Clustering-Under-Bounded-Cost': {'centers': centers3, 'assignment': assignment3, 'points': points, 'group_labels': group_labels}")
    print("}")
    print("plot_welfare_costs_comparison(pipeline_results, 'adult', k=5, lambda_param=0.5, p=2, save_path='welfare_costs.png')")
    
    print("\n2. Loading results from files and plotting:")
    print("result_files = {")
    print("    'Samira_SF_Python': 'path/to/samira_results.json',")
    print("    'Welfare_Clustering': 'path/to/welfare_results.json',")
    print("    'Fair-Clustering-Under-Bounded-Cost': 'path/to/fair_results.json'")
    print("}")
    print("load_and_plot_results(result_files, 'adult', k=5, lambda_param=0.5, p=2, save_path='welfare_costs.png')") 