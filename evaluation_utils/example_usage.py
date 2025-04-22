import numpy as np
import json
import os
from welfare_evaluation import evaluate_welfare_cost
from evaluation_utils.plotting import plot_welfare_costs_comparison
from evaluation_utils.standardize_results import standardize_and_save_results

def load_results_from_pipeline(pipeline_name, result_file):
    """
    Load results from a specific pipeline's output file.
    
    Args:
        pipeline_name: str - name of the pipeline ('Samira_SF_Python', 'Welfare_Clustering', or 'Fair-Clustering-Under-Bounded-Cost')
        result_file: str - path to the result file
        
    Returns:
        dict - loaded results
    """
    with open(result_file, 'r') as f:
        results = json.load(f)
    return results

def evaluate_pipeline_results(pipeline_name, result_file, data, group_labels, lambda_param=0.5, p=2):
    """
    Evaluate welfare cost for results from any pipeline.
    
    Args:
        pipeline_name: str - name of the pipeline
        result_file: str - path to the result file
        data: numpy array - original data points
        group_labels: numpy array - group labels for each point
        lambda_param: float - weight between distance and fairness costs
        p: int - distance metric parameter (1 for k-median, 2 for k-means)
        
    Returns:
        dict - welfare evaluation results
    """
    # Load results from the pipeline
    results = load_results_from_pipeline(pipeline_name, result_file)
    
    # Extract necessary components based on pipeline
    if pipeline_name == 'Samira_SF_Python':
        centers = np.array(results['centers'])
        assignment = np.array(results['assignment'])
    elif pipeline_name == 'Welfare_Clustering':
        centers = np.array(results['centers'])
        assignment = np.array(results['assignment'])
    elif pipeline_name == 'Fair-Clustering-Under-Bounded-Cost':
        centers = np.array(results['centers'])
        assignment = np.array(results['assignment'])
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    
    # Evaluate welfare cost
    welfare_metrics = evaluate_welfare_cost(
        centers=centers,
        assignment=assignment,
        points=data,
        group_labels=group_labels,
        lambda_param=lambda_param,
        p=p
    )
    
    return welfare_metrics

# Example usage
if __name__ == "__main__":
    # This is just an example - you would need to provide actual data and result files
    print("Example of how to use the welfare evaluation function:")
    print("\n1. Import the function:")
    print("from evaluation_utils.welfare_evaluation import evaluate_welfare_cost")
    
    print("\n2. Load your data and results:")
    print("data = np.load('your_data.npy')")
    print("group_labels = np.load('your_group_labels.npy')")
    print("results = load_results_from_pipeline('Welfare_Clustering', 'path/to/results.json')")
    
    print("\n3. Evaluate welfare cost:")
    print("welfare_metrics = evaluate_welfare_cost(")
    print("    centers=results['centers'],")
    print("    assignment=results['assignment'],")
    print("    points=data,")
    print("    group_labels=group_labels,")
    print("    lambda_param=0.5,")
    print("    p=2")
    print(")")
    
    print("\n4. Access the results:")
    print("max_cost = welfare_metrics['max_welfare_cost']")
    print("group_costs = welfare_metrics['group_costs']")

    # Example of plotting
    print("\nExample of plotting:")
    print("from evaluation_utils.plotting import plot_welfare_costs_comparison")
    
    print("\nPrepare your results:")
    print("pipeline_results = {")
    print("    'Samira_SF_Python': {'centers': centers1, 'assignment': assignment1, 'points': points, 'group_labels': group_labels},")
    print("    'Welfare_Clustering': {'centers': centers2, 'assignment': assignment2, 'points': points, 'group_labels': group_labels},")
    print("    'Fair-Clustering-Under-Bounded-Cost': {'centers': centers3, 'assignment': assignment3, 'points': points, 'group_labels': group_labels}")
    print("}")
    
    print("\nPlot the results:")
    print("plot_welfare_costs_comparison(")
    print("    pipeline_results=pipeline_results,")
    print("    dataset_name='adult',")
    print("    k=5,")
    print("    lambda_param=0.5,")
    print("    p=2,")
    print("    save_path='welfare_costs.png')")

    # Example of running the command-line script
    print("\nExample of running the command-line script:")
    print("python evaluation_utils/run_comparison.py")
    print("    --dataset adult")
    print("    --k 5")
    print("    --lambda_param 0.5")
    print("    --p 2")
    print("    --samira_results path/to/samira_results.json")
    print("    --welfare_results path/to/welfare_results.json")
    print("    --fair_results path/to/fair_results.json")
    print("    --output_dir plots")

    # Example of standardized results
    print("\nExample of standardized results:")
    print("{")
    print("    \"pipeline\": \"Pipeline_Name\",")
    print("    \"dataset\": \"dataset_name\",")
    print("    \"timestamp\": \"ISO format timestamp\",")
    print("    \"data\": {")
    print("        \"points\": [...],")
    print("        \"group_labels\": [...]")
    print("    },")
    print("    \"results\": {")
    print("        k: {")
    print("            \"standard\": {")
    print("                \"centers\": [...],")
    print("                \"assignment\": [...],")
    print("                \"metrics\": {")
    print("                    \"welfare_cost\": {...},")
    print("                    \"runtime\": float,")
    print("                    \"group_costs\": [...]")
    print("                }")
    print("            },")
    print("            \"fair\": {")
    print("                ... ")
    print("            }")
    print("        }")
    print("    }")
    print("}")

    # Step 1: Load your data and group labels
    data = np.load('path/to/your_data.npy')
    group_labels = np.load('path/to/your_group_labels.npy')

    # Step 2: Define paths for original and standardized results
    original_results = {
        'Samira_SF_Python': 'path/to/samira_results.pkl',
        'Welfare_Clustering': 'path/to/welfare_results.json',
        'Fair-Clustering-Under-Bounded-Cost': 'path/to/fcbc_results.pkl'
    }

    standardized_dir = 'standardized_results'
    os.makedirs(standardized_dir, exist_ok=True)

    standardized_results = {
        'Samira_SF_Python': os.path.join(standardized_dir, 'samira_standardized.json'),
        'Welfare_Clustering': os.path.join(standardized_dir, 'welfare_standardized.json'),
        'Fair-Clustering-Under-Bounded-Cost': os.path.join(standardized_dir, 'fcbc_standardized.json')
    }

    # Step 3: Standardize results from each pipeline
    for pipeline_name, result_file in original_results.items():
        if os.path.exists(result_file):
            print(f"Standardizing {pipeline_name} results...")
            standardize_and_save_results(
                pipeline_name=pipeline_name,
                result_file=result_file,
                output_file=standardized_results[pipeline_name],
                data=data,
                group_labels=group_labels,
                lambda_param=0.5,
                p=2
            )

    # Step 4: Load standardized results for plotting
    pipeline_results = {}
    for pipeline_name, result_file in standardized_results.items():
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                standardized = json.load(f)
                
                # Extract data for a specific k value (e.g., k=5)
                k = 5
                if k in standardized['results']:
                    result = standardized['results'][k]
                    
                    # Use fair clustering results if available, otherwise standard
                    if result['fair'] is not None:
                        clustering_result = result['fair']
                    else:
                        clustering_result = result['standard']
                    
                    pipeline_results[pipeline_name] = {
                        'centers': np.array(clustering_result['centers']),
                        'assignment': np.array(clustering_result['assignment']),
                        'points': np.array(standardized['data']['points']),
                        'group_labels': np.array(standardized['data']['group_labels'])
                    }

    # Step 5: Plot welfare costs comparison
    if pipeline_results:
        plot_welfare_costs_comparison(
            pipeline_results=pipeline_results,
            dataset_name='your_dataset',
            k=5,
            lambda_param=0.5,
            p=2,
            save_path='welfare_costs_comparison.png'
        ) 