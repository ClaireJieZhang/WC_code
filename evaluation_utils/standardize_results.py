import json
import pickle
import numpy as np
from datetime import datetime
from welfare_evaluation import evaluate_welfare_cost

def standardize_samira_results(result_file, data, group_labels, lambda_param=0.5, p=2):
    """
    Convert Samira_SF_Python results to standardized format.
    
    Args:
        result_file: str - path to .pkl result file
        data: numpy array - original data points
        group_labels: numpy array - group labels for each point
        lambda_param: float - weight between distance and fairness costs
        p: int - distance metric parameter (1=k-median, 2=k-means)
        
    Returns:
        dict - standardized results
    """
    with open(result_file, 'rb') as f:
        results = pickle.load(f)
    
    standardized = {
        "pipeline": "Samira_SF_Python",
        "dataset": result_file.split('/')[-1].split('_')[0],  # Extract dataset name from filename
        "timestamp": datetime.now().isoformat(),
        "data": {
            "points": data.tolist(),
            "group_labels": group_labels.tolist()
        },
        "results": {}
    }
    
    for k, k_results in results.items():
        # Calculate welfare costs for both standard and fair clustering
        welfare_std = evaluate_welfare_cost(
            centers=k_results['centers'],
            assignment=k_results['clustering'],
            points=data,
            group_labels=group_labels,
            lambda_param=lambda_param,
            p=p
        )
        
        welfare_fair = evaluate_welfare_cost(
            centers=k_results['centers_f'],
            assignment=k_results['clustering_f'],
            points=data,
            group_labels=group_labels,
            lambda_param=lambda_param,
            p=p
        )
        
        standardized["results"][k] = {
            "standard": {
                "centers": k_results['centers'].tolist(),
                "assignment": k_results['clustering'].tolist(),
                "metrics": {
                    "welfare_cost": welfare_std,
                    "runtime": k_results['runtime'],
                    "group_costs": k_results['cost'].tolist()
                }
            },
            "fair": {
                "centers": k_results['centers_f'].tolist(),
                "assignment": k_results['clustering_f'].tolist(),
                "metrics": {
                    "welfare_cost": welfare_fair,
                    "runtime": k_results['runtime_f'],
                    "group_costs": k_results['cost_f'].tolist()
                }
            }
        }
    
    return standardized

def standardize_welfare_results(result_file, data, group_labels):
    """
    Convert Welfare_Clustering results to standardized format.
    Note: These results are already in a similar format, just need minor restructuring.
    
    Args:
        result_file: str - path to .json result file
        data: numpy array - original data points
        group_labels: numpy array - group labels for each point
        
    Returns:
        dict - standardized results
    """
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Welfare results are already well-structured, just need to ensure consistent format
    standardized = {
        "pipeline": "Welfare_Clustering",
        "dataset": results.get('dataset', 'unknown'),
        "timestamp": results.get('timestamp', datetime.now().isoformat()),
        "data": {
            "points": data.tolist(),
            "group_labels": group_labels.tolist()
        },
        "results": {
            results['config']['k']: {
                "standard": None,  # Welfare only does fair clustering
                "fair": {
                    "centers": results['centers'],
                    "assignment": results['assignment'],
                    "metrics": {
                        "welfare_cost": results['welfare_cost'],
                        "runtime": results.get('runtime', None),
                        "group_costs": results['welfare_cost']['group_costs']
                    }
                }
            }
        }
    }
    
    return standardized

def standardize_fcbc_results(result_file, data, group_labels, lambda_param=0.5, p=2):
    """
    Convert Fair-Clustering-Under-Bounded-Cost (FCBC) results to standardized format.
    
    Args:
        result_file: str - path to .pkl result file
        data: numpy array - original data points
        group_labels: numpy array - group labels for each point
        lambda_param: float - weight between distance and fairness costs
        p: int - distance metric parameter (1=k-median, 2=k-means)
        
    Returns:
        dict - standardized results
    """
    with open(result_file, 'rb') as f:
        results = pickle.load(f)
    
    standardized = {
        "pipeline": "Fair-Clustering-Under-Bounded-Cost",
        "dataset": result_file.split('/')[-1].split('_')[0],  # Extract dataset name from filename
        "timestamp": datetime.now().isoformat(),
        "data": {
            "points": data.tolist(),
            "group_labels": group_labels.tolist()
        },
        "results": {}
    }
    
    # Need to extract k from results structure
    # This may need adjustment based on actual results format
    for k in results.K_VALS:
        k_results = results.get_data(k)
        
        welfare = evaluate_welfare_cost(
            centers=k_results['centers'],
            assignment=k_results['assignment'],
            points=data,
            group_labels=group_labels,
            lambda_param=lambda_param,
            p=p
        )
        
        standardized["results"][k] = {
            "standard": None,  # Fair clustering only does fair version
            "fair": {
                "centers": k_results['centers'].tolist(),
                "assignment": k_results['assignment'].tolist(),
                "metrics": {
                    "welfare_cost": welfare,
                    "runtime": k_results.get('runtime', None),
                    "group_costs": k_results.get('group_costs', None)
                }
            }
        }
    
    return standardized

def save_standardized_results(standardized_results, output_file):
    """
    Save standardized results to a JSON file.
    
    Args:
        standardized_results: dict - standardized results from any pipeline
        output_file: str - path to save the JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(standardized_results, f, indent=2)
    
    print(f"✅ Standardized results saved to: {output_file}")

def standardize_and_save_results(pipeline_name, result_file, output_file, data, group_labels, lambda_param=0.5, p=2):
    """
    Standardize and save results from any pipeline.
    
    Args:
        pipeline_name: str - name of the pipeline
        result_file: str - path to original result file
        output_file: str - path to save standardized results
        data: numpy array - original data points
        group_labels: numpy array - group labels for each point
        lambda_param: float - weight between distance and fairness costs
        p: int - distance metric parameter (1=k-median, 2=k-means)
    """
    if pipeline_name == 'Samira_SF_Python':
        standardized = standardize_samira_results(result_file, data, group_labels, lambda_param, p)
    elif pipeline_name == 'Welfare_Clustering':
        standardized = standardize_welfare_results(result_file, data, group_labels)
    elif pipeline_name == 'Fair-Clustering-Under-Bounded-Cost':
        standardized = standardize_fcbc_results(result_file, data, group_labels, lambda_param, p)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    
    save_standardized_results(standardized, output_file) 