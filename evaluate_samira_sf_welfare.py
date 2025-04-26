import os
import numpy as np
import pandas as pd
import pickle
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost

def parse_numpy_array(array_str):
    """
    Parse a numpy array string into a numpy array.
    """
    # Remove 'array(' and ')' from the string
    array_str = array_str.strip('array()')
    
    # Use regex to find all numbers, including scientific notation
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    numbers = re.findall(pattern, array_str)
    
    # Convert all numbers to float
    numbers = [float(x) for x in numbers]
    
    # Reshape into 2D array if we have multiple rows
    if '],' in array_str:
        # Count number of rows by counting '],'
        num_rows = array_str.count('],') + 1
        # Reshape the array
        return np.array(numbers).reshape(num_rows, -1)
    else:
        return np.array(numbers)

def load_samira_sf_results(cache_dir, k):
    """
    Load Samira's SF results for a specific k value.
    """
    # Load data matrix and group labels
    data_matrix = np.load(os.path.join(cache_dir, "data_matrix.npy"))
    group_labels = np.load(os.path.join(cache_dir, "group_labels.npy"))
    
    # Find the most recent results file
    results_dir = os.path.join(cache_dir, "socially_fair_results")
    result_files = [f for f in os.listdir(results_dir) if f.startswith("sf_results_") and f.endswith(".pkl")]
    if not result_files:
        raise FileNotFoundError(f"No Samira's SF results found in {results_dir}")
    
    # Sort by timestamp in filename and get the most recent
    latest_file = sorted(result_files)[-1]
    with open(os.path.join(results_dir, latest_file), 'rb') as f:
        results = pickle.load(f)
    
    # Get results for this k value
    k_results = results['results'][str(k)]
    
    # Get centers and assignment from fair clustering results
    centers = np.array(k_results['fair']['centers'])
    assignment = np.array(k_results['fair']['assignment'])
    
    # Print debugging information
    print(f"\nData dimensions:")
    print(f"data_matrix shape: {data_matrix.shape}")
    print(f"group_labels shape: {group_labels.shape}")
    print(f"centers shape: {centers.shape}")
    print(f"assignment shape: {assignment.shape}")
    print(f"assignment min: {assignment.min()}, max: {assignment.max()}")
    
    # Check if assignment length matches data length
    if len(assignment) != len(data_matrix):
        print(f"\nWARNING: Assignment length ({len(assignment)}) does not match data length ({len(data_matrix)})")
        print("This suggests the assignment array is incomplete or incorrect.")
        print("Attempting to fix by using the standard clustering assignment instead...")
        
        # Try using the standard clustering assignment
        standard_assignment = np.array(k_results['standard']['assignment'])
        if len(standard_assignment) == len(data_matrix):
            print("Using standard clustering assignment instead.")
            assignment = standard_assignment
        else:
            print(f"Standard assignment also has incorrect length: {len(standard_assignment)}")
            print("This is a critical error. Please check the data and results.")
            raise ValueError("Assignment array length does not match data length")
    
    return data_matrix, group_labels, centers, assignment

def evaluate_samira_sf_costs(cache_dir, k, lambda_param=0.5):
    """
    Evaluate welfare costs for Samira's SF results.
    """
    # Load data and results
    data_matrix, group_labels, centers, assignment = load_samira_sf_results(cache_dir, k)
    
    # Calculate welfare cost
    welfare_metrics = evaluate_welfare_cost(
        centers=centers,
        assignment=assignment,
        points=data_matrix,
        group_labels=group_labels,
        lambda_param=lambda_param
    )
    
    # Create result row
    result_row = {
        'k': k,
        'lambda_param': lambda_param,
        'max_welfare_cost': welfare_metrics['max_welfare_cost'],
        'group_costs': welfare_metrics['group_costs']
    }
    
    # Convert to DataFrame
    df_welfare = pd.DataFrame([result_row])
    
    # Save results
    results_dir = os.path.join(cache_dir, "welfare_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    output_file = os.path.join(results_dir, f"samira_sf_k{k}_welfare_costs.csv")
    df_welfare.to_csv(output_file, index=False)
    
    print(f"\nResults for k={k}:")
    print(f"Welfare cost: {df_welfare['max_welfare_cost'].iloc[0]:.4f}")
    print(f"\nDetailed results saved to: {output_file}")
    
    return df_welfare

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate welfare costs for Samira's SF results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing cached data")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight between distance and fairness costs")
    
    args = parser.parse_args()
    
    evaluate_samira_sf_costs(args.cache_dir, args.k, args.lambda_param) 