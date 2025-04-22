# Clustering Evaluation Utilities

This directory contains utilities for evaluating clustering results across different pipelines.

## Welfare Cost Evaluation

The `welfare_evaluation.py` module provides a function to evaluate the welfare cost (D_h) for clustering solutions from any pipeline.

### Function: `evaluate_welfare_cost`

```python
from evaluation_utils.welfare_evaluation import evaluate_welfare_cost

welfare_metrics = evaluate_welfare_cost(
    centers=centers,
    assignment=assignment,
    points=points,
    group_labels=group_labels,
    lambda_param=0.5,  # optional
    p=2  # optional
)
```

#### Parameters:
- `centers`: numpy array of shape (k, d) - cluster centers
- `assignment`: numpy array of shape (n,) - cluster assignments for each point
- `points`: numpy array of shape (n, d) - data points
- `group_labels`: numpy array of shape (n,) - group labels for each point
- `lambda_param`: float in [0,1] - weight between distance and fairness costs (default: 0.5)
- `p`: int in {1,2} - 1 for k-median, 2 for k-means (default: 2)

#### Returns:
- `max_welfare_cost`: maximum D_h across all groups
- `group_costs`: dictionary mapping each group to its D_h value

## Plotting Utilities

The `plotting.py` module provides functions to visualize welfare costs across different pipelines.

### Function: `plot_welfare_costs_comparison`

```python
from evaluation_utils.plotting import plot_welfare_costs_comparison

plot_welfare_costs_comparison(
    pipeline_results=pipeline_results,
    dataset_name='adult',
    k=5,
    lambda_param=0.5,
    p=2,
    save_path='welfare_costs.png'  # optional
)
```

#### Parameters:
- `pipeline_results`: dict - keys are pipeline names, values are dicts containing:
  - 'centers': numpy array of cluster centers
  - 'assignment': numpy array of cluster assignments
  - 'points': numpy array of data points
  - 'group_labels': numpy array of group labels
- `dataset_name`: str - name of the dataset (for plot title)
- `k`: int - number of clusters
- `lambda_param`: float - weight between distance and fairness costs (default: 0.5)
- `p`: int - distance metric parameter (1 for k-median, 2 for k-means) (default: 2)
- `save_path`: str - path to save the plot (optional)

### Function: `load_and_plot_results`

```python
from evaluation_utils.plotting import load_and_plot_results

result_files = {
    'Samira_SF_Python': 'path/to/samira_results.json',
    'Welfare_Clustering': 'path/to/welfare_results.json',
    'Fair-Clustering-Under-Bounded-Cost': 'path/to/fair_results.json'
}

load_and_plot_results(
    result_files=result_files,
    dataset_name='adult',
    k=5,
    lambda_param=0.5,
    p=2,
    save_path='welfare_costs.png'  # optional
)
```

#### Parameters:
- `result_files`: dict - keys are pipeline names, values are paths to result files
- `dataset_name`: str - name of the dataset
- `k`: int - number of clusters
- `lambda_param`: float - weight between distance and fairness costs (default: 0.5)
- `p`: int - distance metric parameter (1 for k-median, 2 for k-means) (default: 2)
- `save_path`: str - path to save the plot (optional)

## Example Usage

See `example_usage.py` for a complete example of how to use this utility with results from any of the three pipelines:

1. Samira_SF_Python
2. Welfare_Clustering
3. Fair-Clustering-Under-Bounded-Cost

## How to Use with Your Pipeline

1. Import the evaluation function:
   ```python
   from evaluation_utils.welfare_evaluation import evaluate_welfare_cost
   ```

2. Load your data and results:
   ```python
   data = np.load('your_data.npy')
   group_labels = np.load('your_group_labels.npy')
   results = load_results_from_pipeline('Your_Pipeline', 'path/to/results.json')
   ```

3. Evaluate welfare cost:
   ```python
   welfare_metrics = evaluate_welfare_cost(
       centers=results['centers'],
       assignment=results['assignment'],
       points=data,
       group_labels=group_labels,
       lambda_param=0.5,
       p=2
   )
   ```

4. Access the results:
   ```python
   max_cost = welfare_metrics['max_welfare_cost']
   group_costs = welfare_metrics['group_costs']
   ```

5. Plot the results:
   ```python
   from evaluation_utils.plotting import plot_welfare_costs_comparison
   
   pipeline_results = {
       'Your_Pipeline': {
           'centers': results['centers'],
           'assignment': results['assignment'],
           'points': data,
           'group_labels': group_labels
       }
   }
   
   plot_welfare_costs_comparison(
       pipeline_results=pipeline_results,
       dataset_name='your_dataset',
       k=5,
       lambda_param=0.5,
       p=2,
       save_path='welfare_costs.png'
   )
   ```

# Evaluation Utilities

This directory contains utilities for evaluating and comparing different clustering pipelines.

## Data Loading and Preprocessing

The `data_utils.py` module provides consistent data loading and preprocessing functions for all pipelines:

```python
from evaluation_utils.data_utils import load_adult_data

# Load the adult dataset
data, group_labels, group_names = load_adult_data(max_points=1000)
```

This ensures that all pipelines use the same data and preprocessing steps.

## Standardizing Results

The `standardize_results.py` module provides functions to convert results from different pipelines to a standardized format:

```python
from evaluation_utils.standardize_results import standardize_and_save_results

# Standardize results from a pipeline
standardize_and_save_results(
    pipeline_name="Samira_SF_Python",
    result_file="Samira_SF_Python/results/adult_k5-5_results.pkl",
    output_file="standardized_results/samira_standardized.json",
    data=data,
    group_labels=group_labels,
    lambda_param=0.5,
    p=2
)
```

## Running All Pipelines

The `run_all_pipelines.py` script provides a convenient way to run all three pipelines with the same data and standardize the results:

```bash
python evaluation_utils/run_all_pipelines.py --k 5 --lambda_param 0.5 --max_points 1000
```

This will:
1. Load the adult dataset
2. Run all three pipelines with the same data
3. Standardize the results
4. Save the standardized results to the specified output directory

## Comparing Results

The `run_comparison.py` script provides a way to compare welfare costs across different pipelines:

```bash
python evaluation_utils/run_comparison.py \
    --dataset adult \
    --k 5 \
    --lambda_param 0.5 \
    --p 2 \
    --samira_results standardized_results/samira_standardized.json \
    --welfare_results standardized_results/welfare_standardized.json \
    --fair_results standardized_results/fcbc_standardized.json \
    --output_dir plots
```

This will generate a plot comparing the welfare costs across the different pipelines.

## Directory Structure

```
evaluation_utils/
├── data_utils.py           # Common data loading and preprocessing
├── standardize_results.py  # Functions to standardize results
├── run_all_pipelines.py    # Script to run all pipelines
├── run_comparison.py       # Script to compare results
├── welfare_evaluation.py   # Functions to evaluate welfare costs
└── plotting.py             # Functions to plot results
```

## Workflow

1. **Data Preparation**:
   - Use `data_utils.py` to load and preprocess data consistently

2. **Run Pipelines**:
   - Run each pipeline separately, or use `run_all_pipelines.py` to run all at once

3. **Standardize Results**:
   - Use `standardize_results.py` to convert results to a standardized format

4. **Compare Results**:
   - Use `run_comparison.py` to compare welfare costs across pipelines
   - Use `plotting.py` to visualize the results 