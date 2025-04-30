#!/bin/bash

# Check if cache directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <cache_directory>"
    echo "Example: $0 cache/cache_synthetic_noisy"
    exit 1
fi

CACHE_DIR=$1

# Check if cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
    echo "Error: Cache directory '$CACHE_DIR' does not exist"
    exit 1
fi

# Set k range
K_MIN=2
K_MAX=5

echo "Running welfare evaluations for k=$K_MIN to k=$K_MAX in $CACHE_DIR"

# Run FCBC welfare evaluation
echo -e "\nEvaluating FCBC welfare costs..."
python evaluate_fcbc_welfare.py --k_min $K_MIN --k_max $K_MAX --cache_dir "$CACHE_DIR"

# Run Samira's SF welfare evaluation
echo -e "\nEvaluating Samira's SF welfare costs..."
python evaluate_samira_sf_welfare.py --k_min $K_MIN --k_max $K_MAX --cache_dir "$CACHE_DIR"

# Run Welfare Clustering evaluation
echo -e "\nEvaluating Welfare Clustering costs..."
python evaluate_welfare_clustering_welfare.py --k_min $K_MIN --k_max $K_MAX --cache_dir "$CACHE_DIR"

echo -e "\nAll evaluations completed!" 