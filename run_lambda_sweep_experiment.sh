#!/bin/bash

# Set default variables
CACHE_DIR="cache"
DATA_DIR="cache/cache_synthetic_noisy"
CONFIG_PATH="configs/welfare_clustering_config.ini"
K=4
UPWEIGHT=1  # Add upweight factor

# Default lambda sweep parameters
LAMBDA_START=0
LAMBDA_END=0.05
LAMBDA_STEP=0.001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lambda_start)
            LAMBDA_START="$2"
            shift 2
            ;;
        --lambda_end)
            LAMBDA_END="$2"
            shift 2
            ;;
        --lambda_step)
            LAMBDA_STEP="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --upweight)
            UPWEIGHT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p "$CACHE_DIR/welfare_clustering_results"
mkdir -p "$CACHE_DIR/plots"

echo "Running welfare clustering with lambda sweep..."
echo "Lambda range: [$LAMBDA_START, $LAMBDA_END] with step $LAMBDA_STEP"
python run_welfare_clustering_lambda_sweep.py \
    --cache_dir "$DATA_DIR" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$CACHE_DIR/welfare_clustering_results" \
    --upweight "$UPWEIGHT" \
    --lambda_start "$LAMBDA_START" \
    --lambda_end "$LAMBDA_END" \
    --lambda_step "$LAMBDA_STEP"

echo "Generating comparison plots..."
python plot_lambda_sweep_comparison.py \
    --cache_dir "$CACHE_DIR" \
    --k "$K" \
    --output_dir "$CACHE_DIR/plots" \
    --lambda_start "$LAMBDA_START" \
    --lambda_end "$LAMBDA_END" \
    --lambda_step "$LAMBDA_STEP"

echo "Experiment completed! Check results in $CACHE_DIR/plots" 