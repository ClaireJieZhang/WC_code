import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import os
import configparser
from evaluation_utils.utils import read_data, clean_data, scale_data


def read_list(value):
    """Helper function to read comma-separated lists from config files."""
    return [item.strip() for item in value.split(',')]



def load_and_prepare_data(ini_path, dataset_name, max_points=None, ml_model_flag=False, p_acc=1.0):
    """
    Unified data loading for WC, FCBC, SF. Encodes group using eval conditions,
    applies cleaning, column selection, and normalization.

    Returns:
        df_scaled: scaled DataFrame (without group col)
        color_flag_array: np.ndarray (group labels 0/1)
        df_full: full cleaned DataFrame (with group col)
        group_names: list of original group names
    """
    config = configparser.ConfigParser(converters={"list": read_list})
    config.read(ini_path)

    # Step 1: Read and optionally subsample
    df = read_data(config, dataset_name)
    if max_points and len(df) > max_points:
        df = df.head(max_points)


    
    # Step 3: Clean data (categorical encoding)
    df, _ = clean_data(df, config, dataset_name)
    print("Category encoding for sex:", dict(enumerate(df["sex"].astype("category").cat.categories)))


    # Step 2: Assign group labels using condition logic
    variable_of_interest = config[dataset_name].getlist("fairness_variable")
    assert len(variable_of_interest) == 1
    variable = variable_of_interest[0]
    bucket_conditions = config[dataset_name].getlist(f"{variable}_conditions")

    color_flag_array = np.zeros(len(df), dtype=int)
    group_counts = [0 for _ in bucket_conditions]


    for i, row in df.iterrows():
        for bucket_idx, bucket in enumerate(bucket_conditions):
            try:
                if eval(bucket)(row[variable]):
                    color_flag_array[i] = bucket_idx
                    group_counts[bucket_idx] += 1
                    break
            except Exception as e:
                raise RuntimeError(f"Error evaluating condition `{bucket}` on row {i}: {e}")


    # Step 4: Select specified columns
    selected_columns = config[dataset_name].getlist("columns")
    df = df[selected_columns]

    # Step 5: Optional scaling
    if config["DEFAULT"].getboolean("scaling", fallback=True):
        df = scale_data(df)

    # Step 6: Optional group names
    group_names_key = f"{variable}_group_names"
    if group_names_key in config[dataset_name]:
        group_names = config[dataset_name].getlist(group_names_key)
    else:
        group_names = [f"Group {i}" for i in range(len(bucket_conditions))]

    print(f"[INFO] Group counts: {dict(enumerate(group_counts))}")
    return df.to_numpy(), np.array(color_flag_array), df, group_names




def normalize_data(data):
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: numpy array - Data to normalize
        
    Returns:
        normalized_data: numpy array - Normalized data
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # Avoid divide-by-zero: only keep columns with non-zero std
    keep = stds != 0
    data_centered = data - means
    data_scaled = data_centered[:, keep] / stds[keep]
    
    return data_scaled 