import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import os
import configparser
from fairlearn.datasets import fetch_adult

def read_list(value):
    """Helper function to read comma-separated lists from config files."""
    return [item.strip() for item in value.split(',')]

def load_adult_data(data_path=None, max_points=None):
    """
    Load and preprocess the adult dataset consistently.
    
    Args:
        data_path: str, optional - Path to the adult.csv file. If None, uses fairlearn's fetch_adult.
        max_points: int, optional - Maximum number of points to use. If None, use all.
        
    Returns:
        data: numpy array - Preprocessed features
        group_labels: numpy array - Group labels (1 for Female, 2 for Male)
        group_names: list - Names of the groups
    """
    if data_path is None or not os.path.exists(data_path):
        # Use fairlearn's fetch_adult if no path provided or file doesn't exist
        X, y = fetch_adult(as_frame=True, return_X_y=True)
        
        # Drop rows with any missing entries
        X_clean = X.replace('?', np.nan).dropna()
        y_clean = y.loc[X_clean.index]
        
        # Extract sensitive attribute
        sensitive = X_clean['sex']
        X_clean = X_clean.drop(columns=['sex'])
        
        # Ordinal encode all features
        enc = OrdinalEncoder()
        X_encoded = enc.fit_transform(X_clean)
        
        # Convert sensitive attribute to binary (1 for Female, 2 for Male)
        group_labels = (sensitive == 'Male').astype(int) + 1
        group_names = ['Female', 'Male']
        
        data = X_encoded
    else:
        # Load from CSV file
        df = pd.read_csv(data_path)
        
        # Extract sensitive attribute
        sensitive = df['sex']
        df = df.drop(columns=['sex'])
        
        # Handle missing values
        df = df.replace('?', np.nan).dropna()
        
        # Ordinal encode all features
        enc = OrdinalEncoder()
        data = enc.fit_transform(df)
        
        # Convert sensitive attribute to binary (1 for Female, 2 for Male)
        group_labels = (sensitive == 'Male').astype(int) + 1
        group_names = ['Female', 'Male']
    
    # Subsample if needed
    if max_points is not None and len(data) > max_points:
        indices = np.random.choice(len(data), max_points, replace=False)
        data = data[indices]
        group_labels = group_labels[indices]
    
    return data, group_labels, group_names

def load_data_from_config(config_file, dataset_name, max_points=None):
    """
    Load data using a config file (for compatibility with existing pipelines).
    
    Args:
        config_file: str - Path to the config file
        dataset_name: str - Name of the dataset section in the config
        max_points: int, optional - Maximum number of points to use
        
    Returns:
        data: numpy array - Preprocessed features
        group_labels: numpy array - Group labels
        group_names: list - Names of the groups
    """
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)
    
    if dataset_name == 'adult':
        # Use the common adult data loader
        data_path = config[dataset_name].get('csv_file', None)
        return load_adult_data(data_path, max_points)
    
    # For other datasets, implement similar loaders
    # ...
    
    raise ValueError(f"Dataset {dataset_name} not supported in the common data loader yet")

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