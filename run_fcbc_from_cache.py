import pandas as pd
import numpy as np
from Fair_Clustering_Under_Bounded_Cost.fcbc_wrapper import run_fcbc_pipeline_with_loaded_data

# Load preprocessed shared data
df_clean = pd.read_csv("cache/df_clean.csv")
group_labels = np.load("cache/group_labels.npy")

# Call FCBC pipeline using wrapper
run_fcbc_pipeline_with_loaded_data(
    df=df_clean,
    svar_all=group_labels,
    dataset_name="adult",
    config_file="configs/fcbc_adult.ini",
    data_dir="data/",
    num_clusters=10,
    deltas=[0.1]  # or read from config if needed
)
