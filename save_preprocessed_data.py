import os
import numpy as np
import pandas as pd
from evaluation_utils.data_utils import load_data

def save_cached_data(config_file, dataset_name, cache_dir="cache", max_points=None):
    """
    Loads and saves unified data to disk for use by WC, SF, and FCBC pipelines.

    Args:
        config_file: str, path to .ini file for shared loading
        dataset_name: str, section name in the config
        cache_dir: str, directory where outputs are written
        max_points: int or None, optional subsampling
    """
    print(f"📦 Loading and caching data for dataset: {dataset_name}")
    os.makedirs(cache_dir, exist_ok=True)

    data_matrix, group_labels, df_clean, group_names = load_data(
        config_file=config_file,
        dataset_name=dataset_name,
        max_points=max_points
    )

    np.save(os.path.join(cache_dir, "data_matrix.npy"), data_matrix)
    np.save(os.path.join(cache_dir, "group_labels.npy"), group_labels)
    df_clean.to_csv(os.path.join(cache_dir, "df_clean.csv"), index=False)

        # Save group names
    with open(os.path.join(cache_dir, "group_names.txt"), "w") as f:
        for name in group_names:
            f.write(name + "\n")

    print("✅ Saved:")
    print(f"  → data_matrix.npy     (shape: {data_matrix.shape})")
    print(f"  → group_labels.npy    (len: {len(group_labels)})")
    print(f"  → df_clean.csv        (shape: {df_clean.shape})")
    print(f"  → group_names.txt     ({len(group_names)} lines)")


    return {
        "data_matrix": data_matrix,
        "group_labels": group_labels,
        "df_clean": df_clean,
        "group_names": group_names
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/standard_run.ini")
    parser.add_argument("--dataset_name", type=str, default="adult")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--max_points", type=int, default=None)
    args = parser.parse_args()

    save_cached_data(
        config_file=args.config_file,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        max_points=args.max_points
    )
