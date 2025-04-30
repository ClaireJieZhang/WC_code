import numpy as np
import argparse
import os

def inspect_group_labels(cache_dir):
    # Load group_labels.npy
    group_labels_path = os.path.join(cache_dir, "group_labels.npy")
    if not os.path.exists(group_labels_path):
        print(f"File not found: {group_labels_path}")
        return
    
    group_labels = np.load(group_labels_path)

    # Display information
    print(f"[INFO] group_labels shape: {group_labels.shape}")
    print(f"[INFO] group_labels dtype: {group_labels.dtype}")
    print(f"[INFO] unique labels and their counts:")
    unique_labels, counts = np.unique(group_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} points")

    # Optionally print first few labels
    print("\n[INFO] First 10 group labels:")
    print(group_labels[:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect group_labels.npy")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing group_labels.npy")
    args = parser.parse_args()

    inspect_group_labels(args.cache_dir)
