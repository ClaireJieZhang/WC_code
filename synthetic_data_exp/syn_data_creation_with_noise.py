import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd  # <== ADD THIS

def generate_noisy_synthetic_data(r=1, R=5, D=8, n_per_cluster=20, cache_dir="cache/cache_synthetic_noisy"):
    np.random.seed(42)

    # Define centers
    red_center_r1 = np.array([-R, D])
    blue_center_r1 = np.array([R, D])
    red_center_r2 = np.array([-r, 0])
    blue_center_r2 = np.array([r, 0])

    # Gaussian noise
    def sample_points(center, n, std_dev):
        return np.random.normal(loc=center, scale=std_dev, size=(n, 2))

    # Generate points
    red_points_r1 = sample_points(red_center_r1, n_per_cluster, 0.3*r)
    blue_points_r1 = sample_points(blue_center_r1, n_per_cluster, 0.3*r)
    red_points_r2 = sample_points(red_center_r2, n_per_cluster, 0.1*r)
    blue_points_r2 = sample_points(blue_center_r2, n_per_cluster, 0.1*r)

    # Stack all
    X = np.vstack([red_points_r1, blue_points_r1, red_points_r2, blue_points_r2])

    # Labels: 0 = red, 1 = blue
    colors = np.array([0]*n_per_cluster + [1]*n_per_cluster + [0]*n_per_cluster + [1]*n_per_cluster)

    # Save the dataset
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "data_matrix.npy"), X)
    np.save(os.path.join(cache_dir, "group_labels.npy"), colors)
    with open(os.path.join(cache_dir, "group_names.txt"), "w") as f:
        f.write("red\n")
        f.write("blue\n")

    # Save df_clean.csv (new)
    df_clean = pd.DataFrame(X, columns=["x", "y"])  # Give synthetic feature names
    df_clean.to_csv(os.path.join(cache_dir, "df_clean.csv"), index=False)

    print(f"Saved noisy synthetic dataset to '{cache_dir}'")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[colors == 0, 0], X[colors == 0, 1], c='red', label='Red points', alpha=0.6)
    plt.scatter(X[colors == 1, 0], X[colors == 1, 1], c='blue', label='Blue points', alpha=0.6)
    plt.axhline(y=D/2, color='gray', linestyle='--', linewidth=1)
    plt.title('Noisy Synthetic Dataset: Two Regions with Spread')
    plt.xlabel('Feature 1 (x)')
    plt.ylabel('Feature 2 (y)')
    plt.grid(True)
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(cache_dir, "noisy_synthetic_data.png"))

    return X, colors

# Generate
generate_noisy_synthetic_data()
