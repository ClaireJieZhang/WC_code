import numpy as np
import matplotlib.pyplot as plt
import os

def generate_synthetic_data(overlap_factor=10, r=1, R=7, D=80, cache_dir="cache_synthetic"):
    np.random.seed(42)

    # Centers of the clusters
    red_center_r1 = np.array([-R, D])     # Region 1, red cluster (upper)
    blue_center_r1 = np.array([R, D])      # Region 1, blue cluster (upper)

    red_center_r2 = np.array([-r, 0])      # Region 2, red cluster (lower)
    blue_center_r2 = np.array([r, 0])       # Region 2, blue cluster (lower)

    # Arrange 4 points per cluster around their center (forming tiny squares)
    square_offsets = np.array([
        [-r/2, -r/2],
        [-r/2, r/2],
        [r/2, -r/2],
        [r/2, r/2]
    ])

    # Region R1
    red_points_r1 = red_center_r1 + square_offsets
    blue_points_r1 = blue_center_r1 + square_offsets

    # Region R2
    red_points_r2 = red_center_r2 + square_offsets
    blue_points_r2 = blue_center_r2 + square_offsets

    # Repeat points for overlap
    red_points_r1 = np.repeat(red_points_r1, overlap_factor, axis=0)
    blue_points_r1 = np.repeat(blue_points_r1, overlap_factor, axis=0)
    red_points_r2 = np.repeat(red_points_r2, overlap_factor, axis=0)
    blue_points_r2 = np.repeat(blue_points_r2, overlap_factor, axis=0)

    # Stack all points
    X = np.vstack([red_points_r1, blue_points_r1, red_points_r2, blue_points_r2])

    # Create color labels: 0 = red, 1 = blue
    colors = np.array([0]*len(red_points_r1) + [1]*len(blue_points_r1) + 
                      [0]*len(red_points_r2) + [1]*len(blue_points_r2))

    # Save the dataset
    os.makedirs(cache_dir, exist_ok=True)

    # Save data matrix (only features)
    np.save(os.path.join(cache_dir, "data_matrix.npy"), X)

    # Save group labels
    np.save(os.path.join(cache_dir, "group_labels.npy"), colors)

    # Save group names
    with open(os.path.join(cache_dir, "group_names.txt"), "w") as f:
        f.write("red\n")
        f.write("blue\n")

    print(f"Saved synthetic dataset to '{cache_dir}'")

    return X, colors

# Generate and Save
X, colors = generate_synthetic_data()

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(X[colors == 0, 0], X[colors == 0, 1], c='red', label='Red points', alpha=0.6)
plt.scatter(X[colors == 1, 0], X[colors == 1, 1], c='blue', label='Blue points', alpha=0.6)
plt.axhline(y=80, color='gray', linestyle='--', linewidth=1)  # Midline for visual reference
plt.legend()
plt.title('Synthetic Dataset: Two vertically separated regions')
plt.xlabel('Feature 1 (x)')
plt.ylabel('Feature 2 (y)')
plt.grid(True)
plt.show()
