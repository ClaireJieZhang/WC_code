import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from matplotlib.colors import LinearSegmentedColormap

# Try to import seaborn, but don't fail if it's not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not found. Using matplotlib default style.")

def apply_style():
    """Apply a consistent style to the plots."""
    # Set a clean, modern style without relying on seaborn's style file
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana'],
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
    })

def plot_assignment_edges(x_frac, group_labels, centers, ax, threshold=0.1):
    """
    Plot fractional assignment edges between points and clusters.
    
    Args:
        x_frac (np.ndarray): Fractional assignment matrix (n x k)
        group_labels (np.ndarray): Group labels for each point
        centers (np.ndarray): Cluster centers (k x d)
        ax (matplotlib.axes.Axes): Axes to plot on
        threshold (float): Minimum assignment value to show an edge
    """
    # Create a custom colormap for the edges
    colors = ['#1f77b4', '#ff7f0e']  # Blue for group 0, Orange for group 1
    cmap = LinearSegmentedColormap.from_list('group_cmap', colors)
    
    # Project points and centers to 2D if they're in higher dimensions
    if centers.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(x_frac)
        centers_2d = pca.transform(centers)
    else:
        points_2d = x_frac
        centers_2d = centers
    
    # First plot the edges (so they're behind points)
    for i in range(len(x_frac)):
        point = points_2d[i]
        group = int(group_labels[i])
        
        # Count number of significant assignments for this point
        significant_assignments = np.sum(x_frac[i] > threshold)
        if significant_assignments > 1:
            # This point has multiple significant assignments
            assignments_str = []
            for j in range(len(centers)):
                assignment = x_frac[i, j]
                if assignment > threshold:
                    center = centers_2d[j]
                    # Line width based on assignment value
                    width = assignment * 2
                    # Color based on group with alpha based on assignment
                    color = colors[group]
                    # Clip alpha to [0,1] to handle numerical imprecision
                    alpha = np.clip(assignment, 0, 1)
                    ax.plot([point[0], center[0]], [point[1], center[1]],
                           color=color, alpha=alpha, linewidth=width,
                           zorder=1)  # Lower zorder to be behind points
                    assignments_str.append(f"C{j}: {assignment:.2f}")
            
            # Add annotation for split assignments
            ax.annotate(
                f"Split: {', '.join(assignments_str)}", 
                (point[0], point[1]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8,
                zorder=4
            )
        else:
            # Single strong assignment
            for j in range(len(centers)):
                assignment = x_frac[i, j]
                if assignment > threshold:
                    center = centers_2d[j]
                    width = assignment * 2
                    color = colors[group]
                    # Clip alpha to [0,1] to handle numerical imprecision
                    alpha = np.clip(assignment, 0, 1)
                    ax.plot([point[0], center[0]], [point[1], center[1]],
                           color=color, alpha=alpha, linewidth=width,
                           zorder=1)
    
    # Plot points colored by group (on top of edges)
    unique_groups = np.unique(group_labels)
    for group in unique_groups:
        mask = group_labels == group
        ax.scatter(points_2d[mask, 0], points_2d[mask, 1], 
                  c=[colors[int(group)]], label=f'Group {int(group)}',
                  alpha=0.6, s=30, zorder=2)
    
    # Plot centers (on top of everything)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
              c='red', marker='*', s=200, label='Centers',
              zorder=3)
    
    ax.set_title('Fractional Assignment Edges\n(Line thickness = assignment strength)')
    ax.legend()

    # Add explanation text
    ax.text(0.02, 0.98, 
            "Points with multiple assignments > 0.1 are annotated\nLine thickness shows assignment strength",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            fontsize=8)

def plot_assignment_lines(x_frac, group_labels, centers, data_points, output_dir="results", 
                         title="Assignment Lines", save_name="assignment_lines.png",
                         lambda_param=None):
    """
    Plot assignments as lines between points and their assigned clusters.
    Line thickness and opacity indicate assignment strength.
    
    Args:
        x_frac (np.ndarray): Fractional assignment matrix (n x k)
        group_labels (np.ndarray): Group labels for each point
        centers (np.ndarray): Cluster centers (k x d)
        data_points (np.ndarray): Original data points (n x d)
        output_dir (str): Directory to save the plot
        title (str): Title for the plot
        save_name (str): Name of the output file
        lambda_param (float): Lambda parameter used in the LP solver
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add lambda to title if provided
    if lambda_param is not None:
        title = f"{title} (λ={lambda_param:.3f})"
    
    # Apply our custom style
    apply_style()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    try:
        # Project points and centers to 2D if they're in higher dimensions
        if data_points.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(data_points)
            centers_2d = pca.transform(centers)
        else:
            points_2d = data_points
            centers_2d = centers
        
        # Create a colormap for groups
        unique_groups = np.unique(group_labels)
        group_colors = ['#1f77b4', '#ff7f0e']  # Blue for group 0, Orange for group 1
        
        # Plot points colored by group
        ax = fig.add_subplot(111)
        
        # First plot the assignment lines (so they're behind points)
        for i in range(len(x_frac)):
            point = points_2d[i]
            group = int(group_labels[i])
            color = group_colors[group]
            
            # Plot lines to each cluster with assignment > 0.1
            for j in range(len(centers)):
                assignment = x_frac[i, j]
                if assignment > 0.1:  # Only show significant assignments
                    center = centers_2d[j]
                    # Line width based on assignment value
                    width = assignment * 3
                    # Clip alpha to [0,1] to handle numerical imprecision
                    alpha = np.clip(0.3 + assignment * 0.7, 0, 1)
                    ax.plot([point[0], center[0]], [point[1], center[1]],
                           color=color, alpha=alpha, linewidth=width,
                           zorder=1)  # Lower zorder to be behind points
        
        # Plot points colored by group (on top of lines)
        for group in unique_groups:
            mask = group_labels == group
            if np.sum(mask) > 0:
                ax.scatter(points_2d[mask, 0], points_2d[mask, 1], 
                          c=[group_colors[int(group)]], label=f'Group {int(group)}',
                          alpha=0.6, s=30, zorder=2)
        
        # Plot centers (on top of everything)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                  c='red', marker='*', s=200, label='Centers',
                  zorder=3)
        
        # Add explanation text
        ax.text(0.02, 0.98, 
                "Line thickness and opacity show assignment strength\nOnly assignments > 0.1 are shown",
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8)
        
        ax.set_title(title)
        ax.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot with error handling
        output_path = os.path.join(output_dir, save_name)
        try:
            # First try saving with high DPI
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        except Exception as e:
            print(f"Warning: Failed to save with high DPI: {e}")
            try:
                # If that fails, try saving with lower DPI
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
            except Exception as e:
                print(f"Warning: Failed to save with lower DPI: {e}")
                try:
                    # If that fails too, try saving without DPI specification
                    plt.savefig(output_path, bbox_inches='tight')
                except Exception as e:
                    print(f"Error: Failed to save plot: {e}")
                    raise
        
        print(f"Assignment lines plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        raise
    finally:
        # Always close the figure to prevent memory leaks
        plt.close(fig)

def plot_lp_clustering(x_frac, group_labels, centers, data_points, output_dir="results", 
                      title="LP Clustering Solution",
                      save_name="lp_clustering_solution.png",
                      lambda_param=None):
    """
    Plot the clustering formed from LP solutions by taking the maximum fractional assignment for each point.
    
    Args:
        x_frac (np.ndarray): Fractional assignment matrix (n x k)
        group_labels (np.ndarray): Group labels for each point
        centers (np.ndarray): Cluster centers (k x d)
        data_points (np.ndarray): Original data points (n x d)
        output_dir (str): Directory to save the plot
        title (str): Title for the plot
        save_name (str): Name of the output file
        lambda_param (float): Lambda parameter used in the LP solver
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add lambda to title if provided
    if lambda_param is not None:
        title = f"{title} (λ={lambda_param:.3f})"
    
    # Apply our custom style
    apply_style()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    try:
        # Get the assigned cluster for each point (maximum fractional assignment)
        assigned_clusters = np.argmax(x_frac, axis=1)
        
        # Count points with non-integer assignments
        non_integer_assignments = np.sum(np.max(x_frac, axis=1) < 0.99)
        print(f"\nLP Clustering Statistics:")
        print(f"Total points: {len(x_frac)}")
        print(f"Points with non-integer assignments: {non_integer_assignments} ({non_integer_assignments/len(x_frac)*100:.1f}%)")
        
        # Project points and centers to 2D if they're in higher dimensions
        if data_points.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(data_points)
            centers_2d = pca.transform(centers)
        else:
            points_2d = data_points
            centers_2d = centers
        
        # Create a colormap for clusters
        n_clusters = len(centers)
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # Create a colormap for groups
        unique_groups = np.unique(group_labels)
        group_colors = ['#1f77b4', '#ff7f0e']  # Blue for group 0, Orange for group 1
        
        # Plot points colored by cluster and group
        ax = fig.add_subplot(111)
        
        # First plot points colored by cluster
        for i in range(n_clusters):
            mask = assigned_clusters == i
            if np.sum(mask) > 0:
                ax.scatter(points_2d[mask, 0], points_2d[mask, 1], 
                          c=[cluster_colors[i]], label=f'Cluster {i}',
                          alpha=0.6, s=30, zorder=2)
        
        # Plot centers
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                  c='red', marker='*', s=200, label='Centers',
                  zorder=3)
        
        # Add group information as edge colors
        for group in unique_groups:
            mask = group_labels == group
            if np.sum(mask) > 0:
                # Add a second scatter plot with edge colors for groups
                ax.scatter(points_2d[mask, 0], points_2d[mask, 1], 
                          facecolors='none', edgecolors=group_colors[int(group)],
                          linewidth=1.5, label=f'Group {int(group)}',
                          alpha=0.8, s=40, zorder=2)
        
        # Add cluster proportions information
        cluster_sizes = np.bincount(assigned_clusters, minlength=n_clusters)
        cluster_proportions = cluster_sizes / len(assigned_clusters)
        
        # Calculate group proportions within each cluster
        group_proportions = []
        for i in range(n_clusters):
            cluster_mask = assigned_clusters == i
            if np.sum(cluster_mask) > 0:
                cluster_group_labels = group_labels[cluster_mask]
                group_counts = np.bincount(cluster_group_labels, minlength=len(unique_groups))
                group_proportions.append(group_counts / np.sum(group_counts))
            else:
                group_proportions.append(np.zeros(len(unique_groups)))
        
        # Add cluster and group proportion information to the plot
        info_text = "Cluster Proportions:\n"
        for i in range(n_clusters):
            info_text += f"Cluster {i}: {cluster_proportions[i]:.2f}\n"
        
        info_text += "\nGroup Proportions within Clusters:\n"
        for i in range(n_clusters):
            info_text += f"Cluster {i}: "
            for j, group in enumerate(unique_groups):
                info_text += f"Group {int(group)}: {group_proportions[i][int(group)]:.2f} "
            info_text += "\n"
        
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8)
        
        ax.set_title(title)
        ax.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot with error handling
        output_path = os.path.join(output_dir, save_name)
        try:
            # First try saving with high DPI
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        except Exception as e:
            print(f"Warning: Failed to save with high DPI: {e}")
            try:
                # If that fails, try saving with lower DPI
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
            except Exception as e:
                print(f"Warning: Failed to save with lower DPI: {e}")
                try:
                    # If that fails too, try saving without DPI specification
                    plt.savefig(output_path, bbox_inches='tight')
                except Exception as e:
                    print(f"Error: Failed to save plot: {e}")
                    raise
        
        print(f"LP clustering plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        raise
    finally:
        # Always close the figure to prevent memory leaks
        plt.close(fig)

def plot_lp_fractional_solution(x_frac, group_labels, centers, output_dir="results", 
                              title="LP Fractional Assignment Solution",
                              save_name="lp_fractional_solution.png",
                              lambda_param=None,
                              data_points=None):
    """
    Plot the fractional assignment solution from the LP solver.
    
    Args:
        x_frac (np.ndarray): Fractional assignment matrix (n x k)
        group_labels (np.ndarray): Group labels for each point
        centers (np.ndarray): Cluster centers (k x d)
        output_dir (str): Directory to save the plot
        title (str): Title for the plot
        save_name (str): Name of the output file
        lambda_param (float): Lambda parameter used in the LP solver
        data_points (np.ndarray, optional): Original data points for clustering plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add lambda to title if provided
    if lambda_param is not None:
        title = f"{title} (λ={lambda_param:.3f})"
    
    # Apply our custom style
    apply_style()
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3)
    
    try:
        # Plot 1: Heatmap of fractional assignments
        ax1 = fig.add_subplot(gs[0])
        if HAS_SEABORN:
            sns.heatmap(x_frac, cmap='YlOrRd', ax=ax1)
        else:
            im = ax1.imshow(x_frac, cmap='YlOrRd', aspect='auto')
            plt.colorbar(im, ax=ax1)
        ax1.set_title('Fractional Assignment Matrix')
        ax1.set_xlabel('Cluster Index')
        ax1.set_ylabel('Point Index')
        
        # Plot 2: Group-wise average assignments
        ax2 = fig.add_subplot(gs[1])
        unique_groups = np.unique(group_labels)
        group_avg_assignments = []
        group_names = []
        
        for group in unique_groups:
            group_mask = group_labels == group
            group_avg = np.mean(x_frac[group_mask], axis=0)
            group_avg_assignments.append(group_avg)
            group_names.append(f'Group {int(group)}')
        
        group_avg_assignments = np.array(group_avg_assignments)
        
        # Plot group-wise averages as a bar plot
        x = np.arange(len(centers))
        width = 0.35
        
        for i, group_avg in enumerate(group_avg_assignments):
            ax2.bar(x + i*width, group_avg, width, label=group_names[i])
        
        ax2.set_title('Average Fractional Assignment by Group')
        ax2.set_xlabel('Cluster Index')
        ax2.set_ylabel('Average Assignment Value')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels([f'C{i+1}' for i in range(len(centers))])
        ax2.legend()
        
        # Plot 3: Assignment edges
        ax3 = fig.add_subplot(gs[2])
        plot_assignment_edges(x_frac, group_labels, centers, ax3)
        
        # Add overall title
        plt.suptitle(title, y=1.05)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot with error handling
        output_path = os.path.join(output_dir, save_name)
        try:
            # First try saving with high DPI
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        except Exception as e:
            print(f"Warning: Failed to save with high DPI: {e}")
            try:
                # If that fails, try saving with lower DPI
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
            except Exception as e:
                print(f"Warning: Failed to save with lower DPI: {e}")
                try:
                    # If that fails too, try saving without DPI specification
                    plt.savefig(output_path, bbox_inches='tight')
                except Exception as e:
                    print(f"Error: Failed to save plot: {e}")
                    raise
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        raise
    finally:
        # Always close the figure to prevent memory leaks
        plt.close(fig)
    
    # Print some statistics
    print("\nLP Fractional Solution Statistics:")
    print(f"Number of points: {len(x_frac)}")
    print(f"Number of clusters: {len(centers)}")
    if lambda_param is not None:
        print(f"Lambda parameter: {lambda_param:.3f}")
    print("\nGroup-wise statistics:")
    for i, group in enumerate(unique_groups):
        group_mask = group_labels == group
        group_assignments = x_frac[group_mask]
        print(f"\nGroup {int(group)}:")
        print(f"  Size: {np.sum(group_mask)}")
        print(f"  Mean assignment: {np.mean(group_assignments):.3f}")
        print(f"  Max assignment: {np.max(group_assignments):.3f}")
        print(f"  Min assignment: {np.min(group_assignments):.3f}")
    
    # If data_points are provided, also plot the clustering and assignment lines
    if data_points is not None:
        # Plot clustering
        clustering_save_name = save_name.replace("fractional", "clustering")
        plot_lp_clustering(
            x_frac=x_frac,
            group_labels=group_labels,
            centers=centers,
            data_points=data_points,
            output_dir=output_dir,
            title="LP Clustering Solution",
            save_name=clustering_save_name,
            lambda_param=lambda_param
        )
        
        # Plot assignment lines
        lines_save_name = save_name.replace("fractional", "assignment_lines")
        plot_assignment_lines(
            x_frac=x_frac,
            group_labels=group_labels,
            centers=centers,
            data_points=data_points,
            output_dir=output_dir,
            title="Assignment Lines",
            save_name=lines_save_name,
            lambda_param=lambda_param
        )

def plot_lp_fractional_from_results(results_file, output_dir="results"):
    """
    Plot LP fractional solution from a results file.
    
    Args:
        results_file (str): Path to the results JSON file
        output_dir (str): Directory to save the plot
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    x_frac = np.array(results['x_frac'])
    group_labels = np.array(results['group_labels'])
    centers = np.array(results['centers'])
    lambda_param = results.get('lambda_param', None)  # Get lambda if it exists
    
    # Generate output filename from input filename
    input_name = Path(results_file).stem
    if lambda_param is not None:
        save_name = f"{input_name}_lambda{lambda_param:.3f}_lp_fractional.png"
    else:
        save_name = f"{input_name}_lp_fractional.png"
    
    # Plot
    plot_lp_fractional_solution(
        x_frac=x_frac,
        group_labels=group_labels,
        centers=centers,
        output_dir=output_dir,
        title=f"LP Fractional Solution - {input_name}",
        save_name=save_name,
        lambda_param=lambda_param
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot LP fractional solution from results file")
    parser.add_argument("--results_file", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save plots")
    
    args = parser.parse_args()
    
    plot_lp_fractional_from_results(args.results_file, args.output_dir) 