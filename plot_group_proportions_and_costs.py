import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

# Set up pipeline info
PIPELINES = {
    'SF': {
        'name': 'Samira SF',
        'json_pattern': 'samira_sf_all_k{}_to_{}_detailed.json',
        'color': 'tab:blue',
    },
    'FCBC': {
        'name': 'FCBC',
        'json_pattern': 'fcbc_all_k{}_to_{}_detailed.json',
        'color': 'tab:orange',
    },
    'WC': {
        'name': 'Welfare Clustering',
        'json_pattern': 'welfare_clustering_all_k{}_to_{}_detailed.json',
        'color': 'tab:green',
    },
}

RESULTS_DIR = 'cache/welfare_evaluation'  # Change if needed
DEFAULT_SAVE_DIR = 'plots'


def load_detailed_results(pipeline_key, k_min, k_max):
    pattern = PIPELINES[pipeline_key]['json_pattern']
    path = os.path.join(RESULTS_DIR, pattern.format(k_min, k_max))
    with open(path, 'r') as f:
        return json.load(f)


def ensure_save_dir(save_dir):
    if not save_dir:
        save_dir = DEFAULT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def plot_group_proportions(results, pipeline_name, k, group_names=None, save_dir=None):
    save_dir = ensure_save_dir(save_dir)
    # Filter for this k
    k_results = [r for r in results if int(r['k']) == k]
    if not k_results:
        print(f"No results for k={k} in {pipeline_name}")
        return
    # For SF: plot both standard and fair, for others: just fair
    for res in k_results:
        method = res.get('method', 'fair')
        cluster_stats = res['cluster_stats']
        expected = res['expected_proportions']
        clusters = sorted(cluster_stats.keys(), key=int)
        groups = sorted(expected.keys(), key=str)
        if group_names:
            group_labels = [group_names.get(str(g), str(g)) for g in groups]
        else:
            group_labels = groups
        # Prepare data
        proportions = np.zeros((len(groups), len(clusters)))
        for ci, cluster_id in enumerate(clusters):
            for gi, group in enumerate(groups):
                proportions[gi, ci] = cluster_stats[cluster_id]['group_proportions'].get(group, 0)
        # Debug output for SF
        if pipeline_name == 'Samira SF':
            print(f"[DEBUG] SF Group Proportions for k={k}, method={method}")
            for ci, cluster_id in enumerate(clusters):
                print(f"  Cluster {cluster_id}: ", {g: proportions[gi, ci] for gi, g in enumerate(groups)})
        # Plot
        fig, ax = plt.subplots(figsize=(1.5*len(clusters)+2, 5))
        bar_width = 0.8 / len(groups)
        x = np.arange(len(clusters))
        for gi, group in enumerate(groups):
            bars = ax.bar(x + gi*bar_width, proportions[gi], bar_width, label=f"{group_labels[gi]}")
            for rect in bars:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Proportion')
        ax.set_title(f'{pipeline_name} (k={k}, {method}) - Group Proportions in Clusters')
        ax.set_xticks(x + bar_width * (len(groups)-1)/2)
        ax.set_xticklabels([str(c) for c in clusters])
        ax.legend()
        plt.tight_layout()
        fname = f'{pipeline_name}_k{k}_{method}_proportions.png'
        plt.savefig(os.path.join(save_dir, fname))
        print(f"[SAVED] {fname}")
        plt.close()


def plot_group_distance_costs(all_results, pipeline_name, group_names=None, save_dir=None):
    save_dir = ensure_save_dir(save_dir)
    k_values = sorted(set(int(r['k']) for r in all_results))
    groups = set()
    for r in all_results:
        groups.update(r['group_distance_costs'].keys())
    groups = sorted(groups, key=str)
    if group_names:
        group_labels = [group_names.get(str(g), str(g)) for g in groups]
    else:
        group_labels = groups
    group_costs = {g: [] for g in groups}
    for k in k_values:
        k_results = [r for r in all_results if int(r['k']) == k]
        if not k_results:
            for g in groups:
                group_costs[g].append(np.nan)
            continue
        r = k_results[0]
        for g in groups:
            group_costs[g].append(r['group_distance_costs'].get(g, np.nan))
    fig, ax = plt.subplots(figsize=(2+len(k_values), 5))
    for gi, g in enumerate(groups):
        ax.plot(k_values, group_costs[g], marker='o', label=group_labels[gi])
        for xi, val in zip(k_values, group_costs[g]):
            if not np.isnan(val):
                ax.annotate(f'{val:.2f}', (xi, val), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    ax.set_xlabel('k (Number of Clusters)')
    ax.set_ylabel('Avg. Distance Cost (per group)')
    ax.set_title(f'{pipeline_name} - Per-Group Avg. Distance Cost')
    ax.legend()
    plt.tight_layout()
    fname = f'{pipeline_name}_group_distance_costs.png'
    plt.savefig(os.path.join(save_dir, fname))
    print(f"[SAVED] {fname}")
    plt.close()


def plot_group_violations(results, pipeline_name, k, group_names=None, save_dir=None):
    save_dir = ensure_save_dir(save_dir)
    # Filter for this k
    k_results = [r for r in results if int(r['k']) == k]
    if not k_results:
        print(f"No results for k={k} in {pipeline_name}")
        return
    for res in k_results:
        method = res.get('method', 'fair')
        cluster_stats = res['cluster_stats']
        expected = res['expected_proportions']
        clusters = sorted(cluster_stats.keys(), key=int)
        groups = sorted(expected.keys(), key=str)
        if group_names:
            group_labels = [group_names.get(str(g), str(g)) for g in groups]
        else:
            group_labels = groups
        # Prepare data
        violations = np.zeros((len(groups), len(clusters)))
        for ci, cluster_id in enumerate(clusters):
            for gi, group in enumerate(groups):
                violations[gi, ci] = cluster_stats[cluster_id]['violations'].get(group, 0)
        # Plot
        fig, ax = plt.subplots(figsize=(1.5*len(clusters)+2, 5))
        bar_width = 0.8 / len(groups)
        x = np.arange(len(clusters))
        for gi, group in enumerate(groups):
            bars = ax.bar(x + gi*bar_width, violations[gi], bar_width, label=f"{group_labels[gi]}")
            for rect in bars:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Violation')
        ax.set_title(f'{pipeline_name} (k={k}, {method}) - Group Violations in Clusters')
        ax.set_xticks(x + bar_width * (len(groups)-1)/2)
        ax.set_xticklabels([str(c) for c in clusters])
        ax.legend()
        plt.tight_layout()
        fname = f'{pipeline_name}_k{k}_{method}_violations.png'
        plt.savefig(os.path.join(save_dir, fname))
        print(f"[SAVED] {fname}")
        plt.close()
        # Compute and plot averaged violation per group
        group_sizes = {g: 0 for g in groups}
        group_total_violation = {g: 0.0 for g in groups}
        for ci, cluster_id in enumerate(clusters):
            cluster_size = cluster_stats[cluster_id]['size']
            for gi, group in enumerate(groups):
                group_prop = cluster_stats[cluster_id]['group_proportions'].get(group, 0)
                group_sizes[group] += group_prop * cluster_size
                group_total_violation[group] += violations[gi, ci] * cluster_size
        avg_violation = {g: (group_total_violation[g] / group_sizes[g] if group_sizes[g] > 0 else 0) for g in groups}
        # Plot averaged violation
        fig, ax = plt.subplots(figsize=(2+len(groups), 5))
        bars = ax.bar(range(len(groups)), [avg_violation[g] for g in groups], tick_label=group_labels)
        for bi, rect in enumerate(bars):
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Group')
        ax.set_ylabel('Avg. Violation (normalized)')
        ax.set_title(f'{pipeline_name} (k={k}, {method}) - Avg. Normalized Violation per Group')
        plt.tight_layout()
        fname2 = f'{pipeline_name}_k{k}_{method}_avg_violation.png'
        plt.savefig(os.path.join(save_dir, fname2))
        print(f"[SAVED] {fname2}")
        plt.close()


def plot_k_vs_group_violation(all_results, pipeline_name, group_names=None, save_dir=None):
    save_dir = ensure_save_dir(save_dir)
    k_values = sorted(set(int(r['k']) for r in all_results))
    groups = set()
    for r in all_results:
        for cluster in r['cluster_stats'].values():
            groups.update(cluster['violations'].keys())
    groups = sorted(groups, key=str)
    if group_names:
        group_labels = [group_names.get(str(g), str(g)) for g in groups]
    else:
        group_labels = groups
    group_violations = {g: [] for g in groups}
    for k in k_values:
        k_results = [r for r in all_results if int(r['k']) == k]
        if not k_results:
            for g in groups:
                group_violations[g].append(np.nan)
            continue
        res = k_results[0]
        cluster_stats = res['cluster_stats']
        clusters = sorted(cluster_stats.keys(), key=int)
        group_sizes = {g: 0 for g in groups}
        group_total_violation = {g: 0.0 for g in groups}
        for ci, cluster_id in enumerate(clusters):
            cluster_size = cluster_stats[cluster_id]['size']
            for gi, group in enumerate(groups):
                group_prop = cluster_stats[cluster_id]['group_proportions'].get(group, 0)
                violation = cluster_stats[cluster_id]['violations'].get(group, 0)
                group_sizes[group] += group_prop * cluster_size
                group_total_violation[group] += violation * cluster_size
        for g in groups:
            if group_sizes[g] > 0:
                group_violations[g].append(group_total_violation[g] / group_sizes[g])
            else:
                group_violations[g].append(np.nan)
    fig, ax = plt.subplots(figsize=(2+len(k_values), 5))
    for gi, g in enumerate(groups):
        ax.plot(k_values, group_violations[g], marker='o', label=group_labels[gi])
        for xi, val in zip(k_values, group_violations[g]):
            if not np.isnan(val):
                ax.annotate(f'{val:.2f}', (xi, val), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    ax.set_xlabel('k (Number of Clusters)')
    ax.set_ylabel('Avg. Normalized Violation (per group)')
    ax.set_title(f'{pipeline_name} - Per-Group Avg. Normalized Violation')
    ax.legend()
    plt.tight_layout()
    fname = f'{pipeline_name}_k_vs_group_violation.png'
    plt.savefig(os.path.join(save_dir, fname))
    print(f"[SAVED] {fname}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize group proportions, distance costs, and violations for clustering pipelines.")
    parser.add_argument('--k_min', type=int, required=True)
    parser.add_argument('--k_max', type=int, required=True)
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save plots (optional)')
    parser.add_argument('--group_names', type=str, default=None, help='Optional JSON file mapping group ids to names')
    args = parser.parse_args()

    group_names = None
    if args.group_names:
        with open(args.group_names, 'r') as f:
            group_names = json.load(f)

    for key, info in PIPELINES.items():
        print(f"\n--- {info['name']} ---")
        results = load_detailed_results(key, args.k_min, args.k_max)
        # Plot group proportions for all k
        for k in range(args.k_min, args.k_max+1):
            plot_group_proportions(results, info['name'], k, group_names, args.save_dir)
            plot_group_violations(results, info['name'], k, group_names, args.save_dir)
        # Plot group distance costs for all k
        plot_group_distance_costs(results, info['name'], group_names, args.save_dir)
        # Plot k vs. per-group violation
        plot_k_vs_group_violation(results, info['name'], group_names, args.save_dir)

if __name__ == '__main__':
    main() 