import argparse
import pickle
import matplotlib.pyplot as plt

def plot_costs(result_file, save_path=None):
    with open(result_file, "rb") as f:
        results = pickle.load(f)

    ks = sorted(results.keys())
    costs_std = [results[k]['cost'] for k in ks]
    costs_fair = [results[k]['cost_f'] for k in ks]

    # Split group-wise costs
    costs_std_g1 = [c[0] for c in costs_std]
    costs_std_g2 = [c[1] for c in costs_std]
    costs_fair_g1 = [c[0] for c in costs_fair]
    costs_fair_g2 = [c[1] for c in costs_fair]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, costs_std_g1, label="Standard - Group 1", linestyle='--', marker='o')
    plt.plot(ks, costs_std_g2, label="Standard - Group 2", linestyle='--', marker='o')
    plt.plot(ks, costs_fair_g1, label="Fair - Group 1", marker='s')
    plt.plot(ks, costs_fair_g2, label="Fair - Group 2", marker='s')

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Group-wise Cost")
    plt.title("Fair vs. Standard Clustering Costs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")

    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot group-wise clustering costs from result file.")
    parser.add_argument("--file", type=str, required=True, help="Path to saved .pkl result file")
    parser.add_argument("--save", type=str, default=None, help="Path to save the plot as a PNG (optional)")
    args = parser.parse_args()

    plot_costs(args.file, save_path=args.save)
