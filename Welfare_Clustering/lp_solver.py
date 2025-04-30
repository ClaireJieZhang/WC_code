import numpy as np
from cplex import Cplex
from scipy.spatial.distance import cdist
import sys



def proportional_color_mixing_lp_solver(df, centers, group_labels, alpha, beta, lambda_param, upweight=1.0):
    n, k = len(df), len(centers)
    group_labels = np.array(group_labels, dtype=int)
    colors = sorted(list(set(group_labels)))
    m = len(colors)

    color_to_indices = {int(h): np.where(group_labels == h)[0] for h in colors}
    color_sizes = {int(h): len(color_to_indices[h]) for h in colors}
    r_h = {int(h): color_sizes[h] / n for h in colors}
    alpha = {int(h): alpha[h] for h in colors}
    beta = {int(h): beta[h] for h in colors}

    df_values = df.values if hasattr(df, 'values') else df
    distance_matrix = cdist(df_values, centers, metric="sqeuclidean")

    problem = Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Create x_{j,i} variables
    x_names = [f"x_{j}_{i}" for j in range(n) for i in range(k)]
    x_costs = distance_matrix.ravel().tolist()
    problem.variables.add(obj=[0.0]*len(x_costs), lb=[0.0]*len(x_costs), ub=[1.0]*len(x_costs), names=x_names)

    # Create auxiliary variables: z, p_{i,h}, q_{i,h}, t_{i,h}
    problem.variables.add(obj=[1.0], names=["z"], lb=[0.0])
    p_names = [f"p_{i}_{h}" for i in range(k) for h in colors]
    q_names = [f"q_{i}_{h}" for i in range(k) for h in colors]
    t_names = [f"t_{i}_{h}" for i in range(k) for h in colors]
    problem.variables.add(obj=[0.0]*(3 * k * m), names=p_names + q_names + t_names, lb=[0.0]*(3 * k * m))

    # Constraint: each point assigned to one center
    for j in range(n):
        row = [f"x_{j}_{i}" for i in range(k)]
        problem.linear_constraints.add(
            lin_expr=[[row, [1.0]*k]], senses=["E"], rhs=[1.0], names=[f"assign_{j}"])

    # Fairness constraints and D_h constraints
    for h in colors:
        if h not in r_h or h not in alpha or h not in beta or h not in color_to_indices:
            print(f"❌ Skipping group {h} due to missing key in alpha/beta/r_h/maps")
            continue

        indices_h = set(color_to_indices[h])
        group_size = float(color_sizes[h])
        dist_term = []
        t_term = []

        for i in range(k):
            # Build terms for lower bound: p_{i,h} >= (r_h - beta_h) * sum_j x_{j,i} - sum_{j in h} x_{j,i}
            x_all = [f"x_{j}_{i}" for j in range(n)]
            x_group = [f"x_{j}_{i}" for j in color_to_indices[h]]

            # Initialize constraint lists for this (i, h) pair
            vars_lower, coefs_lower = [], []
            vars_upper, coefs_upper = [], []

            for j in range(n):
                var_name = f"x_{j}_{i}"

                # Lower bound (r_h - beta_h)
                if j in indices_h:
                    coef = 1.0 - (r_h[h] - beta[h])
                else:
                    coef = - (r_h[h] - beta[h])
                vars_lower.append(var_name)
                coefs_lower.append(coef)

                # Upper bound (r_h + alpha_h)
                if j in indices_h:
                    coef = 1.0 - (r_h[h] + alpha[h])
                else:
                    coef = - (r_h[h] + alpha[h])
                vars_upper.append(var_name)
                coefs_upper.append(coef)

                # Distance cost term (only for group h)
                if j in indices_h:
                    dist_term.append((var_name, lambda_param * distance_matrix[j, i]))

            # Add slack variables
            vars_lower.append(f"p_{i}_{h}")
            coefs_lower.append(1.0)
            vars_upper.append(f"q_{i}_{h}")
            coefs_upper.append(-1.0)

            problem.linear_constraints.add(
                lin_expr=[[vars_lower, coefs_lower]], senses=["G"], rhs=[0.0],
                names=[f"lower_prop_{i}_{h}"])
            problem.linear_constraints.add(
                lin_expr=[[vars_upper, coefs_upper]], senses=["L"], rhs=[0.0],
                names=[f"upper_prop_{i}_{h}"])

            # t_{i,h} ≥ p_{i,h}, t_{i,h} ≥ q_{i,h}
            problem.linear_constraints.add(
                lin_expr=[[[f"t_{i}_{h}", f"p_{i}_{h}"], [1.0, -1.0]]], senses=["G"], rhs=[0.0])
            problem.linear_constraints.add(
                lin_expr=[[[f"t_{i}_{h}", f"q_{i}_{h}"], [1.0, -1.0]]], senses=["G"], rhs=[0.0])

            t_term.append((f"t_{i}_{h}", (1 - lambda_param) * upweight))

        vars_dh, coefs_dh = zip(*dist_term + t_term)
        vars_dh = list(vars_dh) + ["z"]
        coefs_dh = [c / group_size for c in coefs_dh] + [-1.0]
        problem.linear_constraints.add(
            lin_expr=[[vars_dh, coefs_dh]],
            senses=["L"],
            rhs=[0.0],
            names=[f"D_upper_{h}"]
        )

    # Write model
    problem.parameters.preprocessing.presolve.set(0)
    problem.write("model.lp")

    # Logging
    problem.set_log_stream(sys.stdout)
    problem.set_error_stream(sys.stderr)
    problem.set_warning_stream(sys.stderr)
    problem.set_results_stream(sys.stdout)

    problem.solve()

    status_code = problem.solution.get_status()
    status_string = problem.solution.status[status_code]
    print(f"\n[CPLEX] Solver status code: {status_code} → {status_string}")

    if status_code != 1:
        print(f"[CPLEX] LP status: {status_string} (code {status_code})")
        print("[ConflictRefiner] Refining infeasibility conflict set...")
        problem.conflict.refine()
        conflict_file = "conflict.ilp"
        problem.conflict.write(conflict_file)
        print(f"[ConflictRefiner] Written conflict to {conflict_file}")
        raise RuntimeError(f"CPLEX did not find an optimal solution. Status: {status_string}")

    x_vals = problem.solution.get_values()
    x_var_names = problem.variables.get_names()
    x_dict = dict(zip(x_var_names, x_vals))
    x_frac = np.array(x_vals[:n*k]).reshape(n, k)
    z_val = problem.solution.get_values("z")

    return {"x_frac": x_frac, "z": z_val}




def run_connector_and_solve_lp(df, svar_all, centers, alpha, beta, lambda_param=0.5, upweight=1.0):
    result = proportional_color_mixing_lp_solver(
        df=df,
        centers=centers,
        group_labels=svar_all,
        alpha=alpha,
        beta=beta,
        lambda_param=lambda_param,
        upweight=upweight
    )
    return result
