import numpy as np
from cplex import Cplex
from scipy.spatial.distance import cdist


def proportional_color_mixing_lp_solver(df, centers, group_labels, alpha, beta, lambda_param, upweight=1.0):
    n, k = len(df), len(centers)
    colors = np.unique(group_labels)
    m = len(colors)

    color_to_indices = {h: np.where(group_labels == h)[0] for h in colors}
    color_sizes = {h: len(color_to_indices[h]) for h in colors}
    r_h = {h: len(color_to_indices[h]) / n for h in colors}

    # Handle both pandas DataFrame and numpy array
    if hasattr(df, 'values'):
        df_values = df.values
    else:
        df_values = df

    distance_matrix = cdist(df_values, centers, metric="sqeuclidean")

    problem = Cplex()
    #problem.parameters.randomseed.set(42)
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Create x_{j,i} variables
    x_names = [f"x_{j}_{i}" for j in range(n) for i in range(k)]
    x_costs = distance_matrix.ravel().tolist()
    problem.variables.add(obj=[0.0]*len(x_costs), lb=[0.0]*len(x_costs), ub=[1.0]*len(x_costs), names=x_names)

    # Create auxiliary variables: z, p_{i,h}, q_{i,h}, t_{i,h}
    z_name = ["z"]
    problem.variables.add(obj=[1.0], names=z_name, lb=[0.0])

    p_names = [f"p_{i}_{h}" for i in range(k) for h in colors]
    q_names = [f"q_{i}_{h}" for i in range(k) for h in colors]
    t_names = [f"t_{i}_{h}" for i in range(k) for h in colors]
    problem.variables.add(obj=[0.0]*(3 * k * m), names=p_names + q_names + t_names, lb=[0.0]*(3 * k * m))

    # Constraint: each point assigned to one center
    for j in range(n):
        row = [f"x_{j}_{i}" for i in range(k)]
        problem.linear_constraints.add(
            lin_expr=[[row, [1.0]*k]], senses=["E"], rhs=[1.0], names=[f"assign_{j}"])

    # Fairness constraints and D_h upper bound constraints
    for h in colors:
        h_idx = int(h)
        indices_h = color_to_indices[h]
        group_size = float(color_sizes[h])
        dist_term = []
        t_term = []

        for i in range(k):
            # Deviation constraints
            x_all = [f"x_{j}_{i}" for j in range(n)]
            x_group = [f"x_{j}_{i}" for j in indices_h]

            lhs_1 = [[f"p_{i}_{h}"], [-1.0]] + [x_all, [(r_h[h] - beta[h]) for _ in x_all]] + [x_group, [-1.0]*len(x_group)]
            lhs_2 = [[f"q_{i}_{h}"], [-1.0]] + [x_group, [1.0]*len(x_group)] + [x_all, [-(r_h[h] + alpha[h]) for _ in x_all]]

            problem.linear_constraints.add(lin_expr=[lhs_1], senses=["G"], rhs=[0.0])
            problem.linear_constraints.add(lin_expr=[lhs_2], senses=["G"], rhs=[0.0])

            # t_{i,h} >= p_{i,h}, t_{i,h} >= q_{i,h}
            problem.linear_constraints.add(
                lin_expr=[[[f"t_{i}_{h}", f"p_{i}_{h}"], [1.0, -1.0]]], senses=["G"], rhs=[0.0])
            problem.linear_constraints.add(
                lin_expr=[[[f"t_{i}_{h}", f"q_{i}_{h}"], [1.0, -1.0]]], senses=["G"], rhs=[0.0])

            # Accumulate for D_h
            #t_term.append((f"t_{i}_{h}", 1.0))
            t_term.append((f"t_{i}_{h}", (1 - lambda_param) * upweight))

            for j in indices_h:
                dist_term.append((f"x_{j}_{i}", lambda_param * distance_matrix[j][i]))

        # Final D_h <= z constraint
        vars_dh, coefs_dh = zip(*dist_term + t_term)
        coefs_dh = list(coefs_dh)
        coefs_dh = [coef / group_size for coef in coefs_dh]
        vars_dh = list(vars_dh) + ["z"]
        coefs_dh += [-1.0]
        problem.linear_constraints.add(
            lin_expr=[[vars_dh, coefs_dh]], senses=["L"], rhs=[0.0], names=[f"D_upper_{h}"])

    problem.write("model.lp")
    problem.solve()

    x_vals = problem.solution.get_values()
    x_var_names = problem.variables.get_names()

    x_dict = dict(zip(x_var_names, x_vals))  # Map variable name → value

    # Recover n and k
    n = len(df)
    k = len(centers)

    # Get t values
    t_vals = {name: val for name, val in x_dict.items() if name.startswith("t_")}

    color_to_indices = {h: np.where(group_labels == h)[0] for h in np.unique(group_labels)}

    for h in np.unique(group_labels):
        indices_h = color_to_indices[h]
        group_size = len(indices_h)

        # === Term 1: Distance cost contribution
        dist_sum = 0.0
        for j in indices_h:
            for i in range(k):
                var_name = f"x_{j}_{i}"
                dist_sum += lambda_param * distance_matrix[j, i] * x_dict.get(var_name, 0.0)
        dist_term = dist_sum / group_size

        # === Term 2: Deviation penalty contribution
        dev_sum = 0.0
        for i in range(k):
            var_name = f"t_{i}_{h}"
            dev_sum += t_vals.get(var_name, 0.0)
        dev_term = ((1 - lambda_param) * upweight * dev_sum) / group_size

        print(f"\n[Group {h}]")
        print(f"Distance term (λ * avg cost): {dist_term:.5f}")
        print(f"Deviation term ((1-λ) * avg dev): {dev_term:.5f}")
        print(f"Combined group cost (before max): {dist_term + dev_term:.5f}")











    if problem.solution.get_status() != 1:
        raise RuntimeError("CPLEX did not find an optimal solution.")

    sol = problem.solution.get_values()
    x_frac = np.array(sol[:n*k]).reshape(n, k)
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
