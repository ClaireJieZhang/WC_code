

import numpy as np
import networkx as nx
from collections import defaultdict
# epsilon is used for clipping 
epsilon = 0.001
scale_up_factor = 10000

# NOTE: this assumes that x is a 2d numpy array  
def find_proprtions(x,num_colors,color_flag,num_clusters):
    proportions = np.zeros((num_colors,num_clusters))
    for color in range(num_colors):
        rel_color_indices = [i for i, x in enumerate(color_flag) if x == color]
        color_sum = np.sum(x[rel_color_indices,:],axis=0)
        for cluster in range(num_clusters):
            proportions[color,cluster] = color_sum[cluster]
    div_total = np.sum(x,axis=0)
    div_total[np.where(div_total == 0)]=1 
    proportions_normalized = proportions/div_total
    return proportions_normalized, proportions

def check_rounding_and_clip(x,epsilon):
    n,m = x.shape
    valid = True 
    for i in range(n):
        row_count = 0 
        for j in range(m):
            # if almost 1 
            if abs(x[i,j]-1)<= epsilon:
                x[i,j] = 1 
                if row_count==1:
                    print('fail')
                    print(x[i,j])
                    valid= False 
                else:
                    row_count+=1 
            # if not almost 1 and not almost 0 
            elif abs(x[i,j]) > epsilon: 
                print('fail')
                print(x[i,j])
                valid= False 
            # if almost 0 
            elif abs(x[i,j]) <= epsilon: 
                x[i,j]=0 

        if row_count ==0:
            print('fail')
            print(x[i,j])
            valid= False

    return valid , x


def dot(K, L):
   if len(K) != len(L):
           return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def vet_x(x,epsilon):
    n,m = x.shape
    valid = True 
    for i in range(n):
        row_count = 0 
        for j in range(m):
            # if almost 1 
            if (x[i,j]+epsilon)<0:
                valid= False
                #print(x[i,j])

    return valid , x

def check_rounding_and_clip(x, epsilon):
    n, m = x.shape
    valid = True 
    for i in range(n):
        row_count = 0 
        for j in range(m):
            if abs(x[i,j] - 1) <= epsilon:
                x[i,j] = 1
                if row_count == 1:
                    valid = False 
                else:
                    row_count += 1
            elif abs(x[i,j]) > epsilon:
                valid = False 
            elif abs(x[i,j]) <= epsilon:
                x[i,j] = 0
        if row_count == 0:
            valid = False
    return valid, x



 

  


def calculate_assignment_cost(x, distance_matrix):
    return np.sum(x * distance_matrix)




def min_cost_rounding_color_specific(df, centers, distance, color_flag, num_colors, res):

    # number of clusters    
    num_clusters = len(centers)
    color_flag = np.array(color_flag)
    
    lp_sol_val = res["partial_objective"]
    # LP fractional assignments in x 
    x = res["partial_assignment"]
    x = np.reshape(x, (-1,num_clusters))
    lp_correct , _ = vet_x(x,epsilon)

    # NOTE: sometimes CPLEX makes mistakes 
    if not lp_correct:
        raise ValueError('Error: LP has negative values. CPLEX Error')


    # number of points 
    n = len(df)
    # distance converted to matrix form and is rounded to integer values 
    d = np.round_(np.reshape(scale_up_factor*distance, (-1,num_clusters)))
    # get the color_flag in list form 
    print('NF Rounding ...')


    unique_colors = np.unique(color_flag)
    x_rounded = np.zeros_like(x)

    for h in unique_colors:
        G = nx.DiGraph()
        point_indices = np.where(color_flag == h)[0]
        num_points_in_h = len(point_indices)
        center_node_map = {}
        total_floor = 0

        for j in point_indices:
            node_j = f'p_{j}'
            G.add_node(node_j, demand=-1)

        for i in range(num_clusters):
            center_node = f'c_{i}_h_{h}'
            x_sum = sum(x[j, i] for j in point_indices)
            x_floor = int(np.floor(x_sum))
            total_floor += x_floor
            G.add_node(center_node, demand=x_floor)
            center_node_map[i] = center_node

        t_node = f't_{h}'
        G.add_node(t_node, demand=(num_points_in_h - total_floor))

        for j in point_indices:
            for i in range(num_clusters):
                if x[j, i] > 0:
                    cost = d[j, i] / num_points_in_h
                    G.add_edge(f'p_{j}', center_node_map[i], weight=cost, capacity=1)

        for i in range(num_clusters):
            G.add_edge(center_node_map[i], t_node, weight=0, capacity=1)

        flow_cost, flow_dict = nx.network_simplex(G)

        for j in point_indices:
            for i in range(num_clusters):
                point_node = f'p_{j}'
                center_node = center_node_map[i]
                if G.has_edge(point_node, center_node):
                    if flow_dict[point_node][center_node] == 1:
                        x_rounded[j, i] = 1


    non_binary = x_rounded[(x_rounded != 0) & (x_rounded != 1)]
    if len(non_binary) > 0:
        print(f"[DEBUG] Non-binary values found in x_rounded: count = {len(non_binary)}")
        print("Example values:", non_binary[:10])


    success_flag , x_rounded = check_rounding_and_clip(x_rounded,epsilon)
    
    if success_flag: 
        print('\nNetwork Flow Rounding Done.\n')
    else: 
        raise ValueError('NF rounding has returned non-integer solution.')

    # Get color proportions for each color and cluster
    lp_proportions_normalized, lp_proportions = find_proprtions(x,num_colors,color_flag,num_clusters)
    rounded_proportions_normalized, rounded_proportions = find_proprtions(x_rounded,num_colors,color_flag,num_clusters)

    # calculate the objective value according to this
    xlp_cost = calculate_assignment_cost(x, distance)
    rounded_cost = calculate_assignment_cost(x_rounded, distance)

    print("[Rounding] LP distance (non SF) cost:", xlp_cost)
    print("[Rounding] Rounded distance (non SF) cost:", rounded_cost)
    print("[Rounding] Normalized proportions in LP:\n", lp_proportions_normalized)
    print("[Rounding] Normalized proportions after rounding:\n", rounded_proportions_normalized)


    #######################################Need to update below final cost to calculate objective value
    final_cost=rounded_cost
    res["objective"] = final_cost
    res['assignment'] = x_rounded 
    rounded_sol_val = final_cost


    res['partial_proportions'] = lp_proportions.ravel().tolist()
    res['proportions'] = rounded_proportions.ravel().tolist()

    res['partial_proportions_normalized'] = lp_proportions_normalized.ravel().tolist()
    res['proportions_normalized'] = rounded_proportions_normalized.ravel().tolist()
    #lp_sol_val = res["partial_objective"] * scale_up_factor
    
    
   
    ratio_rounded_lp_distance_only = rounded_cost / xlp_cost


    #ratio_rounded_lp = rounded_sol_val/lp_sol_val 
    print("ratio_rounded_lp_distance_only")
    print(ratio_rounded_lp_distance_only)

    if (ratio_rounded_lp_distance_only-epsilon)>1:
        raise ValueError('NF rounding has higher cost. Try increasing scale_up_factor.') 
    else:
        pass 
        #print('\n---------\nratio= rounded_sol_val / lp_sol_val = %f' %  ratio_rounded_lp )
    return res 


def rounding_wrapper(
    lp_assignment,
    distance_matrix,
    color_labels,
    num_clusters,
    num_colors,
    lp_objective,
    df,
    centers,
    epsilon=1e-6
):
    print(">>> Entered rounding_wrapper")

    is_valid_lp, _ = vet_x(lp_assignment, epsilon)
    if not is_valid_lp:
        raise ValueError("Invalid LP solution: negative assignment detected.")

    # Step 1: Prepare LP info
    res = {
        "partial_assignment": lp_assignment,
        "partial_objective": lp_objective,
    }

    # Step 2: Run rounding
    res = min_cost_rounding_color_specific(df, centers, distance_matrix, color_labels, num_colors, res)
    x_rounded = np.reshape(res["assignment"], (len(df), num_clusters))  # ensure matrix

    print(">>> Got x_rounded back")

    # Step 3: Validate rounding result shape and cost (redundant only if you want extra safety)
    is_valid_rounded, x_rounded = check_rounding_and_clip(x_rounded, epsilon)
    if not is_valid_rounded:
        raise ValueError("Invalid rounded assignment: not a valid hard clustering.")

    return res

