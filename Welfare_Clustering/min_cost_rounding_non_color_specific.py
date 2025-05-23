

import numpy as np
import networkx as nx

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



def min_cost_rounding_multi_color(df, centers, distance, color_flag, num_colors, res):
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


    # Define a graph 
    G = nx.DiGraph()

    # Step 1: Add graph vertices with the right demand and color values  
    demand_color_point = [None]*n
    for i in range(n):
        demand_color_point[i] = {'demand':-1, 'color': color_flag[i]} 

    nodes_point = list(range(n)) 
    nodes_attrs_point = zip(nodes_point, demand_color_point)

    # Step 1 DONE 
    G.add_nodes_from(nodes_attrs_point)

    # Step 2: Add colored centers with the right demand and color values
    for color in range(num_colors):

        rel_color_indices = [i for i, x in enumerate(color_flag) if x == color]

        # demand: is an array where the ith index is for the ith cluster  
        demand = np.floor(np.sum(x[rel_color_indices,:],axis=0)).astype(int)
        

        for cluster in range(num_clusters): 
            node_name = 'c'+str(color)+str(cluster)
            G.add_node(node_name, demand=demand[cluster],color=color) 



    # Step 3: Add edges between points and colored centers with the right cost and capacity 
    for color in range(num_colors):
        # get the points with the right color 
        rel_color_indices = [i for i, x in enumerate(color_flag) if x == color]

        for cluster in range(num_clusters): 
            # get the points assigned to this cluster 
            rel_cluster_indices = np.where(x[:,cluster]>0)[0].tolist() 

            assigned_points = list(set(rel_color_indices) & set(rel_cluster_indices)) 

            colored_center = 'c'+str(color)+str(cluster)

            edges = [(i,colored_center,{'capacity':1,'weight':d[i,cluster]}) for i in assigned_points] 

            G.add_edges_from(edges) 


    # Step 4: Add centers (the set S)
    assignment_cluster = np.sum(x,axis=0)
    assignment_cluster_floor = np.floor(assignment_cluster) 

    for cluster in range(num_clusters):
        demands_colors = 0 
        for color in range(num_colors):
            node_name = 'c'+str(color)+str(cluster)
            demands_colors += G.nodes[node_name]['demand']

        demand_cluster_nf = int(assignment_cluster_floor[cluster]-demands_colors)

        node_name = 's'+str(cluster)
        G.add_node(node_name, demand=demand_cluster_nf) 


    # Step 5: Add edges between colored centers and centers 
    for color in range(num_colors):
        rel_color_indices = [i for i, x in enumerate(color_flag) if x == color]

        # assignment is an array where the ith index is for the ith cluster
        assignment = np.sum(x[rel_color_indices,:],axis=0) 
        assignment_floor = np.floor(assignment)

        for cluster in range(num_clusters):
            center_node  = 's'+str(cluster)
            if assignment[cluster] > assignment_floor[cluster]:
                colored_center = 'c'+str(color)+str(cluster)
                G.add_edge(colored_center,center_node,capacity=1,weight=0)
    

    # Step 6: Add t 
    demand_t = int(n- np.sum(assignment_cluster_floor))
    G.add_node('t',demand=demand_t)

    # Step 7: Add edges between the centers and t 
    for cluster in range(num_clusters):
        
        center_node  = 's'+str(cluster)
        if assignment_cluster[cluster] > assignment_cluster_floor[cluster]:
           G.add_edge(center_node,'t',capacity=1,weight=0)
    
    total_demand = sum(nx.get_node_attributes(G, "demand").values())
    print("Total graph demand (should be 0):", total_demand)

    print("📋 Node demand summary (non-zero only):")
    for node, attr in G.nodes(data=True):
        d = attr.get("demand", 0)
        if d != 0:
            print(f"  {node}: demand = {d}")


    # Step 7: Solve the network flow problem 
    flowCost, flowDict = nx.network_simplex(G) 


    # Step 8: convert solution to assignments x 
    x_rounded = np.zeros((n,num_clusters))

    for node, node_flows in flowDict.items(): 
        if type(node) is int: 
            for center, flow in node_flows.items(): 
                if flow==1:
                    string_to_remove = 'c'+str(color_flag[node]) 
                    x_rounded[node,int(center.replace(string_to_remove,''))]=1


    success_flag , x_rounded = check_rounding_and_clip(x_rounded,epsilon)
    
    if success_flag: 
        print('\nNetwork Flow Rounding Done.\n')
    else: 
        raise ValueError('NF rounding has returned non-integer solution.')

    # Get color proportions for each color and cluster
    lp_proportions_normalized, lp_proportions = find_proprtions(x,num_colors,color_flag,num_clusters)
    rounded_proportions_normalized, rounded_proportions = find_proprtions(x_rounded,num_colors,color_flag,num_clusters)

    # calculate the objective value according to this
    x_rounded = x_rounded.ravel()
    distance = distance.ravel()
    final_cost = np.dot(x_rounded, distance)

    res["objective"] = final_cost
    res['assignment'] = x_rounded 
    rounded_sol_val = final_cost


    res['partial_proportions'] = lp_proportions.ravel().tolist()
    res['proportions'] = rounded_proportions.ravel().tolist()

    res['partial_proportions_normalized'] = lp_proportions_normalized.ravel().tolist()
    res['proportions_normalized'] = rounded_proportions_normalized.ravel().tolist()
    #lp_sol_val = res["partial_objective"] * scale_up_factor
    
    
    # ✅ Recompute LP total distance-only cost for fair comparison
    lp_total_cost = np.dot(x.ravel(), distance.ravel())  # x is LP fractional assignment
    
    ratio_rounded_lp_distance_only = rounded_sol_val / lp_total_cost

    print("[INFO] LP distance-only cost:", lp_total_cost)
    print("[INFO] Rounded distance-only cost:", rounded_sol_val)
    print("[INFO] Ratio (rounded / LP):", ratio_rounded_lp_distance_only)

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
    res = min_cost_rounding_multi_color(df, centers, distance_matrix, color_labels, num_colors, res)
    x_rounded = np.reshape(res["assignment"], (len(df), num_clusters))  # ensure matrix

    print(">>> Got x_rounded back")

    # Step 3: Validate rounding result shape and cost (redundant only if you want extra safety)
    is_valid_rounded, x_rounded = check_rounding_and_clip(x_rounded, epsilon)
    if not is_valid_rounded:
        raise ValueError("Invalid rounded assignment: not a valid hard clustering.")

    return res

