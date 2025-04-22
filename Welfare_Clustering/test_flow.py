import networkx as nx

G = nx.DiGraph()
G.add_node("a", demand=-1)
G.add_node("b", demand=1)
G.add_edge("a", "b", capacity=1, weight=1)

cost, flow = nx.network_simplex(G)
print("Flow cost:", cost)