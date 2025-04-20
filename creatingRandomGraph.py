import pandas as pd
import ast
import pickle
import igraph as ig
import networkx as nx



with open("/Users/xingkun/Desktop/network science/project/Network-Science-Project/Q2,3_Output/cumulative_collaboration_graph.pickle", "rb") as f:
    real_G = pickle.load(f)
real_stats_df = pd.read_csv("/Users/xingkun/Desktop/network science/project/Network-Science-Project/Q2,3_Output/yearly_network_stats.csv")

#random_nodes = scientists_df['pid'].astype(str).unique()
#n_random = len(random_nodes)
n_random = real_stats_df['real_num_nodes'].max()

# Get the last year's real edge count
final_year = real_stats_df.index.max()
m_real = int(real_stats_df.loc[final_year, 'real_num_edges'])

# Fast Erdos–Rényi (GNM) with igraph, then convert to NetworkX
g_rand = ig.Graph.Erdos_Renyi(n=n_random, m=m_real, directed=False)
G_random = nx.Graph(g_rand.get_edgelist())

# -----------------------------
# Build an igraph Graph from the NX random graph to compute metrics
# -----------------------------
g = ig.Graph.TupleList(G_random.edges(), directed=False, vertex_name_attr='name')

# Number of nodes and edges
rand_num_nodes = g.vcount()
rand_num_edges = g.ecount()

# Average degree
degrees = g.degree()
rand_avg_degree = sum(degrees) / rand_num_nodes if rand_num_nodes else 0

# Average clustering coefficient (local transitivity)
rand_avg_clustering = g.transitivity_avglocal_undirected()

# Number of connected components and size of largest
comps = g.connected_components()
rand_num_components = len(comps)
lcc = comps.giant()
rand_largest_cc_size = lcc.vcount()

# Maximum degree
rand_max_degree = max(degrees) if degrees else 0

# Average shortest path length on largest component
rand_avg_shortest_path_length = lcc.average_path_length() if lcc.vcount() > 1 else 0

# Closeness centrality on the full graph
close_scores = g.closeness()
rand_avg_closeness = sum(close_scores) / len(close_scores) if close_scores else 0
rand_max_closeness = max(close_scores) if close_scores else 0

# Extend the existing metrics with those from the random network.
rand_stats = {
    'rand_total_nodes': rand_num_nodes,
    'rand_num_edges': rand_num_edges,
    'rand_avg_degree': rand_avg_degree,
    'rand_avg_clustering': rand_avg_clustering,
    'rand_num_components': rand_num_components,
    'rand_largest_cc_size': rand_largest_cc_size,
    'rand_avg_shortest_path_length': rand_avg_shortest_path_length,
    'rand_avg_closeness': rand_avg_closeness,
    'rand_max_closeness': rand_max_closeness,
    'rand_max_degree': rand_max_degree
}

rand_stats_df = pd.DataFrame([rand_stats])
rand_stats_df.to_csv("/Users/xingkun/Desktop/network science/project/Network-Science-Project/2ndRandom_network_stats_final.csv", index=False)

with open("/Users/xingkun/Desktop/network science/project/Network-Science-Project/2ndRandomcumulative_collaboration_graph.pickle", "wb") as f:
    pickle.dump(G_random, f)