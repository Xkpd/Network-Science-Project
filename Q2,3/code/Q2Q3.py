# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %% [markdown]
# ### Question 2  
#
# How has the collaboration network and its properties evolved over time? That is,
# the program should be able to analyze the network properties over time (at
# yearly granularity).

# %%
import pickle

with open("2ndcumulative_collaboration_graph.pickle", "rb") as f:
    real_G = pickle.load(f)

# And to load your stats DataFrame:
import pandas as pd
real_stats_df = pd.read_csv("2ndyearly_network_stats.csv", index_col="Year")

real_G.number_of_nodes(), "nodes,", real_G.number_of_edges(), "edges"
real_stats_df

# %% [markdown]
# display result

# %%
# -------------------------------
# Graph 1: Real Number of Nodes and Real Number of Edges
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(real_stats_df.index, real_stats_df['real_num_nodes'], marker='o', label='Real Number of Nodes')
plt.plot(real_stats_df.index, real_stats_df['real_num_edges'], marker='o', label='Real Number of Edges')
plt.title('Real Number of Nodes and Edges Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Graph 2: Average Degree and Maximum Degree Over Time
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(real_stats_df.index, real_stats_df['real_avg_degree'], marker='o', label='Real Average Degree')
plt.plot(real_stats_df.index, real_stats_df['real_max_degree'], marker='o', label='Real Maximum Degree')
plt.title('Degree Over Time')
plt.xlabel('Year')
plt.ylabel('Degree')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Graph 3: Average Clustering Coefficient Over Time
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(real_stats_df.index, real_stats_df['real_avg_clustering'], marker='o', label='Real Avg Clustering')
plt.title('Average Clustering Coefficient Over Time')
plt.xlabel('Year')
plt.ylabel('Clustering Coefficient')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Graph 4: Number of Connected Components Over Time
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(real_stats_df.index, real_stats_df['real_num_components'], marker='o', label='Real Components')
plt.title('Number of Connected Components Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Components')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Graph 5: Largest Connected Component Size Over Time
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(real_stats_df.index, real_stats_df['real_largest_cc_size'], marker='o', label='Real Largest CC Size')
plt.title('Largest Connected Component Size Over Time')
plt.xlabel('Year')
plt.ylabel('Size of Largest CC')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Graph 6: Shortest Path Over Time
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(real_stats_df.index, real_stats_df['real_avg_shortest_path_length'], marker='o', label='Avg Shortest Path Length')
plt.title('Largest Connected Component Size Over Time')
plt.xlabel('Year')
plt.ylabel('Size of Largest CC')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Graph 7: Shortest Path and Closeness Centrality Metrics Over Time
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(real_stats_df.index, real_stats_df['real_avg_closeness'], marker='o', label='Avg Closeness Centrality')
plt.plot(real_stats_df.index, real_stats_df['real_max_closeness'], marker='o', label='Max Closeness Centrality')
plt.title('Shortest Path & Closeness Centrality Over Time')
plt.xlabel('Year')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### Question 3
#
# Assume that we create a random network from the set of individuals in the input
# file. How does the properties of this network differ from the real collaboration
# network in (1)?

# %%
import pickle

with open("2ndRandomcumulative_collaboration_graph.pickle", "rb") as f:
    G_random = pickle.load(f)

# And to load your stats DataFrame:
import pandas as pd
random_stats_df = pd.read_csv("2ndRandom_network_stats_final.csv")

G_random.number_of_nodes(), "nodes,", G_random.number_of_edges(), "edges"
random_stats_df

# %% [markdown]
# plot graph

# %%

# Create a figure with two subplots (side by side)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Compute spring layouts for a good and consistent layout (fixing a seed for reproducibility)
pos_real = nx.spring_layout(real_G, seed=42)
pos_random = nx.spring_layout(G_random, seed=42)

# Draw the real network on the first subplot
nx.draw_networkx_nodes(real_G, pos_real, node_size=50, node_color='skyblue', ax=axes[0])
nx.draw_networkx_edges(real_G, pos_real, alpha=0.5, ax=axes[0])
axes[0].set_title("Real Collaboration Network", fontsize=16)
axes[0].axis('off')

# Draw the random network on the second subplot
nx.draw_networkx_nodes(G_random, pos_random, node_size=50, node_color='lightgreen', ax=axes[1])
nx.draw_networkx_edges(G_random, pos_random, alpha=0.5, ax=axes[1])
axes[1].set_title("Random Network", fontsize=16)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# plotting

# %%
import matplotlib.pyplot as plt
import networkx as nx
import collections

# -------------------------------------------
# 1. Plot Degree vs. Number of Nodes
# -------------------------------------------

# Compute the degree distribution for G (real network)
degree_sequence_G = [d for n, d in real_G.degree()]
degree_count_G = collections.Counter(degree_sequence_G)
deg_G, count_G = zip(*sorted(degree_count_G.items()))

# Compute the degree distribution for G_random (random network)
degree_sequence_R = [d for n, d in G_random.degree()]
degree_count_R = collections.Counter(degree_sequence_R)
deg_R, count_R = zip(*sorted(degree_count_R.items()))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].bar(deg_G, count_G, color='skyblue')
axes[0].set_title("G: Degree vs. Number of Nodes")
axes[0].set_xlabel("Degree")
axes[0].set_ylabel("Number of Nodes")

axes[1].bar(deg_R, count_R, color='lightgreen')
axes[1].set_title("G_random: Degree vs. Number of Nodes")
axes[1].set_xlabel("Degree")
axes[1].set_ylabel("Number of Nodes")

plt.tight_layout()
plt.show()


# -------------------------------------------
# 2. Plot Degree vs. Probability
# -------------------------------------------

# For G (real network): Normalize counts to get probability distribution
total_nodes_G = real_G.number_of_nodes()
prob_G = [cnt / total_nodes_G for cnt in count_G]

# For G_random (random network): Normalize counts to get probability distribution
total_nodes_R = G_random.number_of_nodes()
prob_R = [cnt / total_nodes_R for cnt in count_R]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].bar(deg_G, prob_G, color='skyblue')
axes[0].set_title("G: Degree vs. Probability")
axes[0].set_xlabel("Degree")
axes[0].set_ylabel("Probability")

axes[1].bar(deg_R, prob_R, color='lightgreen')
axes[1].set_title("G_random: Degree vs. Probability")
axes[1].set_xlabel("Degree")
axes[1].set_ylabel("Probability")

plt.tight_layout()
plt.show()


# -------------------------------------------
# 3. Plot Histogram of Clustering Coefficient Distribution
# -------------------------------------------

# Compute the local clustering coefficients
clust_G = list(nx.clustering(real_G).values())
clust_R = list(nx.clustering(G_random).values())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(clust_G, bins=20, color='skyblue', edgecolor='black')
axes[0].set_title("G: Clustering Coefficient Distribution")
axes[0].set_xlabel("Clustering Coefficient")
axes[0].set_ylabel("Frequency")

axes[1].hist(clust_R, bins=20, color='lightgreen', edgecolor='black')
axes[1].set_title("G_random: Clustering Coefficient Distribution")
axes[1].set_xlabel("Clustering Coefficient")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# -------------------------------------------
# 4. Plot Node Degree vs. Local Clustering Coefficient
# -------------------------------------------

# For G: Create lists of degree and clustering coefficient values
deg_dict_G = dict(real_G.degree())
clust_dict_G = nx.clustering(real_G)
x_G = list(deg_dict_G.values())
y_G = [clust_dict_G[node] for node in real_G.nodes()]

# For G_random: Create lists of degree and clustering coefficient values
deg_dict_R = dict(G_random.degree())
clust_dict_R = nx.clustering(G_random)
x_R = list(deg_dict_R.values())
y_R = [clust_dict_R[node] for node in G_random.nodes()]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(x_G, y_G, color='skyblue', alpha=0.7)
axes[0].set_title("G: Node Degree vs. Local Clustering Coefficient")
axes[0].set_xlabel("Degree")
axes[0].set_ylabel("Clustering Coefficient")

axes[1].scatter(x_R, y_R, color='lightgreen', alpha=0.7)
axes[1].set_title("G_random: Node Degree vs. Local Clustering Coefficient")
axes[1].set_xlabel("Degree")
axes[1].set_ylabel("Clustering Coefficient")

plt.tight_layout()
plt.show()

# %%
# filename: compare_real_vs_random_with_igraph.py

import pickle
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt

# --- 1. Load your graphs from pickle (these are NetworkX Graphs) ---
with open("2ndcumulative_collaboration_graph.pickle", "rb") as f:
    real_nx = pickle.load(f)

with open("2ndRandomcumulative_collaboration_graph.pickle", "rb") as f:
    rand_nx = pickle.load(f)

# --- 2. Convert to igraph.Graph for C‑backed algorithms ---
g_real = ig.Graph.TupleList(real_nx.edges(), directed=False)
g_rand = ig.Graph.TupleList(rand_nx.edges(), directed=False)

# --- 3. Compute centralities ---

# a) Degree centrality = degree / (n-1)
n_real = g_real.vcount()
deg_cent_real = [deg/(n_real-1) for deg in g_real.degree()]

n_rand = g_rand.vcount()
deg_cent_rand = [deg/(n_rand-1) for deg in g_rand.degree()]

# b) Betweenness centrality (unnormalized by default in igraph)
betw_real = g_real.betweenness()
betw_rand = g_rand.betweenness()

# c) Closeness centrality
clos_real = g_real.closeness()
clos_rand = g_rand.closeness()

# --- 4. Plotting ---

# Plot 1: Degree Centrality vs Node Index
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(range(n_real), deg_cent_real, color='skyblue', alpha=0.7)
axes[0].set(title="Real Graph: Degree Centrality", xlabel="Node Index", ylabel="Degree Centrality")
axes[1].scatter(range(n_rand), deg_cent_rand, color='lightgreen', alpha=0.7)
axes[1].set(title="Random Graph: Degree Centrality", xlabel="Node Index", ylabel="Degree Centrality")
plt.tight_layout()
plt.show()

# Plot 2: Betweenness Centrality vs Node Index
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(range(n_real), betw_real, color='skyblue', alpha=0.7)
axes[0].set(title="Real Graph: Betweenness Centrality", xlabel="Node Index", ylabel="Betweenness")
axes[1].scatter(range(n_rand), betw_rand, color='lightgreen', alpha=0.7)
axes[1].set(title="Random Graph: Betweenness Centrality", xlabel="Node Index", ylabel="Betweenness")
plt.tight_layout()
plt.show()

# Plot 3: Histogram of Closeness Centrality
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(clos_real, bins=20, color='skyblue', edgecolor='black')
axes[0].set(title="Real Graph: Closeness Centrality", xlabel="Closeness", ylabel="Frequency")
axes[1].hist(clos_rand, bins=20, color='lightgreen', edgecolor='black')
axes[1].set(title="Random Graph: Closeness Centrality", xlabel="Closeness", ylabel="Frequency")
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

# Create a figure with two subplots (one for the real network and one for the random network)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left subplot: Real network scatter plot
axes[0].scatter(real_stats_df['real_max_degree'], real_stats_df['real_largest_cc_size'], 
                color='blue', marker='o', s=80, label='Real')
axes[0].set_title('Real Network: k_max vs Giant Component Size')
axes[0].set_xlabel('Maximum Degree (k_max)')
axes[0].set_ylabel('Giant Component Size')
axes[0].grid(True)

# Optionally, annotate each point with its year
for i, year in enumerate(real_stats_df.index):
    axes[0].annotate(str(year),
                     (real_stats_df['real_max_degree'].iloc[i], real_stats_df['real_largest_cc_size'].iloc[i]),
                     textcoords="offset points", xytext=(5,5), fontsize=8)

# Right subplot: Random network scatter plot
axes[1].scatter(random_stats_df['rand_max_degree'], random_stats_df['rand_largest_cc_size'], 
                color='green', marker='o', s=80, label='Random')
axes[1].set_title('Random Network: k_max vs Giant Component Size')
axes[1].set_xlabel('Maximum Degree (k_max)')
axes[1].set_ylabel('Giant Component Size')
axes[1].grid(True)

# Optionally, annotate each point with its year
for i, year in enumerate(real_stats_df.index):
    axes[1].annotate(str(year),
                     (random_stats_df['rand_max_degree'].iloc[i], random_stats_df['rand_largest_cc_size'].iloc[i]),
                     textcoords="offset points", xytext=(5,5), fontsize=8)

plt.tight_layout()
plt.show()
