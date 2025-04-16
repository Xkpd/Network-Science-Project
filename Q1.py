import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# Step 1: Load data
scientists_df = pd.read_csv('./output/scientists_cleaned.csv')
papers_df = pd.read_csv('./output/papers_cleaned.csv')

# Step 2: Normalize names
input_names = set(scientists_df['name'].str.lower().str.strip())

def is_valid_collab(authors):
    try:
        authors = [a.lower().strip() for a in eval(authors)]  # safe since we generated the file
        return [a for a in authors if a in input_names]
    except:
        return []

papers_df['valid_authors'] = papers_df['authors_list'].apply(is_valid_collab)
papers_df = papers_df[papers_df['valid_authors'].apply(len) >= 2]

# Step 3: Build collaboration graph
G = nx.Graph()
G.add_nodes_from(input_names)

for _, row in papers_df.iterrows():
    authors = row['valid_authors']
    for a1, a2 in combinations(authors, 2):
        if G.has_edge(a1, a2):
            G[a1][a2]['weight'] += 1
        else:
            G.add_edge(a1, a2, weight=1)

# Step 4: Analyze graph
print("\n Network Statistics:")
print(f"🔹 Number of nodes (scientists): {G.number_of_nodes()}")
print(f"🔹 Number of edges (collaborations): {G.number_of_edges()}")
print(f"🔹 Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"🔹 Density: {nx.density(G):.4f}")
print(f"🔹 Average clustering coefficient: {nx.average_clustering(G):.4f}")
print(f"🔹 Number of connected components: {nx.number_connected_components(G)}")

# Step 5: Optional: Draw the graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=8)
plt.title("Collaboration Network of Data Scientists", fontsize=14)
plt.tight_layout()
plt.show()
