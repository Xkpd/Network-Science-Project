import pandas as pd
import ast
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools
from networkx.algorithms import approximation as approx
import igraph as ig

papers_df = pd.read_csv('/Users/xingkun/Desktop/network science/project/Network-Science-Project/output/papers_cleaned copy.csv')
scientists_df = pd.read_csv('/Users/xingkun/Desktop/network science/project/Network-Science-Project/output/scientists_cleaned.csv')

## normalize names by collapsing internal whitespace and stripping outer whitespace
scientists = set(
    scientists_df['name']
        .str.lower()
        .str.strip()
        .apply(lambda x: ' '.join(x.split()))
)
 
def filter_scientists(lst):
    filtered = []
    for a in dict.fromkeys(lst):
        # normalize author string by collapsing all internal whitespace, then strip and lowercase
        name = ' '.join(a.split()).strip().lower()
        if name in scientists:
            filtered.append(a)
        else:
            print(f"Author not in scientists: {a}")
    return filtered

def parse_authors(authors_str):
    if pd.isnull(authors_str):
        return []
    authors_str = authors_str.strip()
    # If the string starts with '[' it is likely a list-like representation
    if authors_str.startswith('['):
        try:
            authors_list = ast.literal_eval(authors_str)
            if isinstance(authors_list, list):
                return [str(a).strip() for a in authors_list]
            else:
                return []
        except Exception:
            return []
    else:
        # Otherwise assume authors are comma-separated
        return [a.strip() for a in authors_str.split(',') if a.strip()]

papers_df['authors_list'] = papers_df['Authors'].apply(parse_authors)

# Precompute deduped author lists and edge lists for performance
papers_df['deduped_authors'] = papers_df['authors_list'].apply(lambda lst: list(dict.fromkeys(lst)))
papers_df['edge_list'] = papers_df['deduped_authors'].apply(lambda lst: list(itertools.combinations(lst, 2)))

year_groups = papers_df.groupby('Year')
real_stats = {}
real_G = nx.Graph()
added_edges = set()
added_nodes = set()

for year, group in year_groups:
    print(f"\n--- Processing year {year} ---") 
    # Take all papers up through this year
    # (Alternatively: just process only the papers of this year but keep edges in cumulative_G)
    
    # batch process this year's new nodes and edges
    year_edges = set().union(*group['edge_list'])
    year_nodes = set().union(*group['deduped_authors'])

    new_edges = year_edges - added_edges
    new_nodes = year_nodes - added_nodes

    real_G.add_nodes_from(new_nodes)
    real_G.add_edges_from(new_edges)

    added_edges |= new_edges
    added_nodes |= new_nodes
    
    # compute degree dictionary once for performance
    degree_dict = dict(real_G.degree())
    real_num_nodes = len(degree_dict)
    real_num_edges = real_G.number_of_edges()
    real_avg_degree = sum(degree_dict.values()) / real_num_nodes
    real_max_degree = max(degree_dict.values())
    # Compute metrics using igraph for exact, optimized C speed
    ig_G = ig.Graph.TupleList(real_G.edges(), directed=False)
    # local clustering coefficient for each node (zeros for degree<2)
    local_clust = ig_G.transitivity_local_undirected(mode="zero")
    # average clustering coefficient is the mean of local coefficients
    real_avg_clustering = sum(local_clust) / len(local_clust)
    # connected components
    ig_comps = ig_G.clusters()
    real_num_components = len(ig_comps)
    # largest connected component
    ig_giant = ig_comps.giant()
    real_largest_cc_size = ig_giant.vcount()
    # average shortest path length on giant component
    real_avg_shortest_path_length = ig_giant.average_path_length()
    # exact closeness centrality for all nodes
    ig_closeness = ig_G.closeness()
    real_avg_closeness = sum(ig_closeness) / len(ig_closeness)
    real_max_closeness = max(ig_closeness)
    
    real_stats[year] = {
        'real_num_nodes': real_num_nodes,
        'real_num_edges': real_num_edges,
        'real_avg_degree': real_avg_degree,
        'real_max_degree': real_max_degree,
        'real_avg_clustering': real_avg_clustering,
        'real_num_components': real_num_components,
        'real_largest_cc_size': real_largest_cc_size,
        'real_avg_shortest_path_length': real_avg_shortest_path_length,
        'real_avg_closeness': real_avg_closeness,
        'real_max_closeness': real_max_closeness
    }


# Convert to DataFrame
real_stats_df = pd.DataFrame.from_dict(real_stats, orient='index').sort_index()

real_stats_df.to_csv("/Users/xingkun/Desktop/network science/project/Network-Science-Project/2ndyearly_network_stats.csv", index_label="Year")
with open("/Users/xingkun/Desktop/network science/project/Network-Science-Project/2ndcumulative_collaboration_graph.pickle", "wb") as f:
    pickle.dump(real_G, f)