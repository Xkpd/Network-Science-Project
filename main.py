import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
def main():

    # ========== File Paths ==========
    input_file = './input/datascientists.xls'
    scientists_output_file = './output/scientists_cleaned.csv'
    papers_output_file = 'papers.csv'
    error_log_file = 'error_log.txt'

    # ========== Step 1: Load or Generate Cleaned Scientist Data ==========

    if os.path.exists(scientists_output_file):
        print(f"Found existing '{scientists_output_file}', loading it...")
        scientists_df = pd.read_csv(scientists_output_file)
    else:
        print("'scientists_cleaned.csv' not found. Starting from raw input...")

        initial_df = pd.read_excel(input_file)
        print("Collecting final DBLP URLs and PIDs...")
        pids = []
        final_urls = []
        errors_links = []

        for link in tqdm(initial_df['dblp']):
            retries = 5
            delay = 2
            response = None
            for attempt in range(retries):
                try:
                    response = requests.get(link, timeout=10)
                    if response.status_code == 429:
                        print("Too many requests. Sleeping for 60 seconds...")
                        time.sleep(60)
                        continue
                    elif response.status_code == 410:
                        print(f"{link} is gone (410). Skipping.")
                        response = None
                        break
                    elif response.status_code != 200:
                        raise Exception(f"HTTP Error {response.status_code}")
                    break  # success
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{retries} failed for {link}: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay *= 2
                    else:
                        print(f"Skipping {link} after {retries} failed attempts.")
                        response = None

            if response is None:
                pids.append('Error')
                final_urls.append('Error')
                errors_links.append(link)
                continue

            final_url = response.url
            match = re.search(r'pid/(.*).html', final_url)

            if match:
                pid = match.group(1).replace('/', '-')
                pids.append(pid)
                final_urls.append(final_url)
            else:
                pids.append('Error')
                final_urls.append('Error')
                errors_links.append(link)

        # Save cleaned data
        cleaned_df = initial_df.copy()
        cleaned_df['pid'] = pids
        cleaned_df['final_url'] = final_urls
        cleaned_df = cleaned_df[(cleaned_df['pid'] != 'Error') & (cleaned_df['final_url'] != 'Error')]
        cleaned_df = cleaned_df.drop_duplicates(subset='pid', keep='first')
        cleaned_df = cleaned_df.drop_duplicates()
        cleaned_df.to_csv(scientists_output_file, index=False)
        print(f"Saved {len(cleaned_df)} cleaned scientists to {scientists_output_file}")
        scientists_df = cleaned_df

    # ========== Step 2: Scrape Papers (Parallelized) ==========

    def scrape_scientist(row):
        papers = []
        pid = row['pid']
        url = row['final_url']

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            entries = soup.find_all('li', class_=lambda x: x and x.startswith('entry'))

            for entry in entries:
                title_tag = entry.find('span', class_='title')
                year_tag = entry.find('span', class_='year')
                doi_tag = entry.find('a', title='DOI')
                author_tags = entry.find_all('span', itemprop='author')

                title = title_tag.text.strip() if title_tag else 'N/A'

                # Extract year (fallback to regex)
                if year_tag:
                    year = year_tag.text.strip()
                else:
                    year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', entry.text)
                    valid_years = [int(y) for y in year_matches if int(y) <= 2025]
                    if not valid_years:
                        with open("missing_year_fallback.txt", "a", encoding="utf-8") as log_file:
                            log_file.write(f"No valid year for entry in {pid}:\n{entry.text[:300]}\n\n")
                    year = str(max(valid_years)) if valid_years else 'N/A'


                doi_tag = entry.find('a', href=re.compile(r'(doi\.org|arxiv\.org)'))
                doi = doi_tag['href'].strip() if doi_tag else 'N/A'

                authors = ', '.join([a.text.strip() for a in author_tags]) if author_tags else 'N/A'

                papers.append({
                    'Title': title,
                    'Year': year,
                    'DOI': doi,
                    'Authors': authors,
                    'file': f"{pid}.xml"
                })

        except Exception as e:
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Error for {pid} at {url}: {e}\n")

        return papers


    import pandas as pd
    import os

    # Create output folder if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    log = []  # Collect logs here

    # Load the CSV
    papers_df = pd.read_csv("./output/papers.csv")
    log.append(f"Initial shape: {papers_df.shape[0]} rows, {papers_df.shape[1]} columns")

    # Step 1: Remove exact duplicate rows
    duplicate_rows = papers_df.duplicated()
    log.append(f"Total exact duplicate rows: {duplicate_rows.sum()}")
    papers_df = papers_df.drop_duplicates()

    # Step 2: Check and remove duplicates by key columns
    for col in ['Title']:
        if col in papers_df.columns:
            count = papers_df.duplicated(subset=col, keep=False).sum()
            log.append(f"- Duplicate entries in \"{col}\": {count}")
            papers_df = papers_df.drop_duplicates(subset=col, keep='first')

    # Step 3: Check for missing (NaN) values
    na_counts = papers_df.isna().sum()
    log.append("Missing (NaN) values per column:")
    log.extend([f"  - {col}: {na_counts[col]}" for col in na_counts.index if na_counts[col] > 0])

    # Step 4: Check for empty strings
    empty_counts = (papers_df == '').sum()
    log.append("Empty string entries per column:")
    log.extend([f"  - {col}: {empty_counts[col]}" for col in empty_counts.index if empty_counts[col] > 0])

    # Step 5: Check for any rows that are partially empty
    partially_empty = papers_df.isna().any(axis=1) | (papers_df == '').any(axis=1)
    log.append(f"Rows with any missing (NaN) or empty strings: {partially_empty.sum()}")

    # Final shape
    log.append(f"Final shape after cleaning: {papers_df.shape[0]} rows")

    # Save cleaned file
    cleaned_path = os.path.join(output_dir, "papers_cleaned.csv")
    papers_df.to_csv(cleaned_path, index=False)
    log.append(f"Cleaned data saved to '{cleaned_path}'")

    # Save log
    log_path = os.path.join(output_dir, "cleaning_log.txt")
    with open(log_path, "w", encoding='utf-8') as f:
        for line in log:
            f.write(line + "\n")

        print(f"Cleaning complete. Outputs written to '{output_dir}' folder.")



    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    from itertools import combinations
    import numpy as np
    from networkx.algorithms.community import greedy_modularity_communities

    # ============== Load data ==============
    scientists_df = pd.read_csv('./output/scientists_cleaned.csv')
    papers_df = pd.read_csv('./output/papers_cleaned.csv')

    # ============== Normalize names ==============
    input_names = set(scientists_df['name'].str.lower().str.strip())

    # ============== Filter valid authors in papers ==============
    def filter_authors(authors):
        try:
            authors = [a.lower().strip() for a in authors.split(',')]  # Handle comma-separated string
            return [a for a in authors if a in input_names]
        except Exception as e:
            print(f"Error processing authors: {e}")
            return []

    papers_df['valid_authors'] = papers_df['Authors'].apply(filter_authors)
    print(papers_df[['Title', 'valid_authors']].head())
    papers_df = papers_df[papers_df['valid_authors'].apply(len) >= 2]

    # ============== Build graph ==============
    G = nx.Graph()
    G.add_nodes_from(input_names)

    for _, row in papers_df.iterrows():
        for a1, a2 in combinations(row['valid_authors'], 2):
            if G.has_edge(a1, a2):
                G[a1][a2]['weight'] += 1
            else:
                G.add_edge(a1, a2, weight=1)

    # ============== Network properties ==============
    print("=== Network Properties ===")
    print(f"Total Nodes (in input list): {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    print(f"Average Degree: {np.mean([deg for _, deg in G.degree()]):.2f}")
    print(f"Network Density: {nx.density(G):.4f}")
    print(f"Number of Connected Components: {nx.number_connected_components(G)}")
    print(f"Average Clustering Coefficient: {nx.average_clustering(G):.4f}")

    # ============== Figure 1: Degree Distribution ==============
    degrees = [deg for node, deg in G.degree()]
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
    plt.title("Figure 1: Degree Distribution of the Collaboration Network")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/Q1_Output/figure1_degree_distribution.png")
    plt.close()

    # ============== Figure 2: Giant Component ==============
    giant_component = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(giant_component)

    print("\n=== Giant Component Properties ===")
    print(f"Nodes in Giant Component: {G_giant.number_of_nodes()}")
    print(f"Edges in Giant Component: {G_giant.number_of_edges()}")
    if nx.is_connected(G_giant):
        print(f"Diameter: {nx.diameter(G_giant)}")
        print(f"Average Shortest Path Length: {nx.average_shortest_path_length(G_giant):.2f}")
    else:
        print("Giant component is not fully connected (unexpected).")
    # Find isolated nodes (nodes that are not in the giant component)
    isolated_nodes = list(set(G.nodes()) - set(giant_component))
    print(f"Number of Isolated Nodes: {len(isolated_nodes)}")

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G_giant, seed=42)
    nx.draw(G_giant, pos, node_size=20, edge_color='gray', node_color='skyblue', with_labels=False)
    plt.title(f"Figure 2: Giant Component ({G_giant.number_of_nodes()} nodes, {G_giant.number_of_edges()} edges)")
    plt.tight_layout()
    plt.savefig("output/Q1_Output/figure2_giant_component.png")
    plt.close()

    # ============== Figure 3: Largest Clique ==============
    cliques = list(nx.find_cliques(G))
    largest_clique = max(cliques, key=len)
    G_clique = G.subgraph(largest_clique)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G_clique, seed=42)  # Changed layout for better visualization
    nx.draw(G_clique, pos, with_labels=True, node_color='orange', edge_color='black', node_size=300, font_size=8)
    plt.title(f"Figure 3: Largest Clique in the Network (n={len(largest_clique)})")
    plt.tight_layout()
    plt.savefig("output/Q1_Output/figure3_largest_clique.png")
    plt.close()

    #============== Figure 4: Degree Centrality vs Degree ==============
    degree_centrality = nx.degree_centrality(G)
    degrees_dict = dict(G.degree())
    x = list(degrees_dict.values())
    y = [degree_centrality[node] for node in degrees_dict]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='green', alpha=0.6)
    plt.title("Figure 4: Degree Centrality vs Degree")
    plt.xlabel("Degree")
    plt.ylabel("Degree Centrality")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/Q1_Output/figure4_degree_vs_centrality.png")
    plt.close()

    # ============== Figure 5: Degree Centrality vs Betweenness Centrality  ==============
    betweenness_centrality = nx.betweenness_centrality(G)
    x = [degree_centrality[node] for node in G.nodes()]
    y = [betweenness_centrality[node] for node in G.nodes()]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='purple', alpha=0.6)
    plt.title("Figure 5: Degree Centrality vs Betweenness Centrality")
    plt.xlabel("Degree Centrality")
    plt.ylabel("Betweenness Centrality")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/Q1_Output/figure5_degree_vs_betweenness.png")
    plt.close()

    # ============ Top 10 Most Central Scientists  ==============
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n=== Top 10 Most Central Scientists (Betweenness Centrality) ===")
    for name, centrality in top_betweenness:
        print(f"{name.title()}: {centrality:.4f}")

    # ============ Figure 6 Ego Network ==============
    central_person = max(betweenness_centrality, key=betweenness_centrality.get)
    central_node_degree = G.degree(central_person)
    all_node_degrees = dict(G.degree())
    ego_network = nx.ego_graph(G, central_person)
    ego_network_size = len(ego_network.nodes)

    plt.figure(figsize=(8, 6))
    pos_ego = nx.spring_layout(ego_network)
    nx.draw(ego_network, pos_ego, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=300, font_size=8)
    plt.title(f"Figure 6: Ego Network of {central_person.title()}")
    plt.tight_layout()
    plt.savefig("output/Q1_Output/figure6_ego_network.png")
    plt.close()

    print(f"The central node is: {central_person.title()}")
    print(f"The degree of the central node is: {central_node_degree}")
    print(f"The number of nodes in the ego network of {central_person.title()} is: {ego_network_size}")

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
        pickle.dump(G_random, f)# ---
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
    stats_df = pd.read_csv("2ndyearly_network_stats.csv", index_col="Year")

    real_G.number_of_nodes(), "nodes,", real_G.number_of_edges(), "edges"
    stats_df

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
    # ---
    # jupyter:
    #   jupytext:
    #     text_representation:
    #       extension: .py
    #       format_name: percent
    #       format_version: '1.3'
    #       jupytext_version: 1.17.0
    #   kernelspec:
    #     display_name: Python 3
    #     name: python3
    # ---

    # %% colab={"base_uri": "https://localhost:8080/"} id="xSvcl_nA58Al" outputId="5037e1e5-a03a-42ef-d7fb-4d1eeace37f0"
    # !pip install powerlaw
    # !pip install countryinfo

    # %% id="qDq5_Y4o6CTA"
    # Importing the libraries
    import os  # Used for file operations
    import requests  # Used for making HTTP requests
    from bs4 import BeautifulSoup  # Used for parsing HTML
    import re  # Used for regular expressions
    import pandas as pd  # Used for data manipulation and analysis
    import numpy as np  # Used for numerical computations
    from tqdm import tqdm  # Used for progress bars
    import time  # Used for time operations
    import xml.etree.ElementTree as ET  # Used for parsing XML
    import ipytree # Used for displaying the XML tag hierarchy in a dynamic tree-like structure
    import json  # Used for JSON operations (used for the XML tag hierarchy)
    import ast  # Used for converting string to list
    import matplotlib.pyplot as plt  # Used for plotting graphs
    import networkx as nx  # Used for creating and analyzing the network
    import seaborn as sns  # Used for plotting graphs
    from IPython.display import Image  # Used for displaying images in the notebook
    import imageio.v2 as imageio  # Used for creating GIFs
    import powerlaw  # Used for fitting power-law distributions
    from countryinfo import CountryInfo  # Used for getting country information
    import copy  # Used to copy nested dictionaries
    import random  # Used for random operations
    import plotly.graph_objects as go  # Used for creating interactive plots
    import math  # Used for mathematical operations

    # %% [markdown] id="c6017e710722668d"
    # ## 5. Network Transformation
    # In this section, we will create a function that will transform the existing collaboration network of the data scientists into a new network. The new network will be transformed based on the following goals:
    # - The maximum degree of any node will not exceed beyond a user-specified k_max, referred to as collaboration cutoff, which is typically smaller than the degrees of hubs.
    # - The transformed network has smaller giant component and larger number of isolates than the original collaboration network.
    # - The transformed network is being built to identify links between High Degree (Famous Data Scientists) & Low Degree nodes (their less famous co-authors)  
    #
    # To maintain the diversity of nodes in the network, we decided against removing any nodes from the network. Therefore, the transformed network will have the same no of nodes as the untransformed network. We instead focus on removing unwanted edges from the network. In general every network transformation technique we tried had 2 parts:
    # 1. Define a metric of edge importance
    # 2. Define a policy to prune edges of less importance

    # %% colab={"base_uri": "https://localhost:8080/"} id="H43lG3FW7fri" outputId="3f8d81c6-a45f-44a7-fb7e-7e4391ff20fe"
    # Define the collaboration network from the JSON
    collaboration_network = json.load(open('/content/network.json', 'r'))

    # Simplying collaboration_network (To make the code lighter to read)
    network = {}
    for node, neighbours in collaboration_network.items():
        for neighbour, weight_dict in neighbours.items():
            if node in network:
                network[node][neighbour] = 1  # Weights are not taken into consideration
            else:
                network[node] = {neighbour: 1}
        # Check if the node has no neighbours
        if neighbours == {}:
            network[node] = {}
    print(len(network.keys()))

    # %% [markdown] id="50f1cc6d427880b0"
    # ### 5.1 Best Reduction Model - Hard Cutoff for Normalised Degree Difference
    # In this section we discuss the technique which was the most successful at reducing the size of the giant component. This technique has two parts:
    #
    # I) **Edge Importance Metric** - To measure the importance of each edge we defined a metric Normalised Degree Difference as:
    #
    # $$ NormalisedDegreeDifference (E_{n1, n2}) = {|{C_d(n1) - C_d(n2)}| \over max(C_d(n1), C_d(n2))} $$  
    #
    # where n1 and n2 are nodes in the graph, E_n1_n2 is the edge connecting n1 & n2, and C_d(x) is the degree of node x. The expected values of Normalised Degree Difference for different edges are:
    # 1. Hub - Hub Edges: Small Numerator and Large Denominator => Very Small - Small Value
    # 2. Hub - Normal Edges: Large Numerator and Large Denominator => Medium - Large Value
    # 3. Normal - Normal Edges: Small Numerator and Small Denominator => Small - Medium Value
    #
    # II) **Edge Pruning Policy** - To prune the edges in the network, we define a hard cutoff, calculated based on the user specified k_max. All edges with an edge importance below this threshold will be removed from the network.

    # %% [markdown] id="I_P34ERV8EqI"
    #

    # %% [markdown] id="d79d6859"
    # #### 5.1.1 Calculate Normalised Degree Difference for each Edge

    # %% id="yZlfqeAH7zlV"
    # Calculating Normalised Degree Difference for each edge
    norm_degree_diff_network = copy.deepcopy(network)
    for node, neighbours in network.items():
        for neighbour, weight in neighbours.items():
            norm_degree_diff = abs(len(network[node]) - len(network[neighbour])) / max(
                (len(network[node]), len(network[neighbour])))
            norm_degree_diff_network[node][neighbour] = norm_degree_diff

    # %% colab={"base_uri": "https://localhost:8080/", "height": 646} id="qj9yWvXt77U3" outputId="ccbabb17-2387-48f7-bdb8-6aa908fb2329"
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Visualising the distribution of normalised degree difference in the network
    norm_degree_diff_distribution = []
    for node, neighbours in norm_degree_diff_network.items():
        for neighbour, weight in neighbours.items():
            norm_degree_diff_distribution.append(weight)

    # Change color here
    sns.histplot(data=norm_degree_diff_distribution, kde=True, color='green')
    plt.title('Norm Degree Difference Distribution of Network')
    plt.xlabel('Norm Degree Difference')
    # plt.yscale("log")
    plt.ylabel("Count")
    plt.show()

    print(f"Norm Degree Difference Distribution Statistics:\n{pd.Series(norm_degree_diff_distribution).describe()}")


    # %% [markdown] id="ad306895"
    # #### 5.1.2 Find the minimum threshold to satisfy k_max & Prune Network

    # %% id="DDjkHU2M8Kbw"
    def find_hard_minimum_threshold_weight(network: dict, k_max: int) -> float:
        """Given k_max this function calculates the minimum threshold weight such that
        k_max is enforced when all edges below that threshold are removed from the
        network."""
        min_weight_threshold = 0.0
        for node, neighbours in tqdm(network.items()):
            if len(neighbours) > k_max:
                no_nodes_to_remove = len(neighbours) - k_max
                weights_list = list(neighbours.values())
                weights_list.sort()
                candidate_min_weight_threshold = weights_list[no_nodes_to_remove - 1]  # -1 is correction for zero indexing
                if candidate_min_weight_threshold > min_weight_threshold:
                    min_weight_threshold = candidate_min_weight_threshold

        return min_weight_threshold


    def prune_network_on_min_weight_threshold(network: dict, min_weight_threshold: float) -> dict:
        """Returns the network after pruning all edges below min_weight_threshold"""
        pruned_network = copy.deepcopy(network)
        for node, neighbours in tqdm(network.items()):
            for neighbour, weight in neighbours.items():
                if weight <= min_weight_threshold:
                    pruned_network[node].pop(neighbour)

        return pruned_network


    # %% colab={"base_uri": "https://localhost:8080/"} id="Rrr3hpFM8V6z" outputId="b77219b1-45e7-4598-b84b-bd582b70dfaa"
    min_norm_degree = find_hard_minimum_threshold_weight(network=norm_degree_diff_network, k_max=60)
    transformed_network = prune_network_on_min_weight_threshold(norm_degree_diff_network, min_norm_degree)

    # %% colab={"base_uri": "https://localhost:8080/", "height": 646} id="CFn3A_jk8aMK" outputId="bbfca3ac-9562-43b0-efc0-9647437c1942"
    # Visualising the degree distribution of the transformed_network
    degree_distribution = []
    for neighbours_dict in transformed_network.values():
        degree_distribution.append(len(neighbours_dict))

    sns.histplot(data=degree_distribution, kde=True, color = 'green')
    plt.title('Degree Distribution of Transformed Network')
    plt.xlabel('Degree')
    # plt.yscale("log")
    plt.ylabel("Count")
    plt.show()

    print(f"Transformed Network Degree Distribution Statistics:\n{pd.Series(degree_distribution).describe()}")


    # %% [markdown] id="fb738c58"
    # #### 5.1.3 Analysing the relation between k_max and giant component size

    # %% id="YJRAJUtQ8vIo"
    def get_network_components(network: dict) -> list:
        """Returns a list of sets where each set is a connected cluster of nodes"""

        def find_node_cluster(cluster_map, node):
            """Returns the head node of a cluster if the node is in a cluster otherwise it returns None"""
            for head_node, node_set in cluster_map.items():
                if node in node_set:
                    return head_node

            return None

        # Isolating the different components
        cluster_map = {}
        for node, neighbours in network.items():
            # 1) Check if current node is in a cluster
            current_node_cluster = find_node_cluster(cluster_map, node)

            # 2) If node is not in a cluster create a new one
            if current_node_cluster is None:
                current_node_cluster = node
                cluster_map[current_node_cluster] = set([node])

            clusters_to_be_merged = set()
            for neighbour in neighbours.keys():
                # 3) If neighbour is part of another cluster mark the cluster for merging
                neighbour_cluster = find_node_cluster(cluster_map, neighbour)
                if (neighbour_cluster is not None) and (neighbour_cluster != current_node_cluster):
                    clusters_to_be_merged.add(neighbour_cluster)
                elif neighbour_cluster != current_node_cluster:
                    # 4) Add neighbouring nodes to the current cluster
                    cluster_map[current_node_cluster].add(neighbour)

            # 5) Merge the clusters that need to be merged
            for cluster in clusters_to_be_merged:
                cluster_content = cluster_map[cluster]
                cluster_map[current_node_cluster] = cluster_map[current_node_cluster].union(cluster_content)
                cluster_map.pop(cluster)

        return list(cluster_map.values())


    def get_giant_component_size(network: dict) -> list:
        components = get_network_components(network)
        return max([len(c) for c in components])


    # %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="8AGaSKs88yiy" outputId="d7d4c8b2-45a7-4d34-d1aa-83a4762441e3"
    # Analysing relation between k_max and size of giant component
    k_max_list = list(range(100, 10, -5))
    giant_component_size = []
    for k_max in k_max_list:
        min_norm_degree = find_hard_minimum_threshold_weight(network=norm_degree_diff_network, k_max=k_max)
        transformed_network = prune_network_on_min_weight_threshold(norm_degree_diff_network, min_norm_degree)
        giant_component_size.append(get_giant_component_size(transformed_network))

    sns.lineplot(x=k_max_list, y=giant_component_size)
    plt.title('k_max vs Size of Giant Component (For Norm Degree Diff - Min Threshold)')
    plt.xlabel('k_max')
    plt.ylabel("Size of Giant Component")
    plt.show()

    # %% colab={"base_uri": "https://localhost:8080/"} id="vkvTJ3AW83TJ" outputId="335c4785-c454-4b52-c753-ec9908803405"
    K_MAX = 35
    min_norm_degree = find_hard_minimum_threshold_weight(network=norm_degree_diff_network, k_max=K_MAX)
    transformed_network = prune_network_on_min_weight_threshold(norm_degree_diff_network, min_norm_degree)

    print(f"K_MAX = {K_MAX}")
    print(f"Size of Giant Component is: {get_giant_component_size(transformed_network)}")
    degree_distribution = [len(v) for v in transformed_network.values()]
    amt_of_zero_degree_nodes = sum([(1 if len(v) == 0 else 0) for v in transformed_network.values()])
    print(f"There are {amt_of_zero_degree_nodes} out of {len(transformed_network)} with a degree of zero.")

    # Saving Transfromed Network
    with open("./transformed_network.json", "w") as f1:
        json.dump(transformed_network, f1, indent=4)


    # %% id="M_cEG3WcH1Gs"

    # Define a function to generate the Graph from the collaboration network JSON file
    def generate_graph(collaboration_network, weight_key='collaborations'):
        G = nx.Graph()
        # Add the links between the authors
        for author, collaborators in collaboration_network.items():
            for collaborator, details in collaborators.items():
                G.add_edge(author, collaborator, weight=details[weight_key])
        # Add the nodes that have no connections
        for author in collaboration_network.keys():
            if collaboration_network[author] == {}:
                G.add_node(author)
        return G


    # %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="mQnquQyn88vu" outputId="889de6ce-5da7-4f63-c43b-4f19635bc4b5"
    # Generate the Graph from the collaboration network
    transformed_network_2 = copy.deepcopy(transformed_network)
    for node, neighbours in transformed_network.items():
        for neighbour, weight in neighbours.items():
            transformed_network_2[node][neighbour] = {'weight': 1}
    print(len(transformed_network_2.keys()))
    G = generate_graph(transformed_network_2, weight_key='weight')

    # Display the number of nodes and edges in the Graph
    print(f'The Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')

    # Display the Graph
    plt.figure(figsize=(15, 15))
    plt.tight_layout()
    pos = nx.spring_layout(G, seed=42, k=0.6)
    nx.draw(G, pos, with_labels=False, node_size=30, edge_color='black', node_color='yellow',
            width=0.5)  # Decrease node_size and set with_labels to False
    plt.title('Transformed Collaboration Network of the Data Scientists')
    # Save the plot as a PNG file in the 'plots' directory
    #plt.savefig(plots_dir + 'transformed_network_graph.png', bbox_inches='tight')
    plt.show()


    # %% [markdown] id="9177dec84b387fa8"
    # #### 5.1.5 Transformed Network vs Original Network - Country, Expertise and Institution Distribution
    # Now, we will analyse the country, expertise and institution distribution of the nodes in the transformed network and compare it with the original network because we want to ensure that the diversity of the nodes in the network is maintained.

    # %% id="zcngESL-9D7Q"
    def get_param_degree(G, data_scientists, param='country'):
        param_degrees = {}
        for node in G.nodes():
            # Fix: Check if the node exists in data_scientists before accessing
            if node in data_scientists['pid'].values:
                p = data_scientists.loc[data_scientists['pid'] == node, param].values[0]
                # Check if the country is not available
                if pd.isnull(p):
                    continue
                if param != 'expertise':
                    # Check if the country is a set
                    if '{' in p:
                        # Convert the string to a set
                        p = ast.literal_eval(p)
                        # Get the first country in the set
                        p = list(p)[0]
                if p in param_degrees:
                    param_degrees[p] += nx.degree(G, node)
                else:
                    param_degrees[p] = nx.degree(G, node)
        # Compute the average node degree for each country
        for p in param_degrees:
            param_degrees[p] = np.mean(param_degrees[p])
        return param_degrees


    # %% id="cU9Ggs5J-Dlw" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="c59868d4-fadb-4671-dbb7-ebe5b1a63936"
    # Import the data of the data scientists
    data_scientists = pd.read_csv('/content/scientists_cleaned.csv')

    # Get the collaboration network data from the JSON file
    collaboration_network = json.load(open('/content/network.json', 'r'))

    # Get the network graph of the original collaboration network
    G_original = generate_graph(collaboration_network)
    # Get the network graph of the transformed collaboration network
    G_transformed = generate_graph(transformed_network_2, weight_key='weight')

    # Get the country node degree sum of the original collaboration network
    country_degree_original = get_param_degree(G_original, data_scientists)
    # Get the country node degree sum of the transformed collaboration network
    country_degree_transformed = get_param_degree(G_transformed, data_scientists)

    # Sort the country node degree sum dictionaries
    country_degree_original = dict(sorted(country_degree_original.items(), key=lambda item: item[1], reverse=True))
    country_degree_transformed = dict(sorted(country_degree_transformed.items(), key=lambda item: item[1], reverse=True))

    # Plot the country node degree sum of the original and transformed collaboration networks side by side
    plt.figure(figsize=(20, 8))

    # Plot the country node degree sum of the original collaboration network
    plt.subplot(1, 2, 1)
    plt.bar(list(country_degree_original.keys()), list(country_degree_original.values()), color='coral')
    plt.title('Country Node Degree Sum of the Original Collaboration Network')
    plt.xlabel('Country')
    plt.ylabel('Node Degree')
    plt.xticks(rotation=90)

    # Plot the country node degree sum of the transformed collaboration network
    plt.subplot(1, 2, 2)
    plt.bar(list(country_degree_transformed.keys()), list(country_degree_transformed.values()), color='mediumseagreen')
    plt.title('Country Node Degree Sum of the Transformed Collaboration Network')
    plt.xlabel('Country')
    plt.ylabel('Node Degree')
    plt.xticks(rotation=90)

    # Save the plot as a PNG file in the 'plots' directory
    #plt.savefig(plots_dir + 'country_node_degree_original_transformed.png', bbox_inches='tight')

    # %% [markdown] id="eb69ddd1e88d548e"
    # Again, as we can see, the expertise node degree sum of the transformed collaboration network is similar to the original collaboration network. This indicates that the diversity of the nodes in the network is maintained after the transformation. Lastly, we will compare the institution distribution of the nodes in the transformed network with the original network.

    # %% id="BHcnDhdX-RIe" colab={"base_uri": "https://localhost:8080/", "height": 829} outputId="98997938-6f55-4c87-e2b1-cb87cb73e81d"
    # Get the institution node degree sum of the original collaboration network
    institution_degree_original = get_param_degree(G_original, data_scientists, param='institution')
    # Get the institution node degree sum of the transformed collaboration network
    institution_degree_transformed = get_param_degree(G_transformed, data_scientists, param='institution')

    # Sort the institution node degree sum dictionaries
    institution_degree_original = dict(sorted(institution_degree_original.items(), key=lambda item: item[1], reverse=True))
    institution_degree_transformed = dict(sorted(institution_degree_transformed.items(), key=lambda item: item[1], reverse=True))

    # Get the top 10 institutions in the original collaboration network
    top_institutions_original = dict(list(institution_degree_original.items())[:10])
    # Get the top 10 institutions in the transformed collaboration network
    top_institutions_transformed = dict(list(institution_degree_transformed.items())[:10])

    # Plot the top institution node degree sum of the original and transformed collaboration networks side by side
    plt.figure(figsize=(20, 8))

    # Plot the institution node degree sum of the original collaboration network
    plt.subplot(1, 2, 1)
    plt.bar(list(top_institutions_original.keys()), list(top_institutions_original.values()), color='coral')
    plt.title('Top Institution Node Degree Sum of the Original Collaboration Network')
    plt.xlabel('Institution')
    plt.ylabel('Node Degree')
    plt.xticks(rotation=90)

    # Plot the institution node degree sum of the transformed collaboration network
    plt.subplot(1, 2, 2)
    plt.bar(list(top_institutions_transformed.keys()), list(top_institutions_transformed.values()), color='mediumseagreen')
    plt.title('Top Institution Node Degree Sum of the Transformed Collaboration Network')
    plt.xlabel('Institution')
    plt.ylabel('Node Degree')
    plt.xticks(rotation=90)






if __name__ == '__main__':
    main()

# Save the plot as a PNG file in the 'plots' directory
#plt.savefig(plots_dir + 'institution_node_degree_original_transformed.png', bbox_inches='tight')

# %% [markdown] id="c3a165365aec2bd2"
# As we can see, the institution node degree sum of the transformed collaboration network is similar to the original collaboration network. We have some changes in the positions, but they are minor, overall the top 10 institutions with the highest node degree sum are very similar. This indicates that the diversity of the nodes in the network is maintained after the transformation. This concludes the transformation of the collaboration network of the data scientists. In the next section, we will discuss an alternate reduction model we used.

