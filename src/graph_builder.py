import pandas as pd
import numpy as np
import networkx as nx

def get_centrality_for_window(returns_window):
    corr_matrix = returns_window.corr()
    
    # Adjacency mapping: [-1, 1] -> [0, 1]
    adj_matrix = (1 + corr_matrix) / 2

    # Remove self-loops (diagonal = 0)
    for col in adj_matrix.columns:
        adj_matrix.loc[col, col] = 0

    # 1. Build the full dense graph
    G_full = nx.from_pandas_adjacency(adj_matrix)
    
    # 2. Extract the Topological Backbone (Maximum Spanning Tree)
    # We use 'maximum' because we want to keep the strongest correlation weights
    G_mst = nx.maximum_spanning_tree(G_full, weight='weight')

    # 3. Calculate centrality strictly on the clean backbone
    centrality = nx.eigenvector_centrality(G_mst, max_iter=1000, weight='weight')

    return centrality

def calculate_rolling_centrality(returns, window=60):
    print(f"Calculating rolling MST eigenvector centrality (window={window} days)...")
    centrality_scores = []
    dates = []

    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        current_date = returns.index[i]
        scores = get_centrality_for_window(window_data)
        centrality_scores.append(scores)
        dates.append(current_date)
        
    centrality_df = pd.DataFrame(centrality_scores, index=dates)
    print("MST Centrality calculation complete!")
    return centrality_df

if __name__ == "__main__":
    from data_loader import fetch_sector_returns
    
    test_returns = fetch_sector_returns(start_date="2022-01-01", end_date="2023-12-31")
    centrality_df = calculate_rolling_centrality(test_returns, window=60)

    print("\nFirst 5 days of MST Centrality Scores:")
    print(centrality_df.head())
