import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
from src.data_loader import fetch_sector_returns

def run_strategy_on_window(returns_df, window_size, start, end, strategy='mst'):    
    """    
    Run the MST/Regime strategy on a slice of returns data using position-based trading lookbacks.
    Returns annualised Sharpe (Rf=0) over the test period.    
    """    
    monthly_returns = []        
    # Generate the start dates for each month in the test year
    test_months = pd.date_range(start=start, end=end, freq='MS')        
    
    for month in test_months:        
        # 1. Position-based Lookback: Filter data up to current month, then take exact trading days
        prior_data = returns_df.loc[:month]        
        window_data = prior_data.tail(window_size)        
        
        # Safety catch for beginning of historical dataset
        if len(window_data) < window_size:            
            continue                
            
        # Build correlation matrix and MST
        corr = window_data.corr()        
        adj = (1 + corr) / 2        
        
        # Create a clean, mutable copy of the array to avoid read-only view bugs
        adj_mat = adj.to_numpy().copy()
        np.fill_diagonal(adj_mat, 0)                
        
        # Re-wrap into a DataFrame to preserve sector string names ('XLK', 'XLU', etc.)
        adj = pd.DataFrame(adj_mat, index=adj.index, columns=adj.columns)
        
        G = nx.from_pandas_adjacency(adj)        
        mst = nx.maximum_spanning_tree(G)        
        centrality = nx.eigenvector_centrality_numpy(mst, weight='weight')                
        ranked = sorted(centrality, key=centrality.get, reverse=True)                
        
        # 3. Handle defensive regime switch filter
        if strategy == 'regime_switch':            
            defensive = {'XLU', 'XLP'}            
            if ranked[0] in defensive or ranked[1] in defensive:                
                monthly_returns.append(0.0) # 100% allocation to cash (0% return)
                continue                
        
        top2 = ranked[:2]        
        
        # 4. Compute clean forward monthly returns (isolate to target month's trading days)
        fwd = returns_df[((returns_df.index.year == month.year) & (returns_df.index.month == month.month))][top2]
        
        if fwd.empty:
            continue
            
        # Sum daily log returns to get total monthly return per sector, then take portfolio average
        total_monthly_return = fwd.sum(axis=0).mean()
        monthly_returns.append(total_monthly_return)        
        
    sr = pd.Series(monthly_returns)    
    if sr.empty or sr.std() == 0:        
        return np.nan    
    
    # Calculate annualized Sharpe using standard monthly scaling
    annualised_sharpe = (sr.mean() / sr.std()) * np.sqrt(12)    
    return round(annualised_sharpe, 3)

print("Fetching data for Walk-Forward Validation...")
raw_returns = fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31")
daily_log_returns = np.log(1 + raw_returns) # Convert to log returns for additive rolling windows

# --- Walk-forward folds ---
folds = [    
    ('2019-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),    
    ('2019-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),    
    ('2019-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
]

windows = [30, 60, 90, 120]
results = []

print("Running Walk-Forward cross-validation folds...")
for (train_start, train_end, test_start, test_end) in folds:    
    for w in windows:        
        for strat in ['mst', 'regime_switch']:            
            sharpe_oos = run_strategy_on_window(                
                daily_log_returns,                  
                window_size=w,                
                start=test_start,                
                end=test_end,                
                strategy=strat            
            )            
            results.append({                
                'Fold': f"{test_start[:4]}",                
                'Window': w,                
                'Strategy': strat,                
                'OOS Sharpe': sharpe_oos            
            })

wf_df = pd.DataFrame(results)
print("\n=== OUT-OF-SAMPLE WALK-FORWARD SHARPE RATIO RESULTS ===")
print(wf_df.pivot_table(index=['Strategy','Window'], columns='Fold', values='OOS Sharpe'))