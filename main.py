import pandas as pd
from src.data_loader import fetch_sector_returns
from src.graph_builder import calculate_rolling_centrality
from src.backtester import run_backtest

def optimize_parameters():
    print("Fetching historical data once for all tests...")
    raw_returns = fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31")
    
    # We will test a fast, medium, and slow moving network
    windows_to_test = [20, 60, 120]
    
    for w in windows_to_test:
        print(f"\n{'='*50}")
        print(f"TESTING LOOKBACK WINDOW: {w} DAYS")
        print(f"{'='*50}")
        
        centrality_df = calculate_rolling_centrality(raw_returns, window=w)
        # Note: This will overwrite the equity_curve.png for each run, 
        # but we just want to read the terminal metrics for now.
        run_backtest(raw_returns, centrality_df, top_n=2)

if __name__ == "__main__":
    optimize_parameters()
