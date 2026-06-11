import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from src.data_loader import fetch_tbill_rates
from src.stats import run_all_tests

def calculate_metrics(returns, benchmark_returns=None, rf=0.0):
    # Ensure returns is a clean 1D float Series
    returns = pd.Series(returns.values.flatten(), index=returns.index, dtype=float)
    
    ann_ret = (1 + returns).prod() ** (12 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    
    # Handle rf as either a scalar or a Series of monthly rates
    if isinstance(rf, (pd.Series, np.ndarray)):
        rf_vals = pd.Series(rf.values.flatten(), dtype=float)
        ann_rf = (1 + rf_vals).prod() ** (12 / len(rf_vals)) - 1
    else:
        ann_rf = float(rf)
    
    sharpe = (ann_ret - ann_rf) / ann_vol if ann_vol != 0 else 0
    
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    info_ratio = "N/A"
    if benchmark_returns is not None:
        active_return = returns - benchmark_returns
        tracking_error = active_return.std() * np.sqrt(12)
        info_ratio = (ann_ret - ((1 + benchmark_returns).prod() ** (12 / len(benchmark_returns)) - 1)) / tracking_error if tracking_error != 0 else 0
        info_ratio = f"{info_ratio:.2f}"
        
    return ann_ret, ann_vol, sharpe, max_dd, info_ratio

def run_momentum_baseline(returns, lookback=60, top_n=2):
    print(f"Running pure Price Momentum baseline (Holding Top {top_n})...")
    monthly_returns = (1 + returns).resample('ME').prod() - 1
    
    # Calculate rolling cumulative returns for the lookback window
    rolling_momentum = returns.rolling(lookback).sum()
    monthly_momentum = rolling_momentum.resample('ME').last()
    
    target_holdings = monthly_momentum.apply(
        lambda x: x.nlargest(top_n).index.tolist(), axis=1
    ).shift(1).dropna()
    
    strat_returns = []
    dates = []
    for date, tickers in target_holdings.items():
        if date in monthly_returns.index:
            strat_returns.append(monthly_returns.loc[date, tickers].mean())
            dates.append(date)
            
    return pd.DataFrame({'Momentum': strat_returns}, index=dates)

def run_backtest(returns, centrality, top_n=2):
    print(f"\nRunning Graph Centrality backtest (Holding Top {top_n})...")
    monthly_centrality = centrality.resample('ME').last()
    
    # .shift(1) prevents look-ahead bias
    target_holdings = monthly_centrality.apply(
        lambda x: x.nlargest(top_n).index.tolist(), axis=1
    ).shift(1).dropna()
    
    monthly_returns = (1 + returns).resample('ME').prod() - 1
    
    strategy_returns_raw = []
    strategy_returns_switch = []
    dates = []
    
    defensive_sectors = ['XLU', 'XLP'] 

    print("Fetching T-bill rate data...")
    tbill_monthly = fetch_tbill_rates(
        start_date=str(returns.index[0].date()),
        end_date=str(returns.index[-1].date())
    )
    
    for date, tickers in target_holdings.items():
        if date not in monthly_returns.index:
            continue
            
        mean_return = monthly_returns.loc[date, tickers].mean()
        
        # Guard: skip if mean_return is not a scalar (e.g. NaN or Series)
        if not isinstance(mean_return, (float, int, np.floating)):
            continue

        # Iteration 2: Raw MST Centrality (always invested)
        strategy_returns_raw.append(float(mean_return))

        # Iteration 3: Topological Regime Switch (cash at T-bill rate)
        if any(sector in tickers for sector in defensive_sectors):
            if date in tbill_monthly.index:
                cash_return = float(tbill_monthly.loc[date])
            else:
                nearest = tbill_monthly.index.asof(date)
                cash_return = float(tbill_monthly.loc[nearest]) if nearest is not pd.NaT else 0.0
            strategy_returns_switch.append(cash_return)
        else:
            strategy_returns_switch.append(float(mean_return))

        dates.append(date)
            
    strategy_df = pd.DataFrame({
        'Graph Raw': strategy_returns_raw,
        'Graph Switch': strategy_returns_switch
    }, index=dates)
    
    # Get Momentum Baseline
    momentum_df = run_momentum_baseline(returns, lookback=60, top_n=top_n)
    
    print("Fetching SPY benchmark data...")
    spy_data = yf.download("SPY", start=dates[0], end=dates[-1] + pd.Timedelta(days=31), progress=False)
    spy_prices = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']
    
    # Convert SPY to log returns, then resample to monthly
    spy_daily_returns = np.log(spy_prices / spy_prices.shift(1)).dropna()
    spy_monthly_returns = (1 + spy_daily_returns).resample('ME').prod() - 1
    spy_monthly_returns.name = 'SPY'
    
    # Combine everything
    comparison = pd.concat([strategy_df['Graph Raw'], strategy_df['Graph Switch'], momentum_df['Momentum'], spy_monthly_returns], axis=1).dropna()
    
    # Align T-bill to comparison index
    tbill_aligned = tbill_monthly.reindex(comparison.index).ffill().fillna(0.0)
    tbill_aligned = pd.Series(tbill_aligned.values.flatten(), index=comparison.index)

    # Rf=0 for all strategies except regime switch (for paper's zero-benchmark comparability)
    raw_metrics    = calculate_metrics(comparison['Graph Raw'],    comparison['SPY'], rf=0.0)
    switch_metrics = calculate_metrics(comparison['Graph Switch'], comparison['SPY'], rf=tbill_aligned)
    switch_metrics_zero = calculate_metrics(comparison['Graph Switch'], comparison['SPY'], rf=0.0)
    mom_metrics    = calculate_metrics(comparison['Momentum'],     comparison['SPY'], rf=0.0)
    spy_metrics    = calculate_metrics(comparison['SPY'],          rf=0.0)
    
    print("\n--- PERFORMANCE METRICS (2015-2023) ---")
    print(f"{'Metric':<18} | {'MST Raw (Iter 2)':<16} | {'Regime Rf=0':<13} | {'Regime Rf=Tbill':<15} | {'Momentum':<12} | {'SPY':<10}")
    print("-" * 100)
    print(f"{'Ann. Return':<18} | {f'{raw_metrics[0]:.2%}':<16} | {f'{switch_metrics_zero[0]:.2%}':<13} | {f'{switch_metrics[0]:.2%}':<15} | {f'{mom_metrics[0]:.2%}':<12} | {spy_metrics[0]:.2%}")
    print(f"{'Ann. Volatility':<18} | {f'{raw_metrics[1]:.2%}':<16} | {f'{switch_metrics_zero[1]:.2%}':<13} | {f'{switch_metrics[1]:.2%}':<15} | {f'{mom_metrics[1]:.2%}':<12} | {spy_metrics[1]:.2%}")
    print(f"{'Sharpe Ratio':<18} | {f'{raw_metrics[2]:.2f}':<16} | {f'{switch_metrics_zero[2]:.2f}':<13} | {f'{switch_metrics[2]:.2f}':<15} | {f'{mom_metrics[2]:.2f}':<12} | {spy_metrics[2]:.2f}")
    print(f"{'Max Drawdown':<18} | {f'{raw_metrics[3]:.2%}':<16} | {f'{switch_metrics_zero[3]:.2%}':<13} | {f'{switch_metrics[3]:.2%}':<15} | {f'{mom_metrics[3]:.2%}':<12} | {spy_metrics[3]:.2%}")
    print(f"{'Info Ratio':<18} | {f'{raw_metrics[4]}':<16} | {f'{switch_metrics_zero[4]}':<13} | {f'{switch_metrics[4]}':<15} | {f'{mom_metrics[4]}':<12} | N/A")
    print("\nNote: 'Regime Rf=0' uses zero risk-free rate (cross-strategy comparability).")
    print("      'Regime Rf=Tbill' uses actual 3-month T-bill rate during cash periods.")
    run_all_tests(comparison, tbill_aligned=tbill_aligned)
    # Plotting
    cumulative_returns = (1 + comparison).cumprod()
    plt.figure(figsize=(10, 6))
    
    plt.plot(cumulative_returns.index, cumulative_returns['Graph Switch'], label=f'Graph Strategy (Top {top_n} + Regime Filter)', color='darkblue', linewidth=2)
    plt.plot(cumulative_returns.index, cumulative_returns['Graph Raw'], label=f'Graph Strategy Raw (Top {top_n})', color='orange', alpha=0.7, linewidth=1.5)
    plt.plot(cumulative_returns.index, cumulative_returns['Momentum'], label=f'Price Momentum (Top {top_n})', color='darkred', linestyle='--', linewidth=1.5)
    plt.plot(cumulative_returns.index, cumulative_returns['SPY'], label='SPY Benchmark', color='black', alpha=0.5, linewidth=1.5)
    
    plt.title(f'Cumulative Returns: Topology Regime Switch vs S&P 500')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=300)
    print("\nSaved final performance chart to 'equity_curve.png'")
    return comparison