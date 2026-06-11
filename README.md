# Sector Rotation via Graph Centrality
 
> **Research paper implementation** - *Limitations of Topological Signals in Mega-Cap Momentum Markets: An Iterative Network Analysis of Sector Rotation*
>
> Prof. Neena Goveas · Tanishq Sahu · Vishwam Tiwari
> BITS Pilani, K.K. Birla Goa Campus
 
---
 
## Overview
 
This repository contains the full research codebase for our paper submitted to **Physica A: Statistical Mechanics and its Applications**. The project models the U.S. equity market as a **dynamic, time-varying correlation network** of the 11 GICS sector ETFs, and explores whether graph-theoretic signals - specifically **Eigenvector Centrality** computed over a rolling Maximum Spanning Tree (MST) - can predict sector rotation before it is reflected in price momentum.
 
The core hypothesis: **topology precedes price**. Before a sector breaks out, it first becomes structurally critical in the market's underlying correlation web.
 
---
 
## Key Findings
 
| Strategy | Ann. Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|
| Dense Graph Centrality (Iter. 1) | 7.00% | 0.32 | -25.39% |
| MST Centrality (Iter. 2) | 10.35% | 0.45 | -30.59% |
| MST + Regime Switch (Iter. 3) | 10.31% | 0.45 | -30.59% |
| Centrality Velocity ΔC (Iter. 4) | 6.44% | 0.37 | -29.38% |
| Price Momentum Baseline | 11.81% | 0.57 | -21.08% |
| SPY Benchmark | 9.16% | 0.47 | -25.56% |
 
**Bottom line:** MST-filtered topology is an effective *risk oracle* - it reliably detects systemic stress and flight-to-safety rotations - but underperforms pure price momentum as a standalone alpha signal in mega-cap-dominated, liquidity-driven regimes (2019–2023).
 
---
 
## Repository Structure
 
```
sector-rotation-graph-centrality/
│
├── src/
│   ├── data_loader.py          # Fetches 11 sector ETF returns via yfinance
│   ├── graph_builder.py        # Rolling correlation → adjacency matrix → MST → EC
│   ├── backtester.py           # Strategy engine + SPY/momentum comparison
│   └── visualizer.py           # Heatmap, network snapshots, equity curves
│
├── outputs/
│   ├── centrality_heatmap.png  # Figure 2 — EC over time (2015–2023)
│   ├── network_pre_covid.png   # Figure 3a — Jan 2020
│   ├── network_covid_crash.png # Figure 3b — Mar 2020
│   ├── network_rate_hikes.png  # Figure 3c — Jan 2022
│   ├── equity_curve.png        # Figure 4/5/6 — cumulative returns
│   └── metrics_table.csv       # Table 1 — consolidated performance metrics
│
├── main.py                     # Entry point — runs full pipeline + window optimization
├── requirements.txt
└── README.md
```
 
---
 
## Methodology
 
The strategy is developed iteratively across four phases:
 
### Iteration 1 - Dense Graph Eigenvector Centrality
Build a fully connected weighted graph from the 60-day rolling Pearson correlation matrix. Compute Eigenvector Centrality and go long the top-2 sectors monthly.
 
**Result:** Fails during volatility spikes - all correlations converge to 1.0, turning the network into an indistinguishable "hairball."
 
### Iteration 2 - Maximum Spanning Tree (MST) Filter
Apply Kruskal's algorithm to extract the topological backbone of the market (N−1 edges from N nodes). Compute EC strictly on the sparse MST.
 
```
T_t = argmax  Σ  A_i,j(t)      over all spanning trees T
          (i,j)∈E(T)
```
 
**Result:** Much cleaner signal. Clearly captures XLE's structural rise before the 2022 inflation shock. Still suffers drawdowns when the whole network falls together.
 
### Iteration 3 - Topological Regime Switch
Introduce a binary switch: if defensive sectors (XLU or XLP) rank in the top-2 by EC, interpret this as an institutional "flight to safety" and liquidate to 100% cash.
 
```
S_t = 0   if Rank(x_XLU) ≤ 2  or  Rank(x_XLP) ≤ 2
S_t = 1   otherwise
```
 
**Result:** Successfully dodges localized crashes (visible as flat plateaus in the equity curve). However, causes severe "cash drag" during the 2020–2021 liquidity-driven bull run.
 
### Iteration 4 - Centrality Velocity (ΔC)
Shift from the *level* of centrality to its *first derivative* - buy sectors gaining centrality fastest to front-run institutional rotation.
 
```
Δx_v,t = x_v,t − x_v,t−1
```
 
**Result:** Catastrophic failure. Differencing an already volatile metric amplifies noise. During the March 2020 crisis, the algorithm repeatedly buys false breakouts.
 
---
 
## Data
 
- **Universe:** 11 GICS Sector SPDR ETFs - `XLK, XLV, XLF, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC`
- **Benchmark:** `SPY` (S&P 500 ETF)
- **Period:** January 1, 2015 - December 31, 2023
- **Source:** Yahoo Finance via `yfinance`
- **Returns:** Continuous log returns - `r_i,t = ln(P_i,t / P_i,t−1)`
- **Rolling window:** 60 trading days (~3 months)
---
 
## Installation
 
```bash
git clone https://github.com/your-username/sector-rotation-graph-centrality.git
cd sector-rotation-graph-centrality
pip install -r requirements.txt
```
 
**requirements.txt**
```
yfinance
pandas
numpy
networkx
matplotlib
seaborn
scipy
```
 
---
 
## Usage
 
### Run the full pipeline
 
```bash
python main.py
```
 
This runs all 4 iterations across lookback windows of 20, 60, and 120 days and prints a performance metrics table for each.
 
### Run a single backtest
 
```python
from src.data_loader import fetch_sector_returns
from src.graph_builder import calculate_rolling_centrality
from src.backtester import run_backtest
 
returns = fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31")
centrality = calculate_rolling_centrality(returns, window=60)
run_backtest(returns, centrality, top_n=2)
```
 
### Generate all figures
 
```python
from src.visualizer import plot_centrality_heatmap, plot_network_snapshot
 
plot_centrality_heatmap(centrality)
 
plot_network_snapshot(returns, centrality, "2020-01-15", "outputs/network_pre_covid.png")
plot_network_snapshot(returns, centrality, "2020-03-20", "outputs/network_covid_crash.png")
plot_network_snapshot(returns, centrality, "2022-01-14", "outputs/network_rate_hikes.png")
```
 
---
 
## Core Implementation
 
### Graph construction (`graph_builder.py`)
 
```python
def get_mst_centrality(returns_window):
    corr_matrix = returns_window.corr()
 
    # Affine transform: correlation [-1,1] → adjacency [0,1]
    # Required for Perron-Frobenius theorem (unique positive eigenvector)
    adj_matrix = (1 + corr_matrix) / 2
    for col in adj_matrix.columns:
        adj_matrix.loc[col, col] = 0  # remove self-loops
 
    G_full = nx.from_pandas_adjacency(adj_matrix)
 
    # Extract topological backbone via Maximum Spanning Tree
    G_mst = nx.maximum_spanning_tree(G_full, weight='weight')
 
    centrality = nx.eigenvector_centrality(G_mst, max_iter=1000, weight='weight')
    return centrality
```
 
### Regime switch (`backtester.py`)
 
```python
defensive_sectors = ['XLU', 'XLP']
 
for date, tickers in target_holdings.items():
    if date in monthly_returns.index:
        if any(sector in tickers for sector in defensive_sectors):
            strategy_returns.append(0.0)  # liquidate to cash
        else:
            strategy_returns.append(monthly_returns.loc[date, tickers].mean())
```
 
### Centrality velocity (`backtester.py`)
 
```python
delta_centrality = monthly_centrality.diff()
 
target_holdings = delta_centrality.apply(
    lambda x: x.nlargest(top_n).index.tolist(), axis=1
).shift(1).dropna()
```
 
---
 
## Results & Figures
 
**Figure 2 - Centrality heatmap (2015–2023)**
Each row is a sector, each column a date, color encodes Eigenvector Centrality. Notice the XLE dominance band during the 2022 inflation cycle and the defensive cluster spike in March 2020.
 
**Figure 3 - Network topology at 3 key dates**
Node size = Eigenvector Centrality. The network tightens dramatically during the COVID crash (all sectors correlated), then elongates in 2022 as energy/commodity sectors decouple from tech.
 
**Figures 4-6 - Equity curves**
Each iteration's cumulative return vs. SPY and the 12-month momentum baseline over the full 2015–2023 backtest period.
 
---
 
## Discussion: Why Topology Lost to Momentum
 
Three structural reasons the topological signals underperformed:
 
1. **The Magnificent 7 effect.** The S&P 500 is cap-weighted. XLK's concentration meant pure momentum algorithms implicitly "hugged" the dominant constituent. Our topology correctly identified overextension - and rotated away - but the index kept going up regardless.
2. **Cash drag in liquidity-driven regimes.** The Regime Switch worked perfectly as a risk manager but sat in 0% cash through the entire 2020–2021 fiscal-stimulus bull run. Being right about risk doesn't help when liquidity overrides fundamentals.
3. **Noise amplification in kinematic signals.** Differencing correlation-derived centrality (ΔC) amplifies the instability of the underlying matrices during crises. The velocity model mistakes volatility spikes for capital rotation.
**Conclusion:** MST-filtered Eigenvector Centrality should be used as a *position-sizing or risk management overlay*, not a standalone long-only signal - especially in cap-weighted, momentum-dominated markets.
 
---
 
## Paper
 
> **Limitations of Topological Signals in Mega-Cap Momentum Markets: An Iterative Network Analysis of Sector Rotation**
>
> Prof. Neena Goveas, Tanishq Sahu, Vishwam Tiwari
> Department of Computer Science and Information Systems
> BITS Pilani, K.K. Birla Goa Campus, Goa, India
>
> *Submitted to Physica A: Statistical Mechanics and its Applications*
 
Key references this work builds on:
- Mantegna (1999) - Hierarchical structure in financial markets
- Bonanno et al. (2003) - MST in equity markets
- Onnela et al. (2003) - Dynamics of market correlations
- Bonacich (1987) - Power and centrality measures
- Jegadeesh & Titman (1993) - Cross-sectional momentum
---
 
## Limitations & Future Work
 
- All backtests are **gross of transaction costs**. Estimated round-trip costs of ~10–15 bps per monthly rebalance would modestly reduce all strategy returns, disproportionately penalizing high-turnover iterations (Dense Graph, Velocity).
- The equal-weighted sector universe does not account for **intra-sector concentration** (e.g., NVDA's weight within XLK).
- Future directions: apply topological signals to **equal-weighted indices**, test **non-linear smoothing** (EMA, Kalman filter) on centrality before differencing, and explore **partial liquidation** (rather than binary cash switch) during defensive regime detection.

---
 
*BITS Pilani, K.K. Birla Goa Campus · Department of Computer Science and Information Systems*
