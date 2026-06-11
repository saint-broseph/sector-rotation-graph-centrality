# Limitations of Topological Signals in Mega-Cap Momentum Markets
### An Iterative Network Analysis of Sector Rotation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red.svg)](#citation)
[![BITS Pilani](https://img.shields.io/badge/Institution-BITS%20Pilani%20Goa-orange.svg)](https://www.bits-pilani.ac.in/goa/)

**Authors:** Tanishq Sahu · Vishwam Tiwari · Neena Goveas (Supervisor)  
**Institution:** Department of Computer Science and Information Systems, BITS Pilani, K.K. Birla Goa Campus

---

## Table of Contents

- [Overview](#overview)
- [Core Hypothesis and Finding](#core-hypothesis-and-finding)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [Data Pipeline](#data-pipeline)
  - [Dynamic Network Construction](#dynamic-network-construction)
  - [The Four Iterations](#the-four-iterations)
  - [Walk-Forward Validation](#walk-forward-validation)
  - [Statistical Hypothesis Testing](#statistical-hypothesis-testing)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)
- [Module Reference](#module-reference)
- [Key Findings for Practitioners](#key-findings-for-practitioners)
- [Limitations](#limitations)
- [Citation](#citation)
- [References](#references)

---

## Overview

This repository contains the full research implementation for our paper studying whether **graph-theoretic network topology** can generate predictive signals for sector rotation that outperform traditional price momentum strategies.

We model the S&P 500 as a **dynamic, time-varying correlation network** of the 11 GICS sector ETFs, applying Maximum Spanning Tree (MST) filtering and Eigenvector Centrality to detect structural capital rotation before it is reflected in price. The core question: *does topology precede price?*

The study spans a **9-year backtest (2015–2023)** — deliberately chosen to encompass a period of historically low volatility (2015–2017), the March 2020 COVID-19 liquidity shock, the 2022 inflationary rate-hike cycle, and the contemporary era of extreme mega-cap technology dominance ("Magnificent 7").

Rather than presenting a single optimised backtest, we document the **chronological, iterative evolution** of four distinct topological strategies, honestly analysing why each successive approach failed to consistently outperform a simple 12-month price momentum baseline.

---

## Core Hypothesis and Finding

**Hypothesis:** Before a sector experiences a momentum-driven breakout, it first becomes structurally critical to the market's underlying correlation network. Topology precedes price.

**Finding:** MST-filtered graph topology *does* effectively identify systemic market stress and anticipatory capital flight to defensive sectors. However, it **significantly underperforms** a pure 12-month price momentum baseline in the post-2019 macroeconomic regime.

The primary driver of this underperformance is the extreme market-capitalisation concentration of the "Magnificent 7" mega-cap technology equities. In such heavily cap-weighted environments, pure momentum strategies mathematically override structural macro-indicators by continuously tracking the dominant constituents. Topological signals act as *overly conservative leading indicators* — brilliant at predicting the storm, but unequipped to outrun the momentum of the market's largest components.

---

## Repository Structure

```
sector-rotation-via-graph-centrality/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # ETF price fetching + T-bill rate pipeline
│   ├── graph_builder.py      # Rolling MST construction + eigenvector centrality
│   ├── backtester.py         # All four strategy iterations + performance metrics
│   ├── stats.py              # Hypothesis testing (JK, bootstrap, HAC t-test)
│   ├── walk_forward.py       # Expanding-window walk-forward validation
│   └── visualizer.py         # All figure generation (publication-ready)
│
├── main.py                   # Entry point — runs full pipeline
│
├── figures/                  # Generated publication figures (PDF + PNG)
│   ├── figure_1_dense_graph.{pdf,png}
│   ├── figure_2_centrality_heatmap.{pdf,png}
│   ├── figure_3_network_snapshots.{pdf,png}
│   ├── figure_4_mst_strategy.{pdf,png}
│   ├── figure_5_regime_switch.{pdf,png}
│   └── figure_6_topological_velocity.{pdf,png}
│
├── requirements.txt
└── README.md
```

---

## Methodology

### Data Pipeline

**Universe:** 11 GICS Sector SPDR ETFs — XLK, XLV, XLF, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC  
**Benchmark:** SPY (S&P 500 ETF)  
**Period:** January 1, 2015 – December 31, 2023  
**Source:** Yahoo Finance (`yfinance`) for ETF prices; FRED (`^IRX`) for 3-month T-bill rates  

**Dynamic vertex scaling:** XLRE (Real Estate) launched October 2015; XLC (Communication Services) launched June 2018. The rolling correlation network expands from V=9 to V=11 as these assets become live. Pre-inception rows/columns are omitted from the 60-day window calculation, preventing data-snooping.

**Returns:** Daily continuous log-returns:

$$r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$

### Dynamic Network Construction

The market state at time *t* is formalised as an undirected, weighted graph $G_t = (V, E_t, W_t)$.

**Correlation:** Pearson correlation over a trailing 60-day rolling window:

$$\rho_{i,j}(t) = \frac{\sum_{\tau=t-w}^{t}(r_{i,\tau} - \bar{r}_i)(r_{j,\tau} - \bar{r}_j)}{\sqrt{\sum(r_{i,\tau}-\bar{r}_i)^2 \sum(r_{j,\tau}-\bar{r}_j)^2}}$$

**Affine transformation** maps correlations from [−1, 1] to strictly positive adjacency weights [0, 1] (required by the Perron-Frobenius theorem for unique eigenvector solutions):

$$A_{i,j}(t) = \frac{1 + \rho_{i,j}(t)}{2}$$

**MST extraction** via Kruskal's algorithm:

$$T_t = \arg\max_{T \in \mathcal{T}} \sum_{(i,j) \in E(T)} A_{i,j}(t)$$

**Eigenvector Centrality** on the MST adjacency matrix:

$$x_v(t) = \frac{1}{\lambda} \sum_{u \in V} A_{v,u}(t) \cdot x_u(t)$$

### The Four Iterations

#### Iteration 1 — Dense Graph Centrality
Applies Eigenvector Centrality directly to the full N×N adjacency matrix. **Failure mode:** During market shocks, all pairwise correlations converge toward 1.0 (the "hairball" problem), making all centrality scores statistically indistinguishable. The algorithm effectively picks sectors at random under stress.

**Result:** Sharpe 0.32, Ann. Return 7.00%, Max Drawdown −25.39%

#### Iteration 2 — MST-Filtered Centrality
Filters the dense graph through an MST (exactly N−1 = 10 edges), stripping spurious correlations and retaining only the most critical capital flow pathways. Eigenvector Centrality is then computed on this sparse backbone. **Failure mode:** Holding the "centre of a crashing market" offers no absolute protection — relative topological superiority cannot prevent drawdowns when the entire network depreciates.

**Result:** Sharpe 0.56, Ann. Return 12.55%, Max Drawdown −30.59%

#### Iteration 3 — Topological Regime Switch
Adds a binary cash-trigger: when Utilities (XLU) or Consumer Staples (XLP) migrate to the MST core (top-2 centrality rank), it signals institutional "flight to safety" and the portfolio liquidates 100% to cash at the prevailing 3-month T-bill rate.

$$S_t = \begin{cases} 0 & \text{if } \text{Rank}(x_{\text{XLU}}) \leq 2 \text{ or } \text{Rank}(x_{\text{XLP}}) \leq 2 \\ 1 & \text{otherwise} \end{cases}$$

**Failure mode:** In the post-COVID ZIRP/liquidity-injection environment (2020–2021), structural stress signals fired while the market continued rising on fiscal stimulus, causing severe "cash drag." Maximum drawdowns were also identical to Iteration 2 (−30.59%) because the March 2020 crash was too rapid for the monthly-rebalancing centrality tracker to anticipate.

**Result (Rf=0):** Sharpe 0.56, Ann. Return 12.51%, Max Drawdown −30.59%  
**Result (Rf=T-bill):** Sharpe 0.47 (applying actual 3-month T-bill rate to cash periods)

#### Iteration 4 — Centrality Velocity (ΔC)
Attempts to front-run the signal by trading the *first derivative* of centrality — buying sectors gaining centrality fastest:

$$\Delta x_{v,t} = x_{v,t} - x_{v,t-1}$$

**Failure mode:** Differencing an already-volatile metric amplifies statistical noise catastrophically. During the March 2020 liquidity crisis, the algorithm mistook random correlation spikes for structural rotation, repeatedly entering false breakouts into a falling market.

**Result:** Sharpe 0.37, Ann. Return 6.44%, Max Drawdown −29.38%

### Walk-Forward Validation

To eliminate in-sample optimisation bias in the 60-day window selection, we deploy an expanding-window walk-forward validation across three annual held-out folds (validation begins January 2019 to ensure full 11-ETF universe alignment):

| Fold | Training Window | OOS Test Year |
|------|----------------|---------------|
| 1 | Jan 2019 – Dec 2020 | 2021 |
| 2 | Jan 2019 – Dec 2021 | 2022 |
| 3 | Jan 2019 – Dec 2022 | 2023 |

Parameter grid: `w ∈ {30, 60, 90, 120}` trading days.

**Key finding:** No single window length dominates across all three regimes. In high-liquidity expansions (2021), shorter windows (w=30) best capture fast-moving capital flows; in macro-stressed environments (2022), longer windows (w=90) provide superior noise insulation. The baseline w=60 provides stable mid-range performance across all folds. **Crucially, the qualitative ranking — topological strategies underperforming price momentum — holds in every single fold**, confirming the mega-cap effect is not an artefact of static parameter selection.

### Statistical Hypothesis Testing

Three complementary tests are run in `src/stats.py` comparing each topological strategy against the price momentum baseline (60-day window, T=108 monthly observations):

**1. Jobson-Korkie Sharpe Ratio Equality Test** (Memmel 2003 correction):

| Strategy | SR (strategy) | SR (momentum) | Z-stat | p-value |
|---|---|---|---|---|
| MST Centrality (Iter. 2) | 0.641 | 0.755 | −0.457 | 0.648 |
| Regime Switch (Iter. 3) | 0.640 | 0.755 | −0.459 | 0.646 |

**2. Block Bootstrap 95% Confidence Intervals** (Politis & Romano 1994, B=10,000, block length=4):

| Strategy | Sharpe | CI Lower | CI Upper | Std. Error |
|---|---|---|---|---|
| MST Centrality (Iter. 2) | 0.641 | −0.207 | 1.555 | 0.453 |
| Regime Switch (Iter. 3) | 0.640 | −0.212 | 1.551 | 0.451 |
| Price Momentum (Baseline) | 0.755 | 0.009 | 1.603 | 0.407 |
| SPY Benchmark | 0.636 | −0.075 | 1.512 | 0.405 |

**3. HAC-Corrected t-test on Return Differences** (Newey-West 1987, lag=3):

| Strategy vs. Momentum | Ann. Diff. | HAC SE | t-stat | p-value |
|---|---|---|---|---|
| MST Centrality (Iter. 2) | −0.83% | 1.70% | −0.141 | 0.888 |
| Regime Switch (Iter. 3) | −0.88% | 1.72% | −0.147 | 0.883 |

**Interpretation:** The underperformance is *economically pronounced* (~1.5% annualised return gap, Sharpe 0.64 vs 0.76) but *statistically indeterminate* — no test rejects H₀ at p<0.05. This is the expected result for T=108 monthly observations, where bootstrap standard errors on Sharpe ratios are approximately ±0.45. The conclusion emphasises structural and mechanical drivers rather than asymptotic statistical rejection.

---

## Results Summary

| Strategy | Ann. Return | Sharpe (Rf=0) | Sharpe (Rf=T-bill) | Max Drawdown |
|---|---|---|---|---|
| Dense Graph (Iter. 1) | 7.00% | 0.32 | — | −25.39% |
| MST Centrality (Iter. 2) | 12.55% | 0.56 | 0.56 | −30.59% |
| Regime Switch (Iter. 3) | 12.51% | 0.56 | **0.47** | −30.59% |
| Centrality Velocity (Iter. 4) | 6.44% | 0.37 | — | −29.38% |
| **Price Momentum (Baseline)** | **14.09%** | **0.70** | **0.70** | **−21.08%** |
| SPY Benchmark | 10.89% | 0.57 | 0.57 | −25.56% |

*Rf=T-bill Sharpe applies actual monthly 3-month Treasury bill rate (^IRX) to cash periods in Iteration 3 only.*  
*All Sharpe ratios use Rf=0 unless otherwise noted, for cross-strategy comparability.*

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/yourusername/sector-rotation-via-graph-centrality.git
cd sector-rotation-via-graph-centrality

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`** includes:
```
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
networkx>=3.1
matplotlib>=3.7.0
scipy>=1.11.0
```

> **Note on data access:** ETF price data is fetched live from Yahoo Finance via `yfinance`. T-bill rates are fetched from FRED via `yfinance` (ticker `^IRX`). An internet connection is required on first run. Subsequent runs use cached data if available.

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This runs all four iterations across multiple lookback windows (20, 60, 120 days), generates all figures, prints performance metrics tables, runs all statistical hypothesis tests, and runs walk-forward validation.

### Run individual modules

```python
from src.data_loader import fetch_sector_returns, fetch_tbill_rates
from src.graph_builder import calculate_mst_centrality
from src.backtester import run_backtest
from src.stats import run_all_tests
from src.walk_forward import run_walk_forward

# 1. Fetch data
returns = fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31")
tbill   = fetch_tbill_rates(start_date="2015-01-01",   end_date="2023-12-31")

# 2. Build rolling MST centrality (60-day window)
centrality_df = calculate_mst_centrality(returns, window=60)

# 3. Run backtest (all iterations)
run_backtest(returns, centrality_df, top_n=2)

# 4. Run statistical tests
from src.backtester import get_comparison_df  # returns the aligned DataFrame
comparison, tbill_aligned = get_comparison_df(returns, centrality_df)
run_all_tests(comparison, tbill_aligned=tbill_aligned)

# 5. Run walk-forward validation
run_walk_forward(returns)
```

### Run only statistical tests on existing results

```python
from src.stats import run_all_tests, jobson_korkie_test, bootstrap_sharpe_ci, hac_ttest

# Individual tests
jk  = jobson_korkie_test(strategy_returns, momentum_returns, rf=0.0)
bs  = bootstrap_sharpe_ci(strategy_returns, n_bootstrap=10000)
hac = hac_ttest(strategy_returns, momentum_returns)

print(f"JK Z-stat: {jk['z_stat']}, p-value: {jk['p_value']}")
print(f"Sharpe 95% CI: [{bs['ci_lower']}, {bs['ci_upper']}]")
print(f"HAC t-stat: {hac['t_stat']}, p-value: {hac['p_value']}")
```

---

## Module Reference

### `src/data_loader.py`

| Function | Description |
|---|---|
| `fetch_sector_returns(start, end)` | Downloads 11 GICS sector ETF prices via yfinance, returns daily log-return DataFrame with dynamic vertex scaling for XLRE/XLC inception dates |
| `fetch_tbill_rates(start, end)` | Downloads 3-month T-bill rate (^IRX) from FRED, converts annualised % to monthly decimal rate, returns month-end Series |

### `src/graph_builder.py`

| Function | Description |
|---|---|
| `calculate_mst_centrality(returns, window)` | Computes rolling Pearson correlation → affine transformation → Kruskal MST → Eigenvector Centrality for each day in the return series |

### `src/backtester.py`

| Function | Description |
|---|---|
| `run_backtest(returns, centrality, top_n)` | Runs Iterations 2 and 3 (MST Raw + Regime Switch), fetches T-bill and SPY data, computes all performance metrics, generates equity curve plots |
| `run_momentum_baseline(returns, lookback, top_n)` | 12-month cross-sectional momentum baseline with monthly rebalancing |
| `calculate_metrics(returns, benchmark, rf)` | Annualised return, volatility, Sharpe (with optional rf Series), max drawdown, information ratio |

### `src/stats.py`

| Function | Description |
|---|---|
| `jobson_korkie_test(r_a, r_b, rf)` | Jobson-Korkie (1981) test with Memmel (2003) correction for Sharpe ratio equality |
| `bootstrap_sharpe_ci(returns, rf, n_bootstrap, block_length)` | Block bootstrap confidence intervals for annualised Sharpe (Politis & Romano 1994) |
| `hac_ttest(r_a, r_b, max_lag)` | Newey-West HAC-corrected t-test on mean return differences |
| `run_all_tests(comparison_df, tbill_aligned)` | Master function — runs all three tests, prints formatted table |

### `src/walk_forward.py`

| Function | Description |
|---|---|
| `run_walk_forward(returns)` | Expanding-window walk-forward validation across 3 annual OOS folds, parameter grid w∈{30,60,90,120} |

### `src/visualizer.py`

Generates all six publication figures as both PDF (vector) and PNG (300 DPI raster). Figures are saved to the project root directory.

---

## Key Findings for Practitioners

**1. Topology is a Risk Manager, Not an Alpha Generator**  
MST-filtered centrality is highly effective at identifying hidden, systemic market stress *before* it is reflected in price. It should be used to size positions or manage drawdown risk, not as a standalone long-only signal.

**2. Beware Cap-Weighted Gravity**  
In highly concentrated markets (Magnificent 7 era, post-2019), structural macro-indicators are consistently overridden by pure price momentum. Any topological strategy must incorporate a cap-weighting adjustment or momentum overlay to avoid fighting the dominant index constituents. An N=11 sector MST is also topologically constrained — granular stock-level networks would provide richer structural information.

**3. Cash Drag is Regime-Dependent**  
Binary all-or-nothing liquidation triggers are counterproductive in liquidity-driven regimes (ZIRP, fiscal stimulus). A graduated position-scaling model — reducing to 50% on single defensive node penetration, 0% only on full defensive set dominance — would preserve upside capture while retaining systemic insulation.

**4. Kinematics Require Heavy Smoothing**  
Trading the velocity (ΔC) of a financial network without severe low-pass filtering causes catastrophic overfitting. Correlation matrices are inherently unstable during crises; first-differencing amplifies this noise. Centrality shifts must be evaluated over extended horizons (multi-month smoothing) to separate true structural rotation from transient volatility shocks.

**5. Transaction Costs Widen the Alpha Gap**  
MST Centrality and Regime Switch strategies maintain low turnover (~15% monthly) incurring ~22 bps annual drag. High-kinematic strategies (Dense Graph, Velocity) with near-total monthly reallocation (~200% turnover) incur ~300 bps annual drag — further widening the performance gap vs. price momentum after realistic transaction costs.

---

## Limitations

- **In-sample parameter selection:** The 60-day window and top-2 selection were evaluated statically. Walk-forward validation confirms robustness of qualitative rankings but does not eliminate this concern entirely.
- **GICS composition changes:** ETF constituent rebalances (e.g., META and GOOGL moving to XLC in 2018) affect the historical correlation structure. This is acknowledged but not corrected for.
- **Statistical power:** T=108 monthly observations yields bootstrap standard errors of ~±0.45 on annualised Sharpe ratios. The underperformance is economically meaningful but statistically indeterminate. A longer evaluation period or higher-frequency rebalancing would improve test power.
- **Gross of transaction costs:** All reported metrics are pre-cost. See Section 5.4 of the paper for the transaction cost drag framework.
- **Single market:** Results are specific to US large-cap sector ETFs. Generalisation to equal-weighted indices, international markets, or individual equity universes requires separate validation.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{sahu2024topological,
  title     = {Limitations of Topological Signals in Mega-Cap Momentum Markets:
               An Iterative Network Analysis of Sector Rotation},
  author    = {Sahu, Tanishq and Tiwari, Vishwam and Goveas, Neena},
  year      = {2024},
  note      = {Preprint},
  institution = {BITS Pilani, K.K. Birla Goa Campus}
}
```

**Corresponding author:** Tanishq Sahu — `f20240625@goa.bits-pilani.ac.in` · `sahutanishq06@gmail.com`

---

## References

Key references underpinning the methodology:

- Jegadeesh & Titman (1993) — Cross-sectional momentum premium
- Mantegna (1999) — Hierarchical structure in financial markets via graph theory
- Bonanno et al. (2003) — Topology of correlation-based MSTs in equity markets
- Onnela et al. (2003) — Dynamics of market correlations and MST taxonomy
- Vandewalle et al. (2001) — Market crashes and minimum spanning trees
- Bonacich (1987) — Eigenvector centrality
- Newman (2010) — Networks: An Introduction
- Laloux et al. (1999) — Noise dressing of financial correlation matrices
- Kacperczyk et al. (2005) — Industry concentration of actively managed funds
- Hou, Xue & Zhang (2020) — Replicating anomalies
- Faber (2007) — Quantitative approach to tactical asset allocation
- Jobson & Korkie (1981) + Memmel (2003) — Sharpe ratio hypothesis testing
- Politis & Romano (1994) — Stationary block bootstrap
- Newey & West (1987) — HAC-consistent covariance matrix

Full reference list in the paper.
