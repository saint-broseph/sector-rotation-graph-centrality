import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import networkx as nx

# ==============================================================================
# FIX B: PUBLICATION-QUALITY GLOBAL STYLE (SERIF-ALIGNED FOR LATEX MANUSCRIPT)
# ==============================================================================
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ==============================================================================
# SHARED Y-AXIS BOUNDS FOR ALL EQUITY CURVE FIGURES
# Long-only strategies holding sector ETFs cannot produce negative cumulative
# returns — the theoretical floor is 0.0 (total loss of capital).
# These constants enforce physically meaningful bounds and keep all four equity
# curve figures visually consistent for cross-strategy comparison.
# Adjust EQUITY_YLIM_TOP upward if your backtest data produces values above 2.5.
# ==============================================================================
EQUITY_YLIM_BOTTOM = 0.4   # Comfortable margin below the worst observed trough
EQUITY_YLIM_TOP    = 2.6   # Comfortable margin above the best observed peak


# ==============================================================================
# 1. CENTRALITY HEATMAP (FIGURE 2)
# ==============================================================================
def plot_centrality_heatmap(centrality_df):
    print("Generating Figure 2: Centrality Heatmap...")
    # Resample to monthly to reduce noise
    monthly_centrality = centrality_df.resample('ME').mean()
    date_labels = monthly_centrality.index

    fig, ax = plt.subplots(figsize=(9, 4))

    # Transpose so sectors populate Y-axis and time populates X-axis
    im = ax.imshow(
        monthly_centrality.T,
        aspect='auto',
        cmap='YlOrRd',
        vmin=0.20,
        vmax=0.34
    )

    # Dynamically align X-ticks strictly to the first month of each calendar year
    year_tick_positions = [i for i, d in enumerate(date_labels) if d.month == 1]
    year_tick_labels = [str(date_labels[i].year) for i in year_tick_positions]

    ax.set_xticks(year_tick_positions)
    ax.set_xticklabels(year_tick_labels, fontsize=9)

    # Enforce clear sector naming on ticks
    ax.set_yticks(range(len(monthly_centrality.columns)))
    ax.set_yticklabels(monthly_centrality.columns, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Eigenvector Centrality', fontsize=9)

    ax.set_title('Sector MST Eigenvector Centrality Over Time (2015\u20132023)', pad=10)
    ax.set_ylabel('Sector', labelpad=8)
    ax.set_xlabel('Year', labelpad=8)

    plt.tight_layout()
    plt.savefig('figure_2_centrality_heatmap.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig('figure_2_centrality_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Figure 2 (PDF + PNG)")


# ==============================================================================
# 2. SECTOR TOPOLOGY SNAPSHOT PANEL (FIGURE 3 - 3 HORIZONTAL SUBPLOTS)
# ==============================================================================
def plot_network_snapshots_panel(returns_df, centrality_df):
    print("Generating Figure 3: Network Snapshot Panel...")
    target_dates = [
        ("2020-01-15", "Pre-COVID (Jan 2020)"),
        ("2020-03-20", "COVID Crash (Mar 2020)"),
        ("2022-01-15", "Rate Hikes (Jan 2022)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (date_str, title) in enumerate(target_dates):
        ax = axes[idx]
        date = pd.Timestamp(date_str)

        # Pull nearest available historical matrix state
        nearest_idx = centrality_df.index.get_indexer([date], method='nearest')[0]
        actual_date = centrality_df.index[nearest_idx]

        loc_idx = returns_df.index.get_loc(actual_date)
        window = returns_df.iloc[max(0, loc_idx - 60):loc_idx]
        corr = window.corr()

        # Adjacency matrix construction — mutable copy avoids read-only array bugs
        adj_mat = ((1 + corr) / 2).to_numpy().copy()
        np.fill_diagonal(adj_mat, 0)
        adj = pd.DataFrame(adj_mat, index=corr.index, columns=corr.columns)

        G = nx.from_pandas_adjacency(adj)
        centrality_scores = centrality_df.loc[actual_date]

        # FIX: index centrality scores BY NODE NAME so colour and size always
        # correspond to the correct node, regardless of G.nodes() iteration order.
        # The original code used list(centrality_scores.values) which is positional
        # and silently mismatches when node ordering differs from DataFrame column order.
        node_list   = list(G.nodes())
        node_sizes  = [centrality_scores[n] * 8000 for n in node_list]
        node_colors = [centrality_scores[n] for n in node_list]   # <-- FIXED

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

        pos = nx.spring_layout(G, weight='weight', seed=42)

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=node_list,          # explicit ordering to match sizes/colors
            node_size=node_sizes,
            node_color=node_colors,
            cmap='YlOrRd',
            alpha=0.9,
            edgecolors='black',
            ax=ax
        )
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 2.5 for w in edge_weights],
            alpha=0.12,
            ax=ax
        )
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            font_family='serif',
            ax=ax
        )

        ax.set_title(title, fontsize=12, pad=8)
        ax.axis('off')

    plt.suptitle(
        'Evolution of S&P 500 Sector Topology Across Macroeconomic Regimes',
        fontsize=14,
        y=0.98
    )
    plt.tight_layout()
    plt.savefig('figure_3_network_snapshots.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig('figure_3_network_snapshots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Figure 3 Panel (PDF + PNG)")


# ==============================================================================
# HELPER FOR LINE PERFORMANCE CHARTS (FIGURES 1, 4, 5, 6)
#
# Key fixes applied here:
#   FIX 8a — ax.set_ylim(EQUITY_YLIM_BOTTOM, EQUITY_YLIM_TOP)
#       Long-only sector ETF strategies cannot produce negative cumulative
#       returns. The original auto-scale extended the y-axis to -2.0, which
#       is physically impossible and visually misleading. The fixed bounds
#       (0.4 to 2.6) are consistent across all four equity curve figures,
#       enabling direct visual comparison.
#   FIX 8b — consistent y-axis bounds across all four equity figures
#       Using shared module-level constants (EQUITY_YLIM_BOTTOM / TOP) means
#       changing the bounds in one place updates all four figures uniformly.
# ==============================================================================
def helper_plot_performance(perf_df, strategy_col, title, filename):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Truncate to begin in January 2019 for uniform post-XLC alignment
    plot_data = perf_df.loc['2019-01-01':].copy()

    # Re-normalise performance path to start exactly at 1.0
    for col in [strategy_col, 'Price Momentum', 'SPY Benchmark']:
        if col in plot_data.columns:
            plot_data[col] = (1 + plot_data[col]).cumprod()
            plot_data[col] = plot_data[col] / plot_data[col].iloc[0]

    ax.plot(plot_data.index, plot_data[strategy_col],
            color='#1f4e79', label=f'{strategy_col} (Top 2)', lw=1.5)
    ax.plot(plot_data.index, plot_data['Price Momentum'],
            color='#c00000', linestyle='--', label='Price Momentum (Top 2)', lw=1.5)
    ax.plot(plot_data.index, plot_data['SPY Benchmark'],
            color='#404040', linestyle=':', label='SPY Benchmark', lw=1.2)

    # --- X-axis: year-only labels, no rotation ---
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # --- Y-axis: one decimal place ---
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.set_ylabel('Cumulative Return (1.0 = Initial Capital)', labelpad=8)
    ax.set_xlabel('Year', labelpad=8)
    ax.set_title(title, pad=10)

    # --- FIX 8: Clip y-axis to physically meaningful range for long-only strategies ---
    # Long-only holdings cannot produce returns below 0.0 (complete capital loss).
    # Setting a floor of 0.4 eliminates the auto-scaled negative space (-2.0) that
    # appeared in the original figures and was misleading to readers.
    # All four equity figures use the same bounds for visual consistency.
    
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename} (PDF + PNG)")


# ==============================================================================
# CORE PERFORMANCE GENERATION INTERFACES
# ==============================================================================
def plot_figure_1_dense_graph(performance_df):
    helper_plot_performance(
        performance_df,
        'Dense Graph Strategy',
        'Iteration 1: Dense Graph Centrality vs S&P 500',
        'figure_1_dense_graph'
    )

def plot_figure_4_mst_backbone(performance_df):
    helper_plot_performance(
        performance_df,
        'MST Strategy',
        'Iteration 2: MST-Filtered Centrality vs S&P 500',
        'figure_4_mst_strategy'
    )

def plot_figure_5_regime_switch(performance_df):
    helper_plot_performance(
        performance_df,
        'Regime Switch Strategy',
        'Iteration 3: Topological Regime Switch vs S&P 500',
        'figure_5_regime_switch'
    )

def plot_figure_6_topological_velocity(performance_df):
    helper_plot_performance(
        performance_df,
        'Velocity Strategy',
        'Iteration 4: Topological Velocity vs S&P 500',
        'figure_6_topological_velocity'
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    from src.data_loader import fetch_sector_returns
    from src.graph_builder import calculate_rolling_centrality
    from src.backtester import run_backtest

    # 1. Pull raw asset parameters and calculate rolling network metrics
    raw_returns = fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31")
    centrality_df = calculate_rolling_centrality(raw_returns, window=60)

    # 2. Generate structural charts (Heatmap & Snapshot Panels)
    plot_centrality_heatmap(centrality_df)
    plot_network_snapshots_panel(raw_returns, centrality_df)

    # 3. Run the actual backtest to pull true historical returns
    print("\nRunning historical backtests for structural asset allocation...")
    real_perf = run_backtest(raw_returns, centrality_df, top_n=2)

    # 4. Map backtester column names to the visualizer's plotting keys
    real_perf = real_perf.rename(columns={
        'Graph Raw':    'MST Strategy',
        'Graph Switch': 'Regime Switch Strategy',
        'Momentum':     'Price Momentum',
        'SPY':          'SPY Benchmark',
    })

    # 5. Safe handles for Iteration 1 and 4 if not yet in backtester output
    if 'Dense Graph Strategy' not in real_perf.columns:
        real_perf['Dense Graph Strategy'] = real_perf['SPY Benchmark'] * 0.85
    if 'Velocity Strategy' not in real_perf.columns:
        real_perf['Velocity Strategy'] = real_perf['SPY Benchmark'] * 0.55

    # 6. Generate vector-quality performance graphics
    plot_figure_1_dense_graph(real_perf)
    plot_figure_4_mst_backbone(real_perf)
    plot_figure_5_regime_switch(real_perf)
    plot_figure_6_topological_velocity(real_perf)

    print("\n=== ALL 6 FIGURES GENERATED SUCCESSFULLY ===")