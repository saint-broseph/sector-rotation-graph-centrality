import numpy as np
import pandas as pd
from scipy import stats


# ── 1. JOBSON-KORKIE TEST ──────────────────────────────────────────────────────
# Tests H0: Sharpe(A) == Sharpe(B)
# Memmel (2003) correction applied — this is the standard form used in
# empirical finance. Cite as: Jobson & Korkie (1981), Memmel (2003).

def jobson_korkie_test(r_a, r_b, rf=0.0):
    """
    Tests whether two Sharpe ratios are statistically equal.

    Parameters
    ----------
    r_a : pd.Series  — monthly returns of strategy A
    r_b : pd.Series  — monthly returns of strategy B
    rf  : float      — monthly risk-free rate (default 0.0)

    Returns
    -------
    dict with keys: sharpe_a, sharpe_b, z_stat, p_value, significant
    """
    n = len(r_a)
    excess_a = r_a - rf
    excess_b = r_b - rf

    mu_a  = excess_a.mean()
    mu_b  = excess_b.mean()
    sig_a = excess_a.std(ddof=1)
    sig_b = excess_b.std(ddof=1)

    sr_a = mu_a / sig_a   # monthly Sharpe
    sr_b = mu_b / sig_b

    # Annualise for reporting
    sr_a_ann = sr_a * np.sqrt(12)
    sr_b_ann = sr_b * np.sqrt(12)

    # Covariance terms needed for the Memmel variance estimator
    sig_ab = np.cov(excess_a, excess_b, ddof=1)[0, 1]
    rho    = sig_ab / (sig_a * sig_b)

    # Memmel (2003) asymptotic variance of (SR_a - SR_b)
    var_diff = (1/n) * (
        2
        - 2 * rho
        + 0.5 * sr_a**2
        + 0.5 * sr_b**2
        - sr_a * sr_b * (rho**2 + 0.5)
    )

    if var_diff <= 0:
        return {
            'sharpe_a': sr_a_ann, 'sharpe_b': sr_b_ann,
            'z_stat': np.nan, 'p_value': np.nan, 'significant': False
        }

    z_stat  = (sr_a - sr_b) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))   # two-tailed

    return {
        'sharpe_a'   : round(sr_a_ann, 4),
        'sharpe_b'   : round(sr_b_ann, 4),
        'z_stat'     : round(z_stat, 4),
        'p_value'    : round(p_value, 4),
        'significant': p_value < 0.05
    }


# ── 2. BLOCK BOOTSTRAP SHARPE CONFIDENCE INTERVALS ───────────────────────────
# Standard approach for autocorrelated financial return series.
# Block length rule-of-thumb: T^(1/3) — see Politis & Romano (1994).

def bootstrap_sharpe_ci(returns, rf=0.0, n_bootstrap=10000,
                         confidence=0.95, block_length=None):
    """
    Block bootstrap confidence interval for annualised Sharpe ratio.

    Parameters
    ----------
    returns      : pd.Series — monthly returns
    rf           : float     — monthly risk-free rate
    n_bootstrap  : int       — number of bootstrap replications
    confidence   : float     — confidence level (default 0.95)
    block_length : int|None  — block size; defaults to T^(1/3)

    Returns
    -------
    dict with keys: sharpe, ci_lower, ci_upper, std_error
    """
    excess   = (returns - rf).values
    n        = len(excess)
    bl       = block_length or max(1, int(round(n ** (1/3))))
    n_blocks = int(np.ceil(n / bl))

    bootstrap_sharpes = []
    rng = np.random.default_rng(seed=42)   # reproducible

    for _ in range(n_bootstrap):
        # Draw random block start indices
        starts  = rng.integers(0, n - bl + 1, size=n_blocks)
        sample  = np.concatenate([excess[s:s + bl] for s in starts])[:n]

        mu  = sample.mean()
        sig = sample.std(ddof=1)
        if sig > 0:
            bootstrap_sharpes.append((mu / sig) * np.sqrt(12))

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    alpha = 1 - confidence
    ci_lower, ci_upper = np.percentile(
        bootstrap_sharpes, [alpha/2 * 100, (1 - alpha/2) * 100]
    )

    actual_sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(12)

    return {
        'sharpe'    : round(actual_sharpe, 4),
        'ci_lower'  : round(ci_lower, 4),
        'ci_upper'  : round(ci_upper, 4),
        'std_error' : round(bootstrap_sharpes.std(), 4),
        'block_length': bl
    }


# ── 3. HAC-CORRECTED T-TEST ON RETURN DIFFERENCES ────────────────────────────
# Newey-West HAC standard errors correct for autocorrelation and
# heteroskedasticity in the return difference series.
# Cite as: Newey & West (1987).

def hac_ttest(r_a, r_b, max_lag=None):
    """
    Tests H0: mean(r_a - r_b) == 0 using Newey-West HAC standard errors.

    Parameters
    ----------
    r_a     : pd.Series — monthly returns of strategy A
    r_b     : pd.Series — monthly returns of strategy B
    max_lag : int|None  — HAC lag truncation; defaults to 4*((T/100)^(2/9))

    Returns
    -------
    dict with keys: mean_diff, hac_se, t_stat, p_value, significant
    """
    diff  = (r_a - r_b).values
    n     = len(diff)
    mu    = diff.mean()

    # Newey-West lag selection
    if max_lag is None:
        max_lag = int(np.floor(4 * ((n / 100) ** (2/9))))

    # HAC variance estimator
    gamma_0 = np.sum((diff - mu)**2) / n
    hac_var = gamma_0

    for lag in range(1, max_lag + 1):
        weight  = 1 - lag / (max_lag + 1)          # Bartlett kernel
        gamma_l = np.sum(
            (diff[lag:] - mu) * (diff[:-lag] - mu)
        ) / n
        hac_var += 2 * weight * gamma_l

    hac_se = np.sqrt(hac_var / n)
    t_stat = mu / hac_se if hac_se > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    return {
        'mean_diff'  : round(mu * 12, 4),      # annualised
        'hac_se'     : round(hac_se * np.sqrt(12), 4),
        't_stat'     : round(t_stat, 4),
        'p_value'    : round(p_value, 4),
        'significant': p_value < 0.05,
        'max_lag'    : max_lag
    }


# ── 4. MASTER FUNCTION ────────────────────────────────────────────────────────

def run_all_tests(comparison_df, tbill_aligned=None):
    """
    Runs all three tests comparing each topological strategy
    against the Price Momentum baseline.

    Parameters
    ----------
    comparison_df : pd.DataFrame with columns:
                    'Graph Raw', 'Graph Switch', 'Momentum', 'SPY'
    tbill_aligned : pd.Series of monthly T-bill rates aligned to
                    comparison_df.index (optional, defaults to 0.0)

    Returns
    -------
    results : dict of all test outputs (also prints a formatted table)
    """
    if tbill_aligned is None:
        rf_series = pd.Series(0.0, index=comparison_df.index)
    else:
        rf_series = tbill_aligned

    strategies = {
        'MST Raw (Iter 2)'   : comparison_df['Graph Raw'],
        'Regime Switch (Iter 3)': comparison_df['Graph Switch'],
    }
    baseline = comparison_df['Momentum']

    results = {}

    print("\n" + "="*70)
    print("STATISTICAL HYPOTHESIS TESTS")
    print("Null hypothesis for all tests: strategy performance == momentum baseline")
    print("="*70)

    # ── Jobson-Korkie ──────────────────────────────────────────────────────────
    print("\n── 1. JOBSON-KORKIE SHARPE RATIO EQUALITY TEST ──────────────────────")
    print(f"{'Strategy':<25} {'SR(strat)':>10} {'SR(mom)':>10} "
          f"{'Z-stat':>10} {'p-value':>10} {'Sig?':>8}")
    print("-" * 75)

    for name, ret in strategies.items():
        jk = jobson_korkie_test(ret, baseline, rf=0.0)
        results[f'jk_{name}'] = jk
        sig_str = "YES *" if jk['significant'] else "no"
        print(f"{name:<25} {jk['sharpe_a']:>10.3f} {jk['sharpe_b']:>10.3f} "
              f"{jk['z_stat']:>10.3f} {jk['p_value']:>10.4f} {sig_str:>8}")

    print("\n  * p < 0.05 (two-tailed). Significant = can reject equal Sharpe.")

    # ── Block Bootstrap CIs ────────────────────────────────────────────────────
    print("\n── 2. BLOCK BOOTSTRAP SHARPE CONFIDENCE INTERVALS (95%) ────────────")
    print(f"{'Strategy':<25} {'Sharpe':>8} {'CI Lower':>10} "
          f"{'CI Upper':>10} {'Std Err':>10} {'Block':>7}")
    print("-" * 75)

    all_series = {**strategies, 'Momentum': baseline, 'SPY': comparison_df['SPY']}
    for name, ret in all_series.items():
        bs = bootstrap_sharpe_ci(ret, rf=0.0)
        results[f'bs_{name}'] = bs
        print(f"{name:<25} {bs['sharpe']:>8.3f} {bs['ci_lower']:>10.3f} "
              f"{bs['ci_upper']:>10.3f} {bs['std_error']:>10.3f} "
              f"{bs['block_length']:>7}")

    print("\n  Block bootstrap (Politis & Romano 1994), 10,000 replications, "
          f"seed=42.")

    # ── HAC t-test ─────────────────────────────────────────────────────────────
    print("\n── 3. HAC-CORRECTED t-TEST ON RETURN DIFFERENCES ───────────────────")
    print(f"{'Strategy vs Momentum':<25} {'Ann.Diff':>10} {'HAC SE':>10} "
          f"{'t-stat':>10} {'p-value':>10} {'Sig?':>8}")
    print("-" * 75)

    for name, ret in strategies.items():
        hac = hac_ttest(ret, baseline)
        results[f'hac_{name}'] = hac
        sig_str = "YES *" if hac['significant'] else "no"
        print(f"{name:<25} {hac['mean_diff']:>10.4f} {hac['hac_se']:>10.4f} "
              f"{hac['t_stat']:>10.3f} {hac['p_value']:>10.4f} {sig_str:>8}")

    print("\n  * p < 0.05. Newey-West HAC SE, lag =",
          results[f"hac_MST Raw (Iter 2)"]['max_lag'],
          "(automatic selection).")
    print("  Ann.Diff = annualised mean return difference (strategy minus momentum).")

    print("\n" + "="*70)

    return results