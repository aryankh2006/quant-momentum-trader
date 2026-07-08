import math
import pandas as pd

RISK_FREE_RATE = 0.04   # 4% annual risk-free rate (approximate US T-bill rate)
MONTHS_PER_YEAR = 12


# calculates CAGR, Sharpe, max drawdown, and win rate from a monthly return series
def calculate_metrics(returns_series: pd.Series, benchmark_series: pd.Series) -> dict:
    """
    Calculate key performance metrics for a strategy vs a benchmark.

    Parameters
    ----------
    returns_series   : monthly returns of the strategy (e.g. 0.05 = 5% that month)
    benchmark_series : monthly returns of the benchmark (e.g. SPY)

    Returns
    -------
    Dict containing: CAGR, Sharpe, max_drawdown, win_rate
    """
    metrics = {}

    # --- CAGR ---
    # grow $1 by each monthly return to get the final multiplier,
    # then annualise it based on how many years the series covers
    n_months = len(returns_series)
    n_years  = n_months / MONTHS_PER_YEAR

    # (1 + r1) * (1 + r2) * ... gives us the total growth factor
    total_growth = (1 + returns_series).prod()
    metrics["CAGR"] = total_growth ** (1 / n_years) - 1

    # --- Sharpe Ratio ---
    # monthly risk-free rate = annual rate / 12
    monthly_rf = RISK_FREE_RATE / MONTHS_PER_YEAR

    # excess return = what we earned above what a risk-free investment would have earned
    excess_returns = returns_series - monthly_rf

    mean_excess = excess_returns.mean()
    std_excess  = excess_returns.std()

    if std_excess == 0:
        metrics["Sharpe"] = 0.0
    else:
        # multiply by sqrt(12) to annualise from monthly to yearly
        metrics["Sharpe"] = (mean_excess / std_excess) * math.sqrt(MONTHS_PER_YEAR)

    # --- Max Drawdown ---
    # rebuild the portfolio value curve from returns, starting at $1
    portfolio_curve = (1 + returns_series).cumprod()

    # rolling peak = highest value the portfolio has ever reached up to each month
    rolling_peak = portfolio_curve.cummax()

    # drawdown at each point = how far below the peak we currently are
    drawdown = (portfolio_curve - rolling_peak) / rolling_peak

    # the worst (most negative) drawdown across the whole period
    metrics["max_drawdown"] = drawdown.min()

    # --- Win Rate ---
    # percentage of months where the strategy made money
    winning_months = (returns_series > 0).sum()
    metrics["win_rate"] = winning_months / n_months

    # --- Benchmark CAGR for comparison ---
    bench_growth = (1 + benchmark_series).prod()
    metrics["benchmark_CAGR"] = bench_growth ** (1 / n_years) - 1

    return metrics
