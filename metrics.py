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
    # ddof=1 uses sample standard deviation (divides by n-1, not n),
    # which is the correct convention for financial Sharpe calculation
    std_excess  = excess_returns.std(ddof=1)

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

    # --- Rolling 12-month Sharpe ---
    # stored separately as a Series (one value per month) so the dashboard
    # can plot how the risk-adjusted performance changed over time
    metrics["rolling_sharpe"] = rolling_sharpe(returns_series)

    return metrics


# prints a clean formatted performance report from a metrics dict
def print_report(metrics_dict: dict) -> None:
    """
    Print a formatted summary of strategy performance metrics.

    Parameters
    ----------
    metrics_dict : output from calculate_metrics()
    """
    W = 48  # total width of the report box

    print("=" * W)
    print(f"{'STRATEGY PERFORMANCE REPORT':^{W}}")
    print("=" * W)

    # :<30 = left-align label in 30 chars, :>10.2% = right-align value as percentage
    print(f"{'CAGR':<30}{metrics_dict['CAGR']:>10.2%}")
    print(f"{'Sharpe Ratio':<30}{metrics_dict['Sharpe']:>10.2f}")
    print(f"{'Max Drawdown':<30}{metrics_dict['max_drawdown']:>10.2%}")
    print(f"{'Win Rate':<30}{metrics_dict['win_rate']:>10.2%}")
    print("-" * W)
    print(f"{'Benchmark CAGR (SPY)':<30}{metrics_dict['benchmark_CAGR']:>10.2%}")
    print("=" * W)


# calculates Sharpe ratio over a rolling 12-month window
def rolling_sharpe(returns_series: pd.Series, window: int = 12) -> pd.Series:
    """
    Compute the Sharpe ratio over a rolling window of months.

    A rolling Sharpe shows whether the strategy's risk-adjusted performance
    improved or declined over time — useful for spotting regime changes.

    Parameters
    ----------
    returns_series : monthly returns of the strategy
    window         : number of months in the rolling window (default 12)

    Returns
    -------
    pd.Series of rolling Sharpe values, one per month (NaN for first 11 months).
    """
    monthly_rf = RISK_FREE_RATE / MONTHS_PER_YEAR
    excess     = returns_series - monthly_rf

    # for each 12-month window: mean excess return / std, annualised
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std()

    # avoid division by zero in flat periods
    rolling = (roll_mean / roll_std.replace(0, float("nan"))) * math.sqrt(MONTHS_PER_YEAR)

    return rolling


if __name__ == "__main__":
    import yfinance as yf

    # load backtest results saved by backtester.py
    try:
        df = pd.read_csv(
            "results/performance_report.csv",
            index_col="date",
            parse_dates=True,
        )
    except FileNotFoundError:
        print("No results found. Run backtester.py first.")
        raise SystemExit

    # monthly returns of our strategy
    strategy_returns = df["portfolio_value"].pct_change().dropna()

    # download real SPY data for the same date range as our backtest
    try:
        spy_raw = yf.download(
            "SPY",
            start=df.index[0],
            end=df.index[-1],
            progress=False,
        )["Close"].squeeze()

        # resample SPY daily prices to monthly, then calculate monthly returns
        spy_monthly   = spy_raw.resample("MS").first()
        spy_returns   = spy_monthly.pct_change().dropna()

        # align SPY to exactly the same months as our strategy
        spy_returns = spy_returns.reindex(strategy_returns.index).fillna(0)

    except Exception as e:
        print(f"Could not load SPY benchmark: {e} — using zeros")
        spy_returns = pd.Series(0, index=strategy_returns.index)

    metrics = calculate_metrics(strategy_returns, spy_returns)
    print_report(metrics)
