import pandas as pd
import yfinance as yf
from data_loader import load_prices, TICKERS, START_DATE
from strategy import get_top_picks

STARTING_CAPITAL = 10_000
N_PICKS = 3
TRANSACTION_COST = 0.001  # 0.1% cost per ticker traded each month


# finds the first trading day of each month in the price data
def get_month_starts(prices_df: pd.DataFrame) -> list[pd.Timestamp]:
    """
    Return a list of dates that are the first trading day of each month.
    These are the dates on which we rebalance the portfolio.
    """
    # resample to monthly frequency, taking the first date in each month
    return prices_df.resample("MS").first().dropna(how="all").index.tolist()


# runs the full backtest simulation month by month
def run_backtest(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate the momentum strategy from the start date to today.

    Each month:
      1. Score tickers by 12-1 momentum
      2. Buy the top N picks in equal weight
      3. Hold until next month, then repeat

    Parameters
    ----------
    prices_df : full DataFrame of daily close prices

    Returns
    -------
    DataFrame with one row per month containing portfolio value and holdings.
    """
    month_starts = get_month_starts(prices_df)

    portfolio_value = STARTING_CAPITAL
    results = []

    # we need at least 13 months of history before we can calculate 12-1 momentum,
    # so skip the first 13 months and start trading after that
    for i in range(13, len(month_starts) - 1):
        current_date = month_starts[i]
        next_date    = month_starts[i + 1]

        # get the top momentum picks as of the current rebalance date
        picks = get_top_picks(prices_df, current_date, n=N_PICKS)

        if not picks:
            # no valid picks - skip this month and stay in cash
            results.append({
                "date": current_date,
                "portfolio_value": portfolio_value,
                "tickers": [],
            })
            continue

        # calculate the equal-weight return for this month
        # each ticker gets an equal slice of the portfolio
        weight = 1.0 / len(picks)
        monthly_return = 0.0

        for ticker in picks:
            # price at start of this month and start of next month
            price_start = prices_df.loc[:current_date, ticker].iloc[-1]
            price_end   = prices_df.loc[:next_date,    ticker].iloc[-1]

            if pd.isna(price_start) or pd.isna(price_end):
                # if we can't price a holding, treat that slice as flat (0% return)
                continue

            ticker_return = price_end / price_start - 1
            monthly_return += weight * ticker_return

        # subtract 0.1% cost for each ticker we're trading this month
        # this approximates real-world spread and slippage
        total_cost = TRANSACTION_COST * len(picks)

        # update portfolio value: apply the return then deduct trading costs
        portfolio_value = portfolio_value * (1 + monthly_return) * (1 - total_cost)

        results.append({
            "date": current_date,
            "portfolio_value": round(portfolio_value, 2),
            "tickers": ",".join(picks),
        })

    return pd.DataFrame(results).set_index("date")


# downloads SPY prices and calculates what a buy-and-hold SPY investment
# would have returned over the same period as our backtest
def get_spy_benchmark(start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """
    Return the total SPY return from start_date to end_date.
    Used to compare our strategy against simply buying and holding the S&P 500.
    """
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"].squeeze()
        return float(spy.iloc[-1] / spy.iloc[0] - 1)
    except Exception as e:
        print(f"Could not download SPY benchmark: {e}")
        return 0.0


if __name__ == "__main__":
    print("Loading prices...")
    prices = load_prices(TICKERS, START_DATE)

    print("Running backtest...")
    results = run_backtest(prices)

    start_date = results.index[0]
    end_date   = results.index[-1]
    spy_return = get_spy_benchmark(start_date, end_date)

    total_return = results['portfolio_value'].iloc[-1] / STARTING_CAPITAL - 1

    print(f"\nMonths simulated: {len(results)}")
    print(f"Starting capital: ${STARTING_CAPITAL:,.0f}")
    print(f"Final value:      ${results['portfolio_value'].iloc[-1]:,.0f}")
    print(f"Total return:     {total_return:.1%}")
    print(f"SPY return:       {spy_return:.1%}")
