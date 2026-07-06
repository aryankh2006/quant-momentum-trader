import pandas as pd
from data_loader import load_prices, TICKERS, START_DATE

# 63 trading days is roughly 3 months (markets open ~21 days/month)
LOOKBACK = 63


# scores each ticker by its 3-month return and returns the top n tickers
def get_top_picks(prices_df: pd.DataFrame, date: pd.Timestamp, n: int = 3) -> list[str]:
    """
    Rank tickers by 3-month momentum and return the top n.

    Parameters
    ----------
    prices_df : DataFrame of daily close prices (dates as index, tickers as columns)
    date      : the date we are scoring on
    n         : how many top tickers to return

    Returns
    -------
    List of ticker strings, best momentum first.
    """
    # only look at prices up to and including our target date
    # (we must never look into the future when backtesting)
    history = prices_df.loc[:date]

    # we need at least LOOKBACK rows of history to calculate a return
    if len(history) < LOOKBACK:
        print(f"Not enough history before {date.date()} to score tickers")
        return []

    scores = {}
    for ticker in prices_df.columns:
        price_today = history[ticker].iloc[-1]        # most recent close
        price_past = history[ticker].iloc[-LOOKBACK]  # close 63 days ago

        # calculate the percentage return over the lookback window
        scores[ticker] = price_today / price_past - 1

    # sort tickers from highest to lowest momentum score
    ranked = sorted(scores, key=lambda t: scores[t], reverse=True)

    return ranked[:n]


if __name__ == "__main__":
    prices = load_prices(TICKERS, START_DATE)

    # test on a recent date
    test_date = prices.index[-1]
    picks = get_top_picks(prices, test_date, n=3)

    print(f"Top 3 picks as of {test_date.date()}: {picks}")
