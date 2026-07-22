import pandas as pd
from data_loader import load_prices, TICKERS, START_DATE

# 12-1 momentum uses a 12-month lookback (252 trading days) but skips
# the most recent month (21 trading days) to avoid short-term reversal
LOOKBACK_12M = 252  # ~12 months in trading days
SKIP_1M = 21        # ~1 month in trading days


# scores each ticker by 12-1 momentum and returns the top n tickers
def get_top_picks(prices_df: pd.DataFrame, date: pd.Timestamp, n: int = 3) -> list[str]:
    """
    Rank tickers by 12-1 momentum and return the top n.

    12-1 momentum = return from 12 months ago to 1 month ago, skipping
    the most recent month. Research shows skipping the last month improves
    performance because very recent winners tend to briefly reverse.

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

    # we need at least 12 months of history to calculate the signal
    if len(history) < LOOKBACK_12M:
        return []  # silently skip — backtester handles the empty result

    scores = {}
    for ticker in prices_df.columns:
        # price[t-21]  = 1 month ago  (the "recent" end of our window)
        # price[t-252] = 12 months ago (the "far" end of our window)
        price_recent = history[ticker].iloc[-SKIP_1M]   # 1 month ago
        price_past   = history[ticker].iloc[-LOOKBACK_12M]  # 12 months ago

        # skip this ticker if either price is missing - a NaN return would
        # corrupt the ranking and could bubble a broken ticker to the top
        if pd.isna(price_recent) or pd.isna(price_past):
            continue

        # 12-1 momentum score: return from 12 months ago to 1 month ago
        scores[ticker] = price_recent / price_past - 1

    # sort tickers from highest to lowest momentum score
    ranked = sorted(scores, key=lambda t: scores[t], reverse=True)

    # return however many valid tickers we have, up to n
    return ranked[:n]


if __name__ == "__main__":
    prices = load_prices(TICKERS, START_DATE)

    # test on the most recent available date
    test_date = prices.index[-1]
    picks = get_top_picks(prices, test_date, n=3)

    print(f"\nTop 3 picks as of {test_date.date()} (12-1 momentum):")
    for ticker in picks:
        # recalculate score just for display purposes
        history = prices.loc[:test_date]
        score = history[ticker].iloc[-SKIP_1M] / history[ticker].iloc[-LOOKBACK_12M] - 1
        print(f"  {ticker}: {score:.1%}")
