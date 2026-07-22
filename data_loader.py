import pandas as pd
import yfinance as yf

# the stocks/etfs we're trading
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG"]
START_DATE = "2018-01-01"


# downloads daily close prices for all tickers and returns one combined DataFrame
def load_prices(tickers: list[str], start_date: str) -> pd.DataFrame:
    """
    Download daily closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers    : list of ticker strings, e.g. ["AAPL", "MSFT"]
    start_date : start of the date range, e.g. "2018-01-01"

    Returns
    -------
    pd.DataFrame with dates as the index and tickers as columns.
    Returns an empty DataFrame if the download fails entirely.
    """
    try:
        data = yf.download(tickers, start=start_date)["Close"]
    except Exception as e:
        # yfinance can fail on network issues or rate limits, so don't let
        # the whole script crash - just tell the user what happened
        print(f"Failed to download prices: {e}")
        return pd.DataFrame()

    # if a ticker fails individually it shows up as an all-NaN column,
    # warn about it instead of silently passing bad data downstream
    missing = data.columns[data.isna().all()].tolist()
    if missing:
        print(f"Warning: no data returned for {missing}")

    # drop rows where every single ticker is NaN - those are non-trading days
    # that sneak in at the edges of the date range (e.g. weekends, holidays)
    data = data.dropna(how="all")

    return data


if __name__ == "__main__":
    prices = load_prices(TICKERS, START_DATE)
    print(f"Shape: {prices.shape[0]} trading days x {prices.shape[1]} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"\nFirst 5 rows:\n{prices.head()}")
