import pandas as pd
import yfinance as yf

# the stocks/etfs we're trading
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG"]
START_DATE = "2018-01-01"


# downloads daily close prices for all tickers and returns one combined DataFrame
def load_prices(tickers, start_date):
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

    return data


if __name__ == "__main__":
    prices = load_prices(TICKERS, START_DATE)
    print(prices.shape)
    print(prices.head())
