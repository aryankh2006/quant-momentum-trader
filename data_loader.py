import pandas as pd
import yfinance as yf

# the stocks/etfs we're trading
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG"]
START_DATE = "2018-01-01"


# downloads daily close prices for all tickers and returns one combined DataFrame
def load_prices(tickers, start_date):
    data = yf.download(tickers, start=start_date)["Close"]
    return data


if __name__ == "__main__":
    prices = load_prices(TICKERS, START_DATE)
    print(prices.shape)
    print(prices.head())
