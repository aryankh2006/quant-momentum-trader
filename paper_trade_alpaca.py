import os
import pandas as pd
from datetime import datetime
from data_loader import load_prices, TICKERS, START_DATE
from strategy import get_top_picks
from risk import apply_risk_rules

TRADE_LOG_PATH = "results/trade_log.csv"

# ── DRY RUN FLAG ──────────────────────────────────────────────────────────────
# when True, prints what orders WOULD be submitted but doesn't actually send them
# set to False only when you're ready to trade on your real paper account
DRY_RUN = True

# ── ALPACA CONNECTION ─────────────────────────────────────────────────────────
# API keys are loaded from environment variables so they never appear in code
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"  # paper trading endpoint


# connects to Alpaca and returns the API object
def get_alpaca_api():
    """
    Connect to Alpaca paper trading API using environment variable keys.
    Returns the API object if successful, raises an error if keys are missing.
    """
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        raise ImportError("Install alpaca-trade-api: pip install alpaca-trade-api")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise ValueError(
            "Missing Alpaca API keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "as environment variables before running this script."
        )

    try:
        api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL,
        )
        # test the connection by fetching account info
        account = api.get_account()
        print(f"Connected to Alpaca. Portfolio value: ${float(account.portfolio_value):,.2f}")
        return api
    except Exception as e:
        raise ConnectionError(f"Could not connect to Alpaca: {e}")


# fetches current positions from Alpaca and returns {ticker: market_value}
def get_current_positions(api) -> dict[str, float]:
    """
    Return current Alpaca positions as a dict of {ticker: market_value_in_dollars}.
    Returns an empty dict if there are no open positions.
    """
    try:
        positions = api.list_positions()
        return {p.symbol: float(p.market_value) for p in positions}
    except Exception as e:
        print(f"Could not fetch positions: {e}")
        return {}


# calculates what to buy and sell to match the target weights
def calculate_orders(
    target_weights: dict[str, float],
    current_positions: dict[str, float],
    portfolio_value: float,
) -> dict[str, float]:
    """
    Compare current positions to target weights and return orders needed.

    Parameters
    ----------
    target_weights      : {ticker: weight} from risk.apply_risk_rules()
    current_positions   : {ticker: market_value} from Alpaca
    portfolio_value     : total account value in dollars

    Returns
    -------
    Dict of {ticker: dollar_amount} — positive = buy, negative = sell.
    """
    orders = {}

    # calculate the target dollar value for each ticker
    target_dollars = {t: w * portfolio_value for t, w in target_weights.items()}

    # tickers we need to sell (currently held but not in new targets)
    for ticker, value in current_positions.items():
        if ticker not in target_dollars:
            orders[ticker] = -value  # sell the entire position

    # tickers we need to buy or rebalance
    for ticker, target in target_dollars.items():
        current = current_positions.get(ticker, 0.0)
        diff = target - current
        # only trade if the difference is more than $10 to avoid tiny orders
        if abs(diff) > 10:
            orders[ticker] = diff

    return orders


# submits a single market order to Alpaca
def submit_order(api, ticker: str, dollar_amount: float) -> None:
    """
    Submit a market order for a given dollar amount.
    Positive dollar_amount = buy, negative = sell.
    """
    side = "buy" if dollar_amount > 0 else "sell"

    try:
        # notional = dollar amount (Alpaca supports fractional shares)
        api.submit_order(
            symbol=ticker,
            notional=round(abs(dollar_amount), 2),
            side=side,
            type="market",
            time_in_force="day",
        )
        print(f"  Order submitted: {side.upper()} ${abs(dollar_amount):,.2f} of {ticker}")
    except Exception as e:
        print(f"  Order FAILED for {ticker}: {e}")


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Momentum Paper Trader  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  DRY RUN: {DRY_RUN}")
    print(f"{'='*50}\n")

    # step 1: get latest prices and momentum picks
    print("Step 1: Loading prices and calculating momentum...")
    try:
        prices      = load_prices(TICKERS, START_DATE)
        latest_date = prices.index[-1]
        picks       = get_top_picks(prices, latest_date, n=3)
        print(f"  Top picks as of {latest_date.date()}: {picks}")
    except Exception as e:
        print(f"Failed to load prices or calculate picks: {e}")
        raise SystemExit

    # step 2: connect to Alpaca
    print("\nStep 2: Connecting to Alpaca...")
    try:
        api = get_alpaca_api()
    except Exception as e:
        print(f"Alpaca connection failed: {e}")
        raise SystemExit

    # step 3: get account value and current positions
    print("\nStep 3: Fetching account and positions...")
    try:
        account           = api.get_account()
        portfolio_value   = float(account.portfolio_value)
        current_positions = get_current_positions(api)
        print(f"  Current positions: {list(current_positions.keys()) or 'None'}")
    except Exception as e:
        print(f"Failed to fetch account data: {e}")
        raise SystemExit

    # step 4: apply risk rules to get target weights
    print("\nStep 4: Applying risk rules...")
    target_weights = apply_risk_rules(
        picks,
        portfolio_value=portfolio_value,
        peak_value=portfolio_value,  # on first run, current value = peak
    )
    print(f"  Target weights: {target_weights}")

    # step 5: calculate what needs to change
    print("\nStep 5: Calculating orders...")
    orders = calculate_orders(target_weights, current_positions, portfolio_value)
    if not orders:
        print("  No orders needed — portfolio already matches targets.")
    else:
        for ticker, amount in orders.items():
            direction = "BUY" if amount > 0 else "SELL"
            print(f"  {direction} ${abs(amount):,.2f} of {ticker}")

    # step 6: submit orders (or just print them if DRY_RUN)
    print(f"\nStep 6: {'Simulating' if DRY_RUN else 'Submitting'} orders...")
    for ticker, amount in orders.items():
        if DRY_RUN:
            direction = "BUY" if amount > 0 else "SELL"
            print(f"  [DRY RUN] Would {direction} ${abs(amount):,.2f} of {ticker}")
        else:
            submit_order(api, ticker, amount)

    # step 7: log this run to CSV so we have a record of every trade decision
    print("\nStep 7: Logging to trade log...")
    log_rows = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if orders:
        for ticker, amount in orders.items():
            log_rows.append({
                "timestamp":   timestamp,
                "ticker":      ticker,
                "direction":   "BUY" if amount > 0 else "SELL",
                "amount_usd":  round(abs(amount), 2),
                "dry_run":     DRY_RUN,
            })
    else:
        # log a "no action" row so we know the script ran even with no trades
        log_rows.append({
            "timestamp":  timestamp,
            "ticker":     "NONE",
            "direction":  "NO_ACTION",
            "amount_usd": 0.0,
            "dry_run":    DRY_RUN,
        })

    log_df = pd.DataFrame(log_rows)

    # append to existing log file, or create it if it doesn't exist yet
    try:
        os.makedirs("results", exist_ok=True)
        write_header = not os.path.exists(TRADE_LOG_PATH)
        log_df.to_csv(TRADE_LOG_PATH, mode="a", header=write_header, index=False)
        print(f"  Logged {len(log_rows)} row(s) to {TRADE_LOG_PATH}")
    except Exception as e:
        print(f"  Could not write trade log: {e}")

    print("\nDone.")
