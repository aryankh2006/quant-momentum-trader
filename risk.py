import pandas as pd

# hardcoded sector for each ticker in our universe
TICKER_SECTORS = {
    "SPY":  "ETF",
    "QQQ":  "ETF",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Technology",
    "GOOG": "Technology",
}

MAX_WEIGHT    = 0.40   # no single stock can be more than 40% of the portfolio
CASH_BUFFER   = 0.10   # always keep 10% in cash, so max invested = 90%
MAX_DRAWDOWN  = 0.15   # if portfolio drops 15% from its peak, go to cash


# applies position size limits and cash buffer to a set of picks,
# returns a dict of {ticker: weight} that sums to at most 0.90
def apply_risk_rules(
    picks: list[str],
    portfolio_value: float,
    peak_value: float,
) -> dict[str, float]:
    """
    Take a list of ticker picks and return safe position weights.

    Rules applied in order:
      1. Drawdown kill switch: if down >15% from peak, return {} (all cash)
      2. Cap each position at 40% max weight
      3. Keep 10% cash buffer (max total invested = 90%)

    Parameters
    ----------
    picks           : list of tickers to invest in
    portfolio_value : current total portfolio value in dollars
    peak_value      : highest portfolio value ever reached (for drawdown calc)

    Returns
    -------
    Dict of {ticker: weight} where weights sum to <= 0.90.
    Empty dict means go to cash entirely.
    """
    if not picks:
        return {}

    # rule 1: drawdown kill switch
    # calculate how far we've fallen from our all-time high
    drawdown = (peak_value - portfolio_value) / peak_value
    if drawdown >= MAX_DRAWDOWN:
        print(f"Kill switch triggered: drawdown is {drawdown:.1%} (limit {MAX_DRAWDOWN:.0%})")
        return {}

    # start with equal weight across all picks
    raw_weight = 1.0 / len(picks)

    weights = {}
    for ticker in picks:
        # rule 2: cap each position at 40%
        weights[ticker] = min(raw_weight, MAX_WEIGHT)

    # rule 3: apply cash buffer - scale all weights so they sum to 90% max
    total_weight = sum(weights.values())
    max_invested = 1.0 - CASH_BUFFER  # 0.90

    if total_weight > max_invested:
        # scale down proportionally so total = 90%
        scale = max_invested / total_weight
        weights = {t: w * scale for t, w in weights.items()}

    return weights


# checks whether 2 or more picks are from the same sector and warns the user
def check_sector_concentration(tickers: list[str]) -> None:
    """
    Print a warning if two or more picks are from the same sector.
    Sector concentration means the portfolio isn't as diversified as it looks.
    """
    sector_counts: dict[str, list[str]] = {}

    for ticker in tickers:
        sector = TICKER_SECTORS.get(ticker, "Unknown")
        if sector not in sector_counts:
            sector_counts[sector] = []
        sector_counts[sector].append(ticker)

    for sector, members in sector_counts.items():
        if len(members) >= 2:
            print(f"Warning: sector concentration in {sector}: {members}")


if __name__ == "__main__":
    # quick test to make sure the rules are working
    test_picks = ["AAPL", "MSFT", "NVDA"]

    print("--- Normal scenario ---")
    weights = apply_risk_rules(test_picks, portfolio_value=10_000, peak_value=10_000)
    print(f"Weights: {weights}")
    print(f"Total invested: {sum(weights.values()):.0%}")
    check_sector_concentration(test_picks)

    print("\n--- Kill switch scenario (down 20%) ---")
    weights = apply_risk_rules(test_picks, portfolio_value=8_000, peak_value=10_000)
    print(f"Weights: {weights}")
