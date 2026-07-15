import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
import yfinance as yf
import numpy as np

from metrics import calculate_metrics
from strategy import get_top_picks, LOOKBACK_12M, SKIP_1M
from data_loader import load_prices, TICKERS, START_DATE
from risk import apply_risk_rules

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Momentum Trader Dashboard",
    page_icon="📈",
    layout="wide",
)


# loads the backtest CSV and returns it as a DataFrame
# st.cache_data means Streamlit only re-runs this when the file changes,
# not every time the user clicks a button
@st.cache_data
def load_results() -> pd.DataFrame:
    """Load the saved backtest results from CSV."""
    try:
        df = pd.read_csv(
            "results/performance_report.csv",
            index_col="date",
            parse_dates=True,
        )
        return df
    except FileNotFoundError:
        st.error("No results found. Run backtester.py first.")
        st.stop()


# downloads SPY monthly returns for the same date range as the backtest
@st.cache_data
def load_spy(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Download SPY and return monthly returns aligned to the backtest dates."""
    try:
        spy_raw     = yf.download("SPY", start=start, end=end, progress=False)["Close"].squeeze()
        spy_monthly = spy_raw.resample("MS").first()
        return spy_monthly.pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)


# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    n_picks = st.selectbox(
        label="Number of picks",
        options=[2, 3, 4],
        index=1,                # default to 3
        help="How many top momentum stocks to hold each month",
    )

    st.markdown("---")
    st.caption("Charts update instantly when you change these settings.")


# ── load data ─────────────────────────────────────────────────────────────────
# all heavy data loading is wrapped in @st.cache_data functions above,
# so switching the sidebar n_picks dropdown doesn't re-download anything
df          = load_results()
returns     = df["portfolio_value"].pct_change().dropna()
spy_returns = load_spy(df.index[0], df.index[-1])
spy_returns = spy_returns.reindex(returns.index).fillna(0)
metrics     = calculate_metrics(returns, spy_returns)

# ── header ────────────────────────────────────────────────────────────────────
st.title("📈 Momentum Trading Strategy Dashboard")
st.caption("12-1 momentum strategy backtested from 2019 to today | Universe: SPY, QQQ, AAPL, MSFT, NVDA, AMZN, META, GOOG")

st.divider()

# ── metric cards ──────────────────────────────────────────────────────────────
# four side-by-side cards showing the key performance numbers at a glance
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="CAGR",
    value=f"{metrics['CAGR']:.2%}",
    help="Compound Annual Growth Rate — average yearly return",
)
col2.metric(
    label="Sharpe Ratio",
    value=f"{metrics['Sharpe']:.2f}",
    help="Return per unit of risk. Above 1.0 is good, above 2.0 is excellent",
)
col3.metric(
    label="Max Drawdown",
    value=f"{metrics['max_drawdown']:.2%}",
    help="Worst peak-to-trough loss over the entire backtest period",
)
col4.metric(
    label="Win Rate",
    value=f"{metrics['win_rate']:.2%}",
    help="Percentage of months where the strategy made money",
)

st.divider()

# ── equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve — Strategy vs SPY")

# rebuild SPY as a dollar value starting at the same amount as the strategy's
# first month — this ensures both lines start at the same point on the chart
starting_capital = df["portfolio_value"].iloc[0]
spy_curve = (1 + spy_returns).cumprod() * starting_capital

# align SPY curve to only the dates we have strategy data for
spy_curve = spy_curve.reindex(df.index)

fig = go.Figure()

# strategy line
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["portfolio_value"],
    name="Strategy",
    line=dict(color="#2563EB", width=2),
))

# SPY benchmark line
fig.add_trace(go.Scatter(
    x=spy_curve.index,
    y=spy_curve.values,
    name="SPY (Buy & Hold)",
    line=dict(color="#F97316", width=2, dash="dash"),
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",        # shows both lines' values on hover
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    margin=dict(l=0, r=0, t=10, b=0),
)

# format y-axis as dollars
fig.update_yaxes(tickprefix="$", tickformat=",.0f")

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── rolling sharpe chart ───────────────────────────────────────────────────────
st.subheader("Rolling 12-Month Sharpe Ratio")

rolling = metrics["rolling_sharpe"].dropna()

fig_sharpe = go.Figure()

fig_sharpe.add_trace(go.Scatter(
    x=rolling.index,
    y=rolling.values,
    name="Rolling Sharpe",
    line=dict(color="#7C3AED", width=2),
    fill="tozeroy",     # shades the area between the line and zero
    fillcolor="rgba(124, 58, 237, 0.1)",
))

# draw a horizontal line at 1.0 — the "good" threshold
fig_sharpe.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="#16A34A",
    annotation_text="Sharpe = 1.0 (good)",
    annotation_position="bottom right",
)

# draw a horizontal line at 0 — below this we're underperforming risk-free
fig_sharpe.add_hline(
    y=0,
    line_dash="dot",
    line_color="#DC2626",
)

fig_sharpe.update_layout(
    xaxis_title="Date",
    yaxis_title="Sharpe Ratio",
    height=320,
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False,
)

st.plotly_chart(fig_sharpe, use_container_width=True)

st.divider()

# ── monthly returns heatmap ────────────────────────────────────────────────────
st.subheader("Monthly Returns Heatmap")

# build a year × month grid of returns
# each cell = the strategy return for that specific month
monthly_returns = df["portfolio_value"].pct_change().dropna()
monthly_returns.index = pd.to_datetime(monthly_returns.index)

# pivot into a table: rows = years, columns = month numbers (1-12)
heatmap_df = monthly_returns.groupby([
    monthly_returns.index.year,
    monthly_returns.index.month,
]).first().unstack(level=1)

# rename columns from numbers to month abbreviations
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
heatmap_df.columns = [month_names[m - 1] for m in heatmap_df.columns]

# convert to percentages for display
z    = heatmap_df.values * 100
text = np.where(
    np.isnan(z),
    "",
    [[f"{v:.1f}%" for v in row] for row in z],
)

fig_heat = go.Figure(go.Heatmap(
    z=z,
    x=heatmap_df.columns.tolist(),
    y=heatmap_df.index.tolist(),
    text=text,
    texttemplate="%{text}",
    colorscale="RdYlGn",   # red = bad month, green = good month
    zmid=0,                # centre the colour scale at 0%
    showscale=True,
    colorbar=dict(title="Return %"),
))

fig_heat.update_layout(
    xaxis_title="Month",
    yaxis_title="Year",
    height=350,
    margin=dict(l=0, r=0, t=10, b=0),
)

st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── current holdings table ────────────────────────────────────────────────────
st.subheader("Current Holdings (as of today)")

@st.cache_data
def load_prices_cached() -> pd.DataFrame:
    """Load all price data — cached so it only downloads once per session."""
    return load_prices(TICKERS, START_DATE)

try:
    prices      = load_prices_cached()
    latest_date = prices.index[-1]
    picks       = get_top_picks(prices, latest_date, n=n_picks)
    weights     = apply_risk_rules(picks, portfolio_value=1.0, peak_value=1.0)

    # build a table row for each pick showing its momentum score and weight
    rows = []
    for ticker in picks:
        history      = prices.loc[:latest_date, ticker]
        score        = history.iloc[-SKIP_1M] / history.iloc[-LOOKBACK_12M] - 1
        rows.append({
            "Ticker":           ticker,
            "12-1 Momentum":    f"{score:.1%}",
            "Target Weight":    f"{weights.get(ticker, 0):.1%}",
        })

    holdings_df = pd.DataFrame(rows)
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    st.caption(f"Based on prices as of {latest_date.date()} | {n_picks} picks selected")

except Exception as e:
    st.warning(f"Could not load current holdings: {e}")
