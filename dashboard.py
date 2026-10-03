import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Risk Analyzer", layout="wide")
st.title("📊 ASX Portfolio Risk Analyzer")

tickers = ['BHP.AX', 'CBA.AX', 'CSL.AX', 'VAS.AX', 'WES.AX', 'WOW.AX']
risk_free_rate = 0.04
initial_investment = 10000


# ---------------------------------------------------------------------
# HISTORICAL DATA (cached indefinitely — used for all the statistical analysis)
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    data = yf.download(tickers, start='2023-01-01', end='2026-10-01')
    return data['Close']

close_prices = load_data()
daily_returns = close_prices.pct_change().dropna()

annual_volatility = daily_returns.std() * np.sqrt(252)
annual_return = daily_returns.mean() * 252
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
cov_matrix = daily_returns.cov() * 252

metrics_df = pd.DataFrame({
    'Annual Return': annual_return,
    'Annual Volatility': annual_volatility,
    'Sharpe Ratio': sharpe_ratio
}).sort_values('Sharpe Ratio', ascending=False)


# ---------------------------------------------------------------------
# EFFICIENT FRONTIER SIMULATION (cached — runs once, not on every interaction)
# ---------------------------------------------------------------------
@st.cache_data
def run_frontier_simulation(_mean_returns, _cov_matrix, num_portfolios=10000):
    num_assets = len(tickers)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        port_return = np.sum(weights * _mean_returns)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(_cov_matrix, weights)))
        results[0, i] = port_return
        results[1, i] = port_risk
        results[2, i] = (port_return - risk_free_rate) / port_risk
    return results, weights_record

sim_results, weights_record = run_frontier_simulation(annual_return.values, cov_matrix.values)
max_sharpe_idx = np.argmax(sim_results[2, :])


# ---------------------------------------------------------------------
# MONTE CARLO SIMULATION (cached — depends only on the best portfolio found above)
# ---------------------------------------------------------------------
@st.cache_data
def run_monte_carlo(_best_return, _best_risk, num_simulations=1000, num_days=252):
    daily_mean = _best_return / 252
    daily_std = _best_risk / np.sqrt(252)
    mc_results = np.zeros((num_days, num_simulations))
    for sim in range(num_simulations):
        daily_sim_returns = np.random.normal(daily_mean, daily_std, num_days)
        mc_results[:, sim] = initial_investment * np.cumprod(1 + daily_sim_returns)
    return mc_results

mc_results = run_monte_carlo(sim_results[0, max_sharpe_idx], sim_results[1, max_sharpe_idx])
p5, p50, p95 = np.percentile(mc_results[-1, :], [5, 50, 95])


# ---------------------------------------------------------------------
# LIVE-ish PRICES (short cache TTL — refetches periodically, unlike the data above)
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)  # treat as stale after 5 minutes
def get_latest_prices():
    latest_data = yf.download(tickers, period='5d')['Close']
    return latest_data


# ---------------------------------------------------------------------
# SIDEBAR — interactive controls
# ---------------------------------------------------------------------
st.sidebar.header("Build Your Own Portfolio")
st.sidebar.write("Adjust weights (auto-normalized to 100%)")

user_weights = {}
for ticker in tickers:
    user_weights[ticker] = st.sidebar.slider(ticker, 0, 100, 100 // len(tickers))

total = sum(user_weights.values())
if total == 0:
    st.sidebar.warning("At least one weight must be above 0")
    normalized_weights = None
else:
    normalized_weights = np.array([user_weights[t] / total for t in tickers])


# ---------------------------------------------------------------------
# MAIN AREA — organized into tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💹 Live Prices",
    "📈 Overview",
    "🔗 Correlation",
    "🎯 Efficient Frontier",
    "🎲 Monte Carlo"
])

with tab1:
    st.write("### Current Prices (15-20 min delayed, via Yahoo Finance)")

    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    latest_prices = get_latest_prices()
    current = latest_prices.iloc[-1]
    previous = latest_prices.iloc[-2]

    cols = st.columns(len(tickers))
    for i, ticker in enumerate(tickers):
        change = current[ticker] - previous[ticker]
        pct_change = (change / previous[ticker]) * 100
        cols[i].metric(
            label=ticker,
            value=f"${current[ticker]:.2f}",
            delta=f"{pct_change:+.2f}%"
        )

    st.caption(f"Last updated: {latest_prices.index[-1].strftime('%Y-%m-%d')} • auto-refreshes every 5 min, or click Refresh Now")

with tab2:
    st.write("### Raw Closing Prices")
    st.dataframe(close_prices.tail())

    st.write("### Risk & Return by Stock")
    st.dataframe(
        metrics_df.style
        .format("{:.2%}", subset=['Annual Return', 'Annual Volatility'])
        .format("{:.2f}", subset=['Sharpe Ratio'])
    )

    if normalized_weights is not None:
        user_return = np.sum(normalized_weights * annual_return.values)
        user_risk = np.sqrt(np.dot(normalized_weights.T, np.dot(cov_matrix.values, normalized_weights)))
        user_sharpe = (user_return - risk_free_rate) / user_risk

        st.write("### Your Custom Portfolio (set in the sidebar)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Annual Return", f"{user_return:.2%}")
        c2.metric("Annual Risk (Volatility)", f"{user_risk:.2%}")
        c3.metric("Sharpe Ratio", f"{user_sharpe:.2f}")

with tab3:
    st.write("### Correlation Between Stocks")
    correlation_matrix = daily_returns.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_yticklabels(tickers)
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center')
    fig.colorbar(im, ax=ax, label='Correlation')
    st.pyplot(fig)

with tab4:
    st.write("### 10,000 Simulated Portfolios")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sc = ax2.scatter(sim_results[1, :], sim_results[0, :], c=sim_results[2, :], cmap='viridis', alpha=0.5)
    ax2.scatter(sim_results[1, max_sharpe_idx], sim_results[0, max_sharpe_idx],
                color='red', marker='*', s=400, label='Best Sharpe Ratio')
    ax2.set_xlabel('Risk (Volatility)')
    ax2.set_ylabel('Expected Return')
    ax2.legend()
    fig2.colorbar(sc, ax=ax2, label='Sharpe Ratio')
    st.pyplot(fig2)

    st.write("### Best Portfolio Weights Found")
    best_weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': (weights_record[max_sharpe_idx] * 100).round(2)
    }).sort_values('Weight', ascending=False)
    st.dataframe(best_weights_df)

with tab5:
    st.write("### 1,000 Simulated One-Year Outcomes")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(mc_results, linewidth=0.5, alpha=0.2, color='steelblue')
    ax3.axhline(initial_investment, color='black', linestyle='--', label='Initial Investment')
    ax3.axhline(p5, color='red', label=f'5th percentile: ${p5:,.0f}')
    ax3.axhline(p50, color='green', label=f'Median: ${p50:,.0f}')
    ax3.axhline(p95, color='blue', label=f'95th percentile: ${p95:,.0f}')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.legend()
    st.pyplot(fig3)

    st.success(
        f"Based on the best-Sharpe portfolio: 90% of simulated outcomes fall "
        f"between ${p5:,.0f} and ${p95:,.0f} after 1 year (starting from ${initial_investment:,.0f})."
    )