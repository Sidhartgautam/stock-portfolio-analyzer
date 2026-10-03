# ASX Portfolio Risk Analyzer

A small Python project that analyses the risk and return of a portfolio of Australian (ASX) stocks, with an interactive Streamlit dashboard.

I built this to learn portfolio analysis hands-on, working through each technique step by step rather than just copying code.

**Live app:** _add your Streamlit link here after deployment_

## What it does

- **Overview:** pulls the latest available prices for each stock and calculates portfolio value and weights
- **Risk and return:** volatility, correlation between stocks, and Sharpe ratio
- **Efficient Frontier:** finds the mix of stocks that gives the best return for a given level of risk
- **Monte Carlo simulation:** simulates thousands of possible future paths for the portfolio

> Status: in progress. The overview is built first, and the other sections are being added one at a time.

## Stocks analysed

| Ticker | Company / fund |
|---|---|
| CBA.AX | Commonwealth Bank of Australia |
| BHP.AX | BHP Group |
| CSL.AX | CSL Limited |
| WES.AX | Wesfarmers |
| WOW.AX | Woolworths Group |
| VAS.AX | Vanguard Australian Shares Index ETF |

## How the data works

- Historical daily prices (from 2023) are downloaded with `yfinance` and used for correlation, the efficient frontier and Monte Carlo.
- The latest price is fetched fresh on every refresh and used for the overview.
- Prices from Yahoo Finance can be delayed by about 15 minutes. Outside market hours, the last close is shown.

## Tech used

Python, pandas, NumPy, yfinance, Matplotlib, Streamlit

## Run it locally

```bash
git clone https://github.com/Sidhartgautam/stock-portfolio-analyzer
cd asx-portfolio-risk-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## Project files

- `app.py`: the Streamlit dashboard
- `portfolio_analysis.ipynb`: the notebook where each technique was worked out step by step
- `requirements.txt`: Python packages needed to run the app

## Disclaimer

This is a learning project, not financial advice. Past performance does not predict future results.

## Author

Siddhu, Bachelor of Software Engineering, currently studying a Graduate Diploma in Data Science (progressing into the Master of Data Science) in Sydney.