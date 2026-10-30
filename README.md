# NIFTY 50 – EMA Crossover Backtest

An interactive Streamlit dashboard that backtests a simple EMA crossover strategy on NIFTY 50, Bank Nifty, Sensex, or any Yahoo Finance ticker.

## Features
- Buy when Close > EMA, sell when Close < EMA
- Configurable EMA period, lookback, stop-loss, reward:risk ratio and risk per trade
- Position sizing based on capital at risk
- Equity curve and drawdown charts
- Trade log with CSV download
- Analytics: win rate, profit factor, exit reason breakdown, cumulative P&L

## Installation
```bash
pip install streamlit yfinance pandas numpy plotly
```

## Run
```bash
python -m streamlit run nifty_streamlit.py
```

## Disclaimer
For educational purposes only. Not financial advice. Past performance does not guarantee future results.