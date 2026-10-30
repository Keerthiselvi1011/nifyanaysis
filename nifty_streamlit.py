"""
=============================================================
NIFTY 50 – 50 EMA Crossover Backtest  |  Streamlit Dashboard
=============================================================
Run:  streamlit run nifty_streamlit.py
=============================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY 50 EMA Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="metric-container"] label  { color: #8b949e !important; font-size: 13px; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #58a6ff !important; font-size: 22px; font-weight: 700;
    }

    /* Titles */
    h1, h2, h3 { color: #e6edf3 !important; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; border: none; border-radius: 8px;
        padding: 10px 28px; font-size: 15px; font-weight: 600;
        width: 100%; cursor: pointer;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #2ea043, #3fb950); }

    /* Divider */
    hr { border-color: #30363d; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff; }

    /* Info / warning boxes */
    .stAlert { border-radius: 8px; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #161b22; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR – CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Strategy Configuration")
    st.markdown("---")

    ticker = st.selectbox(
        "📌 Index / Symbol",
        ["^NSEI", "^NSEBANK", "^BSESN", "RELIANCE.NS", "TCS.NS", "INFY.NS"],
        index=0,
        help="Yahoo Finance ticker symbol"
    )
    custom_ticker = st.text_input("Or enter custom ticker", placeholder="e.g. HDFCBANK.NS")
    if custom_ticker.strip():
        ticker = custom_ticker.strip().upper()

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    years_back = st.slider("Lookback (years)", 1, 5, 3)
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=years_back * 365 + 60)

    st.markdown("---")
    st.markdown("### 📐 Indicator")
    ema_period = st.slider("EMA Period", 10, 200, 50, step=5)

    st.markdown("---")
    st.markdown("### 💰 Risk Management")
    initial_capital = st.number_input(
        "Initial Capital (₹)", min_value=10_000,
        max_value=100_000_000, value=1_000_000, step=50_000,
        format="%d"
    )
    risk_pct      = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, step=0.5)
    reward_ratio  = st.slider("Reward : Risk Ratio", 1.0, 5.0, 2.0, step=0.5)
    sl_pct        = st.slider("Stop-Loss % from Entry", 0.5, 5.0, 2.0, step=0.5)

    st.markdown("---")
    run_btn = st.button("🚀 Run Backtest")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 10px 0 5px 0;'>
    <h1 style='font-size:2.2rem; margin-bottom:0;'>📈 NIFTY 50 – EMA Crossover Backtest</h1>
    <p style='color:#8b949e; font-size:1rem; margin-top:6px;'>
        Strategy: Buy when Close &gt; EMA | Sell when Close &lt; EMA &nbsp;|&nbsp;
        Risk Mgmt: 1% Capital Risk per Trade &nbsp;|&nbsp; Live Data via yfinance
    </p>
</div>
<hr/>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def download_data(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def run_backtest(df, ema_period, initial_capital, risk_pct, reward_ratio, sl_pct):
    risk_frac = risk_pct / 100
    sl_frac   = sl_pct  / 100

    df = df.copy()
    df["EMA"] = df["Close"].squeeze().ewm(span=ema_period, adjust=False).mean()

    close = df["Close"].squeeze()
    ema   = df["EMA"].squeeze()
    df["Signal"]     = 0
    df.loc[close > ema, "Signal"] = 1
    df.loc[close < ema, "Signal"] = -1
    df["SignalShift"] = df["Signal"].shift(1)
    df["CrossUp"]    = (df["Signal"] == 1)  & (df["SignalShift"] == -1)
    df["CrossDown"]  = (df["Signal"] == -1) & (df["SignalShift"] == 1)

    capital       = initial_capital
    position_open = False
    entry_price   = 0.0
    stop_loss     = 0.0
    take_profit   = 0.0
    shares        = 0
    trades        = []
    equity_curve  = []

    for date, row in df.iterrows():
        c = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])

        if not position_open and row["CrossUp"]:
            risk_amount = capital * risk_frac
            sl_distance = c * sl_frac
            shares      = max(1, int(risk_amount / sl_distance))
            stop_loss   = c - sl_distance
            take_profit = c + sl_distance * reward_ratio
            entry_price = c
            position_open = True
            entry_date  = date

        elif position_open:
            exit_reason = None
            if row["CrossDown"]:      exit_reason = "Signal Exit"
            elif c <= stop_loss:      exit_reason = "Stop Loss"
            elif c >= take_profit:    exit_reason = "Take Profit"

            if exit_reason:
                pnl = (c - entry_price) * shares
                capital += pnl
                trades.append({
                    "Entry Date" : entry_date,
                    "Exit Date"  : date,
                    "Entry Price": round(entry_price, 2),
                    "Exit Price" : round(c, 2),
                    "Shares"     : shares,
                    "Stop Loss"  : round(stop_loss, 2),
                    "Take Profit": round(take_profit, 2),
                    "PnL (₹)"    : round(pnl, 2),
                    "Exit Reason": exit_reason,
                    "Win"        : pnl > 0,
                })
                position_open = False

        equity_curve.append({"Date": date, "Capital": capital})

    if position_open:
        last_close = float(df["Close"].iloc[-1])
        pnl = (last_close - entry_price) * shares
        capital += pnl
        trades.append({
            "Entry Date" : entry_date,
            "Exit Date"  : df.index[-1],
            "Entry Price": round(entry_price, 2),
            "Exit Price" : round(last_close, 2),
            "Shares"     : shares,
            "Stop Loss"  : round(stop_loss, 2),
            "Take Profit": round(take_profit, 2),
            "PnL (₹)"    : round(pnl, 2),
            "Exit Reason": "End of Data",
            "Win"        : pnl > 0,
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    equity_df["Peak"]     = equity_df["Capital"].cummax()
    equity_df["Drawdown"] = (equity_df["Capital"] - equity_df["Peak"]) / equity_df["Peak"] * 100

    return df, trades_df, equity_df, capital


def color_pnl(val):
    color = "#3fb950" if val > 0 else "#f85149"
    return f"color: {color}; font-weight: 600"


# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px;'>
        <div style='font-size: 5rem;'>📊</div>
        <h2 style='color:#58a6ff;'>Configure & Run Your Backtest</h2>
        <p style='color:#8b949e; font-size:1.05rem; max-width:520px; margin:auto;'>
            Adjust the strategy parameters in the sidebar — EMA period, capital, risk
            management, and date range — then click <strong>🚀 Run Backtest</strong>.
        </p>
        <br/>
        <p style='color:#30363d; font-size:0.85rem;'>
            Live market data is fetched in real time from Yahoo Finance via <code>yfinance</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Download ──────────────────────────────────
with st.spinner(f"📥 Downloading {ticker} data…"):
    raw_df = download_data(
        ticker,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

if raw_df is None or raw_df.empty:
    st.error("❌ No data found. Check ticker symbol or internet connection.")
    st.stop()

# Trim to exact window
three_yr_start = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
df_trim = raw_df[raw_df.index >= three_yr_start].copy()

# ── Run backtest ──────────────────────────────
df_bt, trades_df, equity_df, final_capital = run_backtest(
    df_trim, ema_period, initial_capital, risk_pct, reward_ratio, sl_pct
)

# ── Metrics ───────────────────────────────────
total_trades  = len(trades_df)
wins          = int(trades_df["Win"].sum()) if total_trades > 0 else 0
losses        = total_trades - wins
win_rate      = (wins / total_trades * 100) if total_trades > 0 else 0
total_pnl     = trades_df["PnL (₹)"].sum() if total_trades > 0 else 0
pnl_pct       = (final_capital - initial_capital) / initial_capital * 100
max_drawdown  = equity_df["Drawdown"].min()
avg_win       = trades_df.loc[trades_df["Win"],  "PnL (₹)"].mean() if wins   > 0 else 0
avg_loss      = trades_df.loc[~trades_df["Win"], "PnL (₹)"].mean() if losses > 0 else 0
profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.markdown("### 📊 Performance Summary")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

pnl_delta_color = "normal" if total_pnl >= 0 else "inverse"

with c1: st.metric("Initial Capital",  f"₹{initial_capital:,.0f}")
with c2: st.metric("Final Capital",    f"₹{final_capital:,.0f}",   delta=f"{pnl_pct:+.2f}%")
with c3: st.metric("Total P&L",        f"₹{total_pnl:,.0f}",       delta=f"{pnl_pct:+.2f}%")
with c4: st.metric("Total Trades",     f"{total_trades}")
with c5: st.metric("Win Rate",         f"{win_rate:.1f}%",          delta=f"W:{wins}  L:{losses}")
with c6: st.metric("Max Drawdown",     f"{max_drawdown:.2f}%")
with c7: st.metric("Profit Factor",    f"{profit_factor:.2f}")

st.markdown("<hr/>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Chart", "💹 Equity & Drawdown", "📋 Trade Log", "📉 Trade Analytics"
])


# ── TAB 1 : Price + EMA + Signals ─────────────
with tab1:
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_bt.index,
        open=df_bt["Open"].squeeze(),
        high=df_bt["High"].squeeze(),
        low=df_bt["Low"].squeeze(),
        close=df_bt["Close"].squeeze(),
        name="OHLC",
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
        increasing_fillcolor="#1c2e1c",
        decreasing_fillcolor="#2e1c1c",
    ))

    # EMA line
    fig.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["EMA"],
        name=f"{ema_period} EMA",
        line=dict(color="#f0883e", width=2, dash="dash"),
    ))

    # Buy signals
    if total_trades > 0:
        fig.add_trace(go.Scatter(
            x=trades_df["Entry Date"],
            y=trades_df["Entry Price"],
            mode="markers",
            name="BUY",
            marker=dict(symbol="triangle-up", size=12, color="#3fb950",
                        line=dict(color="#ffffff", width=1)),
        ))
        fig.add_trace(go.Scatter(
            x=trades_df["Exit Date"],
            y=trades_df["Exit Price"],
            mode="markers",
            name="SELL / EXIT",
            marker=dict(symbol="triangle-down", size=12, color="#f85149",
                        line=dict(color="#ffffff", width=1)),
        ))

    fig.update_layout(
        title=dict(text=f"<b>{ticker}  —  {ema_period} EMA Crossover Strategy</b>",
                   font=dict(size=16, color="#e6edf3")),
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        xaxis=dict(gridcolor="#21262d", showgrid=True, rangeslider=dict(visible=False),
                   type="date"),
        yaxis=dict(gridcolor="#21262d", showgrid=True, title="Price (₹)"),
        legend=dict(bgcolor="#21262d", bordercolor="#30363d", borderwidth=1,
                    font=dict(color="#c9d1d9")),
        hovermode="x unified",
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.info(f"📅 **Data range:** {df_bt.index[0].date()} → {df_bt.index[-1].date()}  "
                f"({len(df_bt)} trading days)")
    with col_r:
        last_price = float(df_bt["Close"].iloc[-1])
        last_ema   = float(df_bt["EMA"].iloc[-1])
        signal_txt = "🟢 ABOVE EMA (Bullish)" if last_price > last_ema else "🔴 BELOW EMA (Bearish)"
        st.info(f"🏷️ **Last Close:** ₹{last_price:,.2f}  |  **EMA{ema_period}:** ₹{last_ema:,.2f}  |  **Signal:** {signal_txt}")


# ── TAB 2 : Equity Curve + Drawdown ──────────
with tab2:
    fig2 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Equity Curve", "Drawdown (%)"),
    )

    # Equity
    fig2.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["Capital"],
        name="Portfolio Value",
        line=dict(color="#7ee787", width=2),
        fill="tozeroy",
        fillcolor="rgba(126,231,135,0.05)",
    ), row=1, col=1)

    fig2.add_hline(y=initial_capital, line_dash="dot",
                   line_color="#8b949e", annotation_text="Initial Capital",
                   annotation_font_color="#8b949e", row=1, col=1)

    # Drawdown
    fig2.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["Drawdown"],
        name="Drawdown",
        line=dict(color="#f85149", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.18)",
    ), row=2, col=1)

    fig2.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        height=540,
        showlegend=True,
        legend=dict(bgcolor="#21262d", bordercolor="#30363d"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in [1, 2]:
        fig2.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig2.update_yaxes(gridcolor="#21262d", row=i, col=1)

    fig2.update_yaxes(tickprefix="₹", tickformat=",.0f", row=1, col=1)
    fig2.update_yaxes(ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig2, use_container_width=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.metric("Peak Capital",   f"₹{equity_df['Capital'].max():,.0f}")
    with mc2: st.metric("Trough Capital", f"₹{equity_df['Capital'].min():,.0f}")
    with mc3: st.metric("Max Drawdown",   f"{max_drawdown:.2f}%")
    with mc4:
        recovery = equity_df[equity_df["Capital"] >= initial_capital]
        if len(recovery) > 0 and final_capital >= initial_capital:
            st.metric("End Status", "✅ In Profit")
        else:
            st.metric("End Status", "⚠️ Below Initial")


# ── TAB 3 : Trade Log ─────────────────────────
with tab3:
    if total_trades == 0:
        st.warning("No trades were generated for this configuration.")
    else:
        st.markdown(f"**{total_trades} trades executed** — {wins} wins 🟢  |  {losses} losses 🔴")

        display_df = trades_df.copy()
        display_df["Entry Date"] = display_df["Entry Date"].astype(str).str[:10]
        display_df["Exit Date"]  = display_df["Exit Date"].astype(str).str[:10]
        display_df["Result"]     = display_df["Win"].map({True: "✅ Win", False: "❌ Loss"})

        cols_show = ["Entry Date", "Exit Date", "Entry Price", "Exit Price",
                     "Shares", "Stop Loss", "Take Profit", "PnL (₹)", "Exit Reason", "Result"]

        styled = (
            display_df[cols_show]
            .style
            .map(color_pnl, subset=["PnL (₹)"])
            .format({
                "Entry Price": "₹{:,.2f}",
                "Exit Price" : "₹{:,.2f}",
                "Stop Loss"  : "₹{:,.2f}",
                "Take Profit": "₹{:,.2f}",
                "PnL (₹)"   : "₹{:,.2f}",
            })
            .set_properties(**{"background-color": "#161b22", "color": "#c9d1d9",
                               "border-color": "#30363d"})
        )
        st.dataframe(styled, use_container_width=True, height=480)

        # Download CSV
        csv = display_df[cols_show].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Trade Log as CSV",
            data=csv,
            file_name=f"{ticker}_trade_log.csv",
            mime="text/csv",
        )


# ── TAB 4 : Analytics ─────────────────────────
with tab4:
    if total_trades == 0:
        st.warning("No trades to analyse.")
    else:
        col_a, col_b = st.columns(2)

        # Win/Loss bar
        with col_a:
            fig_wl = go.Figure(go.Bar(
                x=["Wins", "Losses"],
                y=[wins, losses],
                marker_color=["#3fb950", "#f85149"],
                text=[wins, losses],
                textposition="outside",
                textfont=dict(color="white"),
            ))
            fig_wl.update_layout(
                title="Win vs Loss Count",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9"),
                yaxis=dict(gridcolor="#21262d"),
                height=320, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_wl, use_container_width=True)

        # Exit reason pie
        with col_b:
            reason_counts = trades_df["Exit Reason"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=reason_counts.index,
                values=reason_counts.values,
                marker=dict(colors=["#58a6ff", "#3fb950", "#f85149", "#f0883e"]),
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(color="white"),
            ))
            fig_pie.update_layout(
                title="Exit Reason Breakdown",
                paper_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"),
                height=320, margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(bgcolor="#21262d"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # PnL per trade
        fig_pnl = go.Figure(go.Bar(
            x=list(range(1, total_trades + 1)),
            y=trades_df["PnL (₹)"].tolist(),
            marker_color=["#3fb950" if w else "#f85149" for w in trades_df["Win"]],
            hovertext=[
                f"Trade #{i+1}<br>Entry: {row['Entry Date']}<br>"
                f"Exit: {row['Exit Date']}<br>PnL: ₹{row['PnL (₹)']:,.2f}<br>{row['Exit Reason']}"
                for i, (_, row) in enumerate(trades_df.iterrows())
            ],
            hoverinfo="text",
        ))
        fig_pnl.add_hline(y=0, line_color="#8b949e", line_dash="dot")
        fig_pnl.update_layout(
            title="P&L per Trade",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#21262d", title="Trade #"),
            yaxis=dict(gridcolor="#21262d", title="PnL (₹)", tickprefix="₹"),
            height=340, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        # Cumulative PnL line
        cum_pnl = trades_df["PnL (₹)"].cumsum()
        fig_cum = go.Figure(go.Scatter(
            x=list(range(1, total_trades + 1)),
            y=cum_pnl.tolist(),
            mode="lines+markers",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
        ))
        fig_cum.add_hline(y=0, line_color="#8b949e", line_dash="dot")
        fig_cum.update_layout(
            title="Cumulative P&L",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#21262d", title="Trade #"),
            yaxis=dict(gridcolor="#21262d", title="Cumulative PnL (₹)", tickprefix="₹"),
            height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Stats table
        st.markdown("#### 📌 Detailed Stats")
        stat_data = {
            "Metric": [
                "Total Trades", "Winning Trades", "Losing Trades", "Win Rate",
                "Avg Win (₹)", "Avg Loss (₹)", "Profit Factor",
                "Best Trade (₹)", "Worst Trade (₹)",
                "Total P&L (₹)", "Return on Capital",
            ],
            "Value": [
                total_trades, wins, losses, f"{win_rate:.1f}%",
                f"₹{avg_win:,.2f}", f"₹{avg_loss:,.2f}", f"{profit_factor:.2f}",
                f"₹{trades_df['PnL (₹)'].max():,.2f}",
                f"₹{trades_df['PnL (₹)'].min():,.2f}",
                f"₹{total_pnl:,.2f}", f"{pnl_pct:.2f}%",
            ]
        }
        st.dataframe(pd.DataFrame(stat_data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr/>
<div style='text-align:center; color:#484f58; font-size:0.82rem; padding: 10px 0;'>
    NIFTY 50 EMA Backtest Dashboard &nbsp;|&nbsp; YRS Innovations LLP &nbsp;|&nbsp;
    Data: Yahoo Finance (yfinance) &nbsp;|&nbsp; ⚠️ For educational purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)