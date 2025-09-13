"""
AI-Powered Profit-Seeking Trading Bot Simulation
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import talib

BINANCE_REST_BASE = "https://api.binance.com"

st.set_page_config(page_title="AI Profit-Seeking Trading Bot", layout="wide")

# ------------------------- Data Fetch -------------------------

def fetch_klines(symbol: str, interval="1m", limit=1000):
    url = f"{BINANCE_REST_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df

# ------------------------- Feature Engineering -------------------------

def add_features(df):
    df["returns"] = df["close"].pct_change().fillna(0)

    # Technical Indicators
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    macd, macdsig, macdhist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd
    df["macdsig"] = macdsig
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    df["volume_change"] = df["volume"].pct_change().fillna(0)

    # Label (1 if next return > 0, else 0)
    df["target"] = (df["returns"].shift(-1) > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df

# ------------------------- Machine Learning -------------------------

def train_ai_model(df):
    features = ["returns", "rsi", "macd", "macdsig", "bb_upper", "bb_middle", "bb_lower", "volume_change"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("lr", LogisticRegression(max_iter=1000))
    ]
    ensemble = VotingClassifier(estimators=models, voting="soft")
    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return ensemble, acc, X_test, y_test, preds

# ------------------------- Trading Simulation -------------------------

if "balance" not in st.session_state:
    st.session_state.balance = 1000.0
if "position" not in st.session_state:
    st.session_state.position = None
if "trade_log" not in st.session_state:
    st.session_state.trade_log = []

profit_target = st.slider("Profit Target (%)", 0.5, 10.0, 1.0) / 100
stop_loss = st.slider("Stop Loss (%)", 0.5, 10.0, 1.0) / 100
trade_size = st.slider("Trade Size (USDT)", 10, 500, 100)

def open_trade(side, price):
    st.session_state.position = {"side": side, "entry": price, "size": trade_size}
    st.session_state.balance -= trade_size
    st.session_state.trade_log.append(f"OPEN {side} at {price:.2f}")

def close_trade(price):
    pos = st.session_state.position
    if pos["side"] == "LONG":
        pnl = (price - pos["entry"]) / pos["entry"]
    else:
        pnl = (pos["entry"] - price) / pos["entry"]
    st.session_state.balance += pos["size"] * (1 + pnl)
    st.session_state.trade_log.append(f"CLOSE {pos['side']} at {price:.2f}, PnL: {pnl*100:.2f}%")
    st.session_state.position = None

def check_position(price):
    if st.session_state.position is None:
        return
    pos = st.session_state.position
    if pos["side"] == "LONG":
        pnl = (price - pos["entry"]) / pos["entry"]
    else:
        pnl = (pos["entry"] - price) / pos["entry"]
    if pnl >= profit_target or pnl <= -stop_loss:
        close_trade(price)

# ------------------------- Streamlit UI -------------------------

st.title("🤖 AI-Powered Profit-Seeking Trading Bot Simulation")

symbol = st.text_input("Symbol (Binance)", value="BTCUSDT")
interval = st.selectbox("Interval", ["1m","3m","5m"], index=0)

st_autorefresh(interval=2000, key="refresh")

try:
    df = fetch_klines(symbol, interval, 1000)
    df = add_features(df)
    current_price = df["close"].values[-1]

    st.subheader("💰 Current Price")
    st.metric(label=f"{symbol} Price", value=f"{current_price:.4f} USDT")

    # Train AI
    model, acc, X_test, y_test, preds = train_ai_model(df)
    st.subheader("🧠 AI Prediction Module")
    st.write(f"**Ensemble Accuracy (last 200 samples):** {acc*100:.2f}%")

    # Prediction Probabilities
    proba = model.predict_proba([df[["returns","rsi","macd","macdsig","bb_upper","bb_middle","bb_lower","volume_change"]].iloc[-1]])[0]
    st.write(f"Prediction → UP: {proba[1]*100:.2f}%, DOWN: {proba[0]*100:.2f}%")

    # Trade Logic
    if st.session_state.position is None:
        if proba[1] > 0.55:  # 55% confidence UP
            open_trade("LONG", current_price)
        elif proba[0] > 0.55:  # 55% confidence DOWN
            open_trade("SHORT", current_price)
    else:
        check_position(current_price)

    # Account Status
    st.subheader("📦 Account Status")
    st.write(f"Balance: **{st.session_state.balance:.2f} USDT**")
    if st.session_state.position:
        pos = st.session_state.position
        if pos["side"] == "LONG":
            unrealized = (current_price - pos["entry"]) / pos["entry"] * pos["size"]
        else:
            unrealized = (pos["entry"] - current_price) / pos["entry"] * pos["size"]
        st.write(f"Open Position: {pos}")
        st.write(f"Unrealized PnL: {unrealized:.2f} USDT")

    # Trade log
    st.subheader("📑 Trade Log")
    st.write(st.session_state.trade_log[-20:])

    # Charts
    st.subheader("📊 Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["open_time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df["open_time"], y=df["rsi"], name="RSI", yaxis="y2", line=dict(color="blue")))
    fig.update_layout(
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0,100]),
        xaxis=dict(rangeslider=dict(visible=False))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 AI Model Backtest Accuracy")
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(y=(preds == y_test).astype(int), mode="lines", name="Correct (1=Yes, 0=No)"))
    st.plotly_chart(fig_acc, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
