#!/usr/bin/env python3
"""
live_trend_following_monitor_txt.py

Runs once per execution (GitHub Actions friendly)
Logs entries/exits to TXT file
Generates a daily summary automatically
"""

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import os

# ===============================================================
# CONFIGURATION
# ===============================================================
LOG_FILE = "live_trades_log.txt"
SUMMARY_FILE = "last_summary_date.txt"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT", "AVAX/USDT", "MATIC/USDT",
    "DOT/USDT", "LTC/USDT", "DOGE/USDT"
]

TIMEFRAME = "4h"
LOOKBACK_LIMIT = 400

EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
DONCHIAN_PERIOD = 20
VOL_PERIOD = 20
ATR_PERIOD = 14

RSI_ENTRY = 55
RSI_EXIT = 50
VOL_MULTIPLIER = 1.2
STOP_ATR_MULT = 3
TARGET_RR = 2

# Open trades storage (persisted inside log only)
OPEN_TRADES = {}

# ===============================================================
# LOGGING
# ===============================================================
def log(msg):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ===============================================================
# SUMMARY GENERATION
# ===============================================================
def generate_daily_summary():
    """Generate summary for yesterday's trades. Runs once per day."""
    today = datetime.utcnow().date()

    # prevent duplicate summaries
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r") as f:
            last_date = f.read().strip()
            if last_date == str(today):
                return  # Already generated today

    if not os.path.exists(LOG_FILE):
        return

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    yesterday = today - timedelta(days=1)
    ystr = str(yesterday)

    entries = []
    exits = []
    pnls = []

    for line in lines:
        if ystr not in line:
            continue

        if "ENTRY" in line:
            entries.append(line)
        elif "EXIT" in line:
            exits.append(line)
            try:
                # Parse PnL=2.45%
                p = float(line.split("PnL=")[1].split("%")[0])
                pnls.append(p)
            except:
                pass

    # Build summary text
    if not entries and not exits:
        summary_text = f"\n===== DAILY SUMMARY ({ystr}) =====\nNo trades yesterday.\n=================================\n\n"
    else:
        summary_text = (
            f"\n===== DAILY SUMMARY ({ystr}) =====\n"
            f"Total trades: {len(exits)}\n"
            f"Wins: {len([p for p in pnls if p > 0])}\n"
            f"Losses: {len([p for p in pnls if p <= 0])}\n"
        )

        if pnls:
            summary_text += (
                f"Win rate: {len([p for p in pnls if p > 0]) / len(pnls) * 100:.2f}%\n"
                f"Avg PnL: {np.mean(pnls):.2f}%\n"
                f"Best trade: {max(pnls):.2f}%\n"
                f"Worst trade: {min(pnls):.2f}%\n"
                f"Total PnL: {sum(pnls):.2f}%\n"
            )

        summary_text += "=================================\n\n"

    # Append summary
    with open(LOG_FILE, "a") as f:
        f.write(summary_text)

    # Mark as generated
    with open(SUMMARY_FILE, "w") as f:
        f.write(str(today))

    print(summary_text)

# ===============================================================
# INDICATORS
# ===============================================================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ===============================================================
# DATA FETCH
# ===============================================================
def fetch_latest(exchange, symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LOOKBACK_LIMIT)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df

def add_indicators(df):
    df["ema50"] = ema(df["close"], EMA_FAST)
    df["ema200"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["donchian_high"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["vol20"] = df["volume"].rolling(VOL_PERIOD).mean().shift(1)
    df["atr"] = atr(df, ATR_PERIOD)
    return df.dropna()

# ===============================================================
# STRATEGY LOGIC
# ===============================================================
def check_new_signal(df):
    row = df.iloc[-1]

    conditions = (
        row["close"] > row["ema200"] and
        row["ema50"] > row["ema200"] and
        row["close"] > row["donchian_high"] and
        row["rsi"] > RSI_ENTRY and
        row["volume"] > VOL_MULTIPLIER * row["vol20"]
    )

    if conditions:
        entry_price = row["close"]
        stop = entry_price - row["atr"] * STOP_ATR_MULT
        target = entry_price + TARGET_RR * (entry_price - stop)

        return {
            "entry_price": entry_price,
            "stop_loss": stop,
            "target_price": target,
            "rsi_entry": row["rsi"],
            "ema50": row["ema50"],
            "ema200": row["ema200"],
            "atr_entry": row["atr"],
            "entry_time": datetime.utcnow()
        }
    return None

def check_exit_conditions(df, trade):
    row = df.iloc[-1]
    price = row["close"]
    reason = None

    if price <= trade["stop_loss"]:
        reason = "Stop_loss"
    elif price >= trade["target_price"]:
        reason = "Target_hit"
    elif price < row["ema50"]:
        reason = "EMA50_exit"
    elif row["rsi"] < RSI_EXIT:
        reason = "RSI_exit"

    if reason:
        pnl = (price - trade["entry_price"]) / trade["entry_price"] * 100
        return reason, pnl, price
    return None, None, None

# ===============================================================
# MAIN EXECUTION — SINGLE CYCLE (REQUIRED FOR GITHUB ACTIONS)
# ===============================================================
def live_monitor_once():
    exchange = ccxt.binance({"enableRateLimit": True})

    # daily summary generation
    generate_daily_summary()

    log("🔁 Running 4-hour cycle...")

    for symbol in SYMBOLS:
        df = fetch_latest(exchange, symbol)
        df = add_indicators(df)

        # EXIT CHECK
        if symbol in OPEN_TRADES:
            trade = OPEN_TRADES[symbol]
            reason, pnl, price = check_exit_conditions(df, trade)
            if reason:
                log(f"EXIT {symbol} | {reason} | Exit={price:.2f} | PnL={pnl:.2f}%")
                del OPEN_TRADES[symbol]
            continue

        # ENTRY CHECK
        signal = check_new_signal(df)
        if signal:
            OPEN_TRADES[symbol] = signal
            log(
                f"ENTRY {symbol} | Price={signal['entry_price']:.2f} | "
                f"Stop={signal['stop_loss']:.2f} | Target={signal['target_price']:.2f}"
            )

    log("Cycle complete.\n")

# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    live_monitor_once()
