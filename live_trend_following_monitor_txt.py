#!/usr/bin/env python3
"""
live_trend_following_monitor_txt.py

Runs once per execution (GitHub Actions friendly)
Logs entries/exits to TXT file
Generates a daily summary automatically
Stores structured trade history in:
- live_trades.csv
- live_trades.json
"""

import ccxt
import pandas as pd
import numpy as np
import os
import json
import csv
from datetime import datetime, timezone, timedelta

# ===============================================================
# CONFIGURATION
# ===============================================================
LOG_FILE = "live_trades_log.txt"
SUMMARY_FILE = "last_summary_date.txt"

CSV_FILE = "live_trades.csv"
JSON_FILE = "live_trades.json"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT", "AVAX/USDT", "MATIC/USDT",
    "DOT/USDT", "LTC/USDT", "DOGE/USDT"
]

TIMEFRAME = "4h"
LOOKBACK_LIMIT = 400

# indicators
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
DONCHIAN_PERIOD = 20
VOL_PERIOD = 20
ATR_PERIOD = 14

# strategy rules
RSI_ENTRY = 55
RSI_EXIT = 50
VOL_MULTIPLIER = 1.2
STOP_ATR_MULT = 3
TARGET_RR = 2

# open trades stored in memory during a single run
OPEN_TRADES = {}

# ===============================================================
# FILE LOGGING
# ===============================================================
def log(msg):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ===============================================================
# CSV + JSON STORAGE
# ===============================================================
def save_trade_to_csv(trade_record):
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "side",
                "entry_price", "exit_price", "pnl", "reason"
            ])

        writer.writerow([
            trade_record["timestamp"],
            trade_record["symbol"],
            trade_record["side"],
            trade_record.get("entry_price", None),
            trade_record.get("exit_price", None),
            trade_record.get("pnl", None),
            trade_record.get("reason", None)
        ])


def save_trade_to_json(trade_record):
    data = []

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []

    data.append(trade_record)

    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ===============================================================
# DAILY SUMMARY GENERATOR
# ===============================================================
def generate_daily_summary():
    today = datetime.utcnow().date()

    # avoid duplicate summary
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r") as f:
            last = f.read().strip()
            if last == str(today):
                return

    if not os.path.exists(LOG_FILE):
        return

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    yesterday = today - timedelta(days=1)
    ystr = str(yesterday)

    entries, exits, pnls = [], [], []

    for line in lines:
        if ystr not in line:
            continue

        if "ENTRY" in line:
            entries.append(line)
        elif "EXIT" in line:
            exits.append(line)
            try:
                p = float(line.split("PnL=")[1].split("%")[0])
                pnls.append(p)
            except:
                pass

    if not entries and not exits:
        summary = (
            f"\n===== DAILY SUMMARY ({ystr}) =====\n"
            f"No trades yesterday.\n"
            f"=================================\n\n"
        )
    else:
        summary = (
            f"\n===== DAILY SUMMARY ({ystr}) =====\n"
            f"Total trades: {len(exits)}\n"
            f"Wins: {len([p for p in pnls if p > 0])}\n"
            f"Losses: {len([p for p in pnls if p <= 0])}\n"
        )

        if pnls:
            summary += (
                f"Win rate: {len([p for p in pnls if p > 0]) / len(pnls) * 100:.2f}%\n"
                f"Avg PnL: {np.mean(pnls):.2f}%\n"
                f"Best trade: {max(pnls):.2f}%\n"
                f"Worst trade: {min(pnls):.2f}%\n"
                f"Total PnL: {sum(pnls):.2f}%\n"
            )

        summary += "=================================\n\n"

    with open(LOG_FILE, "a") as f:
        f.write(summary)

    with open(SUMMARY_FILE, "w") as f:
        f.write(str(today))

    print(summary)

# ===============================================================
# INDICATORS
# ===============================================================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
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
    return tr.rolling(period).mean()

# ===============================================================
# DATA FETCH
# ===============================================================
def fetch_latest(exchange, symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LOOKBACK_LIMIT)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
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

    cond = (
        row["close"] > row["ema200"] and
        row["ema50"] > row["ema200"] and
        row["close"] > row["donchian_high"] and
        row["rsi"] > RSI_ENTRY and
        row["volume"] > VOL_MULTIPLIER * row["vol20"]
    )

    if cond:
        entry = row["close"]
        stop = entry - row["atr"] * STOP_ATR_MULT
        target = entry + TARGET_RR * (entry - stop)

        return {
            "entry_time": datetime.utcnow().isoformat(),
            "entry_price": entry,
            "stop_loss": stop,
            "target_price": target,
            "rsi_entry": row["rsi"],
            "ema50": row["ema50"],
            "ema200": row["ema200"],
            "atr_entry": row["atr"]
        }

    return None


def check_exit_conditions(df, trade):
    row = df.iloc[-1]
    price = row["close"]

    if price <= trade["stop_loss"]:
        return "Stop_loss", (price - trade["entry_price"]) / trade["entry_price"] * 100, price

    if price >= trade["target_price"]:
        return "Target_hit", (price - trade["entry_price"]) / trade["entry_price"] * 100, price

    if price < row["ema50"]:
        return "EMA50_exit", (price - trade["entry_price"]) / trade["entry_price"] * 100, price

    if row["rsi"] < RSI_EXIT:
        return "RSI_exit", (price - trade["entry_price"]) / trade["entry_price"] * 100, price

    return None, None, None

# ===============================================================
# MAIN (single run for GitHub Actions)
# ===============================================================
def live_monitor_once():
    exchange = ccxt.binanceus({"enableRateLimit": True})

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

                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "side": "EXIT",
                    "entry_price": trade["entry_price"],
                    "exit_price": price,
                    "pnl": pnl,
                    "reason": reason
                }

                save_trade_to_csv(record)
                save_trade_to_json(record)

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

            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": "ENTRY",
                "entry_price": signal["entry_price"]
            }

            save_trade_to_csv(record)
            save_trade_to_json(record)

    log("Cycle complete.\n")

# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    live_monitor_once()

