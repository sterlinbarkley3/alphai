#!/usr/bin/env python3
"""
label_data_v2.py — Generates rich labeled training data using advanced features
Separate files for crypto and stocks for asset-specific models
Also pulls hourly crypto data for 24x more training samples
"""

import os, csv
import yfinance as yf
import pandas as pd

PROJECT_DIR = "/Users/mythreeboyz/pythonuh/ai trader"
DATA_DIR    = "/Users/mythreeboyz/pythonuh/ai trader/data"

LOOKAHEAD   = 10
MIN_CHANGE  = 0.5   # only label if price moved at least 0.5% — filters noise

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","MATIC","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

CRYPTO_YAHOO = {
    "BTC":"BTC-USD","XRP":"XRP-USD","SOL":"SOL-USD","LINK":"LINK-USD",
    "HBAR":"HBAR-USD","XLM":"XLM-USD","ADA":"ADA-USD","DOT":"DOT-USD",
    "AVAX":"AVAX-USD","MATIC":"MATIC-USD","ATOM":"ATOM-USD"
}

from features import compute_features

def get_signal_from_features(f):
    score = f["score"]
    mom   = f["mom5"]
    rsi   = f["rsi"]
    macd  = f["macd_cross"]

    # Stronger signal logic using all indicators
    buy_signals  = 0
    sell_signals = 0

    if f["ma5_cross"] > 0:   buy_signals  += 1
    else:                     sell_signals += 1

    if mom > 1:               buy_signals  += 1
    elif mom < -1:            sell_signals += 1

    if rsi < 40:              buy_signals  += 1  # oversold
    elif rsi > 60:            sell_signals += 1  # overbought

    if macd > 0:              buy_signals  += 1
    else:                     sell_signals += 1

    if f["bb_position"] < -0.5: buy_signals  += 1  # near lower band
    elif f["bb_position"] > 0.5: sell_signals += 1  # near upper band

    if buy_signals >= 4:   return "BUY"
    elif sell_signals >= 4: return "SELL"
    else:                   return "HOLD"

def load_daily_prices(symbol):
    path = os.path.join(DATA_DIR, f"history_{symbol}.txt")
    prices = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: prices.append(float(line))
                    except: pass
    except FileNotFoundError:
        pass
    return prices

def fetch_hourly_crypto(symbol):
    """Pull 2 years of hourly data from Yahoo Finance"""
    yahoo_sym = CRYPTO_YAHOO.get(symbol)
    if not yahoo_sym:
        return []
    try:
        print(f"    Fetching hourly data for {symbol}...", end="", flush=True)
        ticker = yf.Ticker(yahoo_sym)
        # Yahoo only gives 730 days of hourly — that's still ~17,000 data points
        hist = ticker.history(period="730d", interval="1h")
        if hist.empty:
            print(" no data")
            return []
        prices = hist["Close"].dropna().tolist()
        print(f" {len(prices):,} hourly prices")
        return prices
    except Exception as e:
        print(f" error: {e}")
        return []

def generate_labels(prices, symbol, asset_type, rows, min_prices=50):
    count = 0
    for i in range(min_prices, len(prices) - LOOKAHEAD):
        window        = prices[:i]
        current_price = prices[i - 1]
        future_price  = prices[i + LOOKAHEAD - 1]

        feats = compute_features(window)
        if feats is None:
            continue

        signal = get_signal_from_features(feats)
        if signal == "HOLD":
            continue

        price_change = (future_price - current_price) / current_price * 100

        # Filter out tiny moves — only learn from meaningful signals
        if abs(price_change) < MIN_CHANGE:
            continue

        if signal == "BUY":
            label = 1 if price_change > 0 else 0
        else:
            label = 1 if price_change < 0 else 0

        rows.append({
            "symbol":       symbol,
            "asset_type":   asset_type,
            "price":        round(current_price, 6),
            "future_price": round(future_price,  6),
            "price_change": round(price_change,  4),
            "signal":       signal,
            "label":        label,
            **feats,
        })
        count += 1
    return count

def main():
    print("\n  Generating rich labeled training data...\n")

    crypto_rows = []
    stock_rows  = []

    # --- CRYPTO: daily + hourly ---
    print("  CRYPTO ASSETS")
    print("  " + "-"*50)
    for symbol in CRYPTO:
        total = 0

        # Daily prices from existing history
        daily = load_daily_prices(symbol)
        if len(daily) >= 50:
            n = generate_labels(daily, symbol, "CRYPTO", crypto_rows)
            print(f"  {symbol:<6} daily:  {n:>5} samples")
            total += n

        # Hourly prices from Yahoo Finance
        hourly = fetch_hourly_crypto(symbol)
        if len(hourly) >= 50:
            n = generate_labels(hourly, symbol, "CRYPTO", crypto_rows)
            print(f"  {symbol:<6} hourly: {n:>5} samples")
            total += n

        print(f"  {symbol:<6} TOTAL:  {total:>5} samples\n")

    # --- STOCKS: daily only ---
    print("\n  STOCK ASSETS")
    print("  " + "-"*50)
    for symbol in STOCKS:
        daily = load_daily_prices(symbol)
        if len(daily) >= 50:
            n = generate_labels(daily, symbol, "STOCK", stock_rows)
            print(f"  {symbol:<6} {n:>5} samples")

    # --- Save ---
    fieldnames = [
        "symbol","asset_type","price","future_price","price_change","signal","label",
        "ma5_cross","ma10_cross","ma20_cross","price_vs_ma20","price_vs_ma50",
        "mom5","mom10","mom20",
        "vol10","vol20","vol_ratio",
        "rsi","macd_cross","macd_hist","bb_position",
        "roc5","roc10","score",
    ]

    crypto_path = os.path.join(PROJECT_DIR, "training_crypto.csv")
    stock_path  = os.path.join(PROJECT_DIR, "training_stocks.csv")
    combined_path = os.path.join(PROJECT_DIR, "training_data_v2.csv")

    for path, rows in [(crypto_path, crypto_rows), (stock_path, stock_rows), (combined_path, crypto_rows + stock_rows)]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n  {'='*50}")
    print(f"  Crypto samples:  {len(crypto_rows):>8,}")
    print(f"  Stock samples:   {len(stock_rows):>8,}")
    print(f"  Total samples:   {len(crypto_rows)+len(stock_rows):>8,}")
    print(f"\n  Saved:")
    print(f"    training_crypto.csv")
    print(f"    training_stocks.csv")
    print(f"    training_data_v2.csv")
    print(f"\n  Next: run train_model_v2.py\n")

if __name__ == "__main__":
    main()
