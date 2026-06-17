#!/usr/bin/env python3
"""
fetch_ohlcv.py — Pulls full OHLCV history for all assets
Saves as parquet files for fast loading
"""

import os
import yfinance as yf
import pandas as pd

PROJECT_DIR = "/root/alphai"
OHLCV_DIR   = os.path.join(PROJECT_DIR, "ohlcv")
os.makedirs(OHLCV_DIR, exist_ok=True)

ASSETS = {
    # Crypto
    "BTC":   "BTC-USD",
    "XRP":   "XRP-USD",
    "SOL":   "SOL-USD",
    "LINK":  "LINK-USD",
    "HBAR":  "HBAR-USD",
    "XLM":   "XLM-USD",
    "ADA":   "ADA-USD",
    "DOT":   "DOT-USD",
    "AVAX":  "AVAX-USD",
    "ATOM":  "ATOM-USD",
    # Stocks
    "LMT":   "LMT",
    "ABTC":  "ABTC",
    "PFE":   "PFE",
    "ORCL":  "ORCL",
    "AAPL":  "AAPL",
    "NVDA":  "NVDA",
    "MSFT":  "MSFT",
    "AMZN":  "AMZN",
    "JPM":   "JPM",
    "SPY":   "SPY",
    "QQQ":   "QQQ",
}

def fetch_asset(symbol, yahoo_sym):
    path = os.path.join(OHLCV_DIR, f"{symbol}_daily.parquet")

    print(f"  {symbol:<6} fetching daily OHLCV...", end="", flush=True)
    try:
        ticker = yf.Ticker(yahoo_sym)
        hist   = ticker.history(period="max", interval="1d", auto_adjust=True)

        if hist.empty:
            print(" no data")
            return 0

        hist.index = pd.to_datetime(hist.index)
        hist = hist[["Open","High","Low","Close","Volume"]].dropna()
        hist.to_parquet(path)
        print(f" {len(hist):,} days saved")
        return len(hist)

    except Exception as e:
        print(f" error: {e}")
        return 0

def fetch_hourly(symbol, yahoo_sym):
    # Only for crypto — stocks don't have enough hourly history on Yahoo
    path = os.path.join(OHLCV_DIR, f"{symbol}_hourly.parquet")

    print(f"  {symbol:<6} fetching hourly OHLCV...", end="", flush=True)
    try:
        ticker = yf.Ticker(yahoo_sym)
        hist   = ticker.history(period="730d", interval="1h", auto_adjust=True)

        if hist.empty:
            print(" no data")
            return 0

        hist.index = pd.to_datetime(hist.index)
        hist = hist[["Open","High","Low","Close","Volume"]].dropna()
        hist.to_parquet(path)
        print(f" {len(hist):,} hours saved")
        return len(hist)

    except Exception as e:
        print(f" error: {e}")
        return 0

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]

def main():
    print(f"\n  Fetching OHLCV data for {len(ASSETS)} assets...\n")

    total_daily  = 0
    total_hourly = 0

    print("  DAILY (all assets)")
    print("  " + "-"*50)
    for symbol, yahoo_sym in ASSETS.items():
        total_daily += fetch_asset(symbol, yahoo_sym)

    print(f"\n  HOURLY (crypto only)")
    print("  " + "-"*50)
    for symbol in CRYPTO:
        yahoo_sym = ASSETS[symbol]
        total_hourly += fetch_hourly(symbol, yahoo_sym)

    print(f"\n  {'='*50}")
    print(f"  Total daily rows:  {total_daily:>10,}")
    print(f"  Total hourly rows: {total_hourly:>10,}")
    print(f"  Saved to ohlcv/ folder")
    print(f"\n  Next: run label_data_v3.py\n")

if __name__ == "__main__":
    main()
