import yfinance as yf
import requests
import os

# Yahoo Finance symbols for crypto and stocks
YAHOO_SYMBOLS = {
    "BTC": "BTC-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "LINK": "LINK-USD",
    "HBAR": "HBAR-USD",
    "XLM": "XLM-USD",
    "ADA": "ADA-USD",
    "DOT": "DOT-USD",
    "AVAX": "AVAX-USD",
    "MATIC": "MATIC-USD",
    "ATOM": "ATOM-USD",
    "LMT": "LMT",
    "ABTC": "ABTC",
    "PFE": "PFE",
    "ORCL": "ORCL",
    "AAPL": "AAPL",
    "NVDA": "NVDA",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "JPM": "JPM",
    "SPY": "SPY",
    "QQQ": "QQQ",
}

def seed_asset(symbol, yahoo_symbol):
    filename = f"history_{symbol}.txt"

    # Check if file already has lots of data
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing = [l.strip() for l in f if l.strip()]
        if len(existing) > 100:
            print(f"{symbol:8} Already has {len(existing)} prices, skipping.")
            return

    print(f"{symbol:8} Fetching full history...")

    try:
        ticker = yf.Ticker(yahoo_symbol)
        # Pull maximum available history
        hist = ticker.history(period="max")

        if hist.empty:
            print(f"{symbol:8} No data returned, skipping.")
            return

        prices = hist["Close"].dropna().tolist()

        with open(filename, "w") as f:
            for price in prices:
                f.write(str(round(price, 6)) + "\n")

        print(f"{symbol:8} Loaded {len(prices)} prices successfully.")

    except Exception as e:
        print(f"{symbol:8} Error: {e}")


print("=" * 50)
print("       SEEDING HISTORICAL DATA")
print("=" * 50)

for symbol, yahoo_symbol in YAHOO_SYMBOLS.items():
    seed_asset(symbol, yahoo_symbol)

print("=" * 50)
print("Done! Run python3 brain.py to see results.")
print("=" * 50)
