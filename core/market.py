import requests
import yfinance as yf
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

COINBASE_SYMBOLS = {
    "XRP": "XRP-USD", "SOL": "SOL-USD", "LINK": "LINK-USD",
    "HBAR": "HBAR-USD", "XLM": "XLM-USD", "ADA": "ADA-USD",
    "DOT": "DOT-USD", "AVAX": "AVAX-USD", "MATIC": "MATIC-USD",
    "ATOM": "ATOM-USD", "BTC": "BTC-USD",
}

def save_price(symbol, price):
    with open(f"{DATA_DIR}/history_{symbol}.txt", "a") as f:
        f.write(str(price) + "\n")

def get_crypto_price(symbol):
    pair = COINBASE_SYMBOLS.get(symbol, f"{symbol}-USD")
    url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
    try:
        response = requests.get(url, timeout=10)
        price = float(response.json()["data"]["amount"])
        save_price(symbol, price)
        return price
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def get_stock_price(symbol):
    try:
        price = float(yf.Ticker(symbol).fast_info.last_price)
        save_price(symbol, price)
        return price
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def get_price(symbol, asset_type="crypto"):
    return get_crypto_price(symbol) if asset_type == "crypto" else get_stock_price(symbol)

def load_prices(symbol):
    filepath = f"{DATA_DIR}/history_{symbol}.txt"
    prices = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "," in line:
                    try:
                        prices.append(float(line.split(",")[-1]))
                    except ValueError:
                        continue
                else:
                    try:
                        prices.append(float(line))
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass
    return prices

def get_btc_price():
    return get_crypto_price("BTC")
