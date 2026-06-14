import requests
import yfinance as yf
from datetime import datetime


COINBASE_SYMBOLS = {
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
    "BTC": "BTC-USD",
}


def save_price(symbol, price):
    filename = f"history_{symbol}.txt"
    with open(filename, "a") as file:
        file.write(str(price) + "\n")


def get_crypto_price(symbol):
    pair = COINBASE_SYMBOLS.get(symbol, f"{symbol}-USD")
    url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        price = float(data["data"]["amount"])
        save_price(symbol, price)
        return price
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def get_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.fast_info
        price = float(data.last_price)
        save_price(symbol, price)
        return price
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def get_price(symbol, asset_type="crypto"):
    if asset_type == "crypto":
        return get_crypto_price(symbol)
    else:
        return get_stock_price(symbol)


def load_prices(symbol):
    filename = f"history_{symbol}.txt"
    prices = []
    try:
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if "," in line:
                    parts = line.split(",")
                    try:
                        prices.append(float(parts[-1]))
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


# Keep backward compatibility for BTC
def get_btc_price():
    return get_crypto_price("BTC")