#!/usr/bin/env python3
"""
label_data.py — Generates labeled training data for ML
For every point in history, records features + whether the signal was correct.
Output: training_data.csv
"""

import os, csv

DATA_DIR   = "/root/alphai/data"
OUTPUT     = "/root/alphai/training_data.csv"
SHORT_MA   = 5
LONG_MA    = 20
MIN_PRICES = 25
LOOKAHEAD  = 10   # how many prices ahead to check if call was correct

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","MATIC","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

def load_prices(symbol):
    path = os.path.join(DATA_DIR, f"history_{symbol}.txt")
    prices = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: prices.append(float(line))
                except: pass
    return prices

def ma(prices, window):
    if len(prices) < window: return 0.0
    return sum(prices[-window:]) / window

def get_features(prices):
    short_ma   = ma(prices, SHORT_MA)
    long_ma    = ma(prices, LONG_MA)
    recent     = prices[-LONG_MA:]
    momentum   = (prices[-1] - prices[-SHORT_MA]) / prices[-SHORT_MA] * 100 if prices[-SHORT_MA] != 0 else 0
    volatility = (max(recent) - min(recent)) / min(recent) * 100 if min(recent) != 0 else 0
    ma_cross   = (short_ma - long_ma) / long_ma * 100 if long_ma != 0 else 0
    price_vs_long = (prices[-1] - long_ma) / long_ma * 100 if long_ma != 0 else 0

    score = 0
    if short_ma > long_ma:   score += 2
    elif short_ma < long_ma: score -= 2
    if momentum > 2:         score += 2
    elif momentum > 0.5:     score += 1
    elif momentum < -2:      score -= 2
    elif momentum < -0.5:    score -= 1
    if volatility > 10:      score -= 1

    if score >= 4:    signal = "STRONG BUY"
    elif score >= 2:  signal = "BUY"
    elif score <= -4: signal = "STRONG SELL"
    elif score <= -2: signal = "SELL"
    else:             signal = "HOLD"

    return {
        "short_ma":      round(short_ma, 4),
        "long_ma":       round(long_ma, 4),
        "ma_cross_pct":  round(ma_cross, 4),
        "momentum":      round(momentum, 4),
        "volatility":    round(volatility, 4),
        "price_vs_long": round(price_vs_long, 4),
        "score":         score,
        "signal":        signal,
    }

def main():
    rows = []
    total = 0

    for symbol in CRYPTO + STOCKS:
        prices = load_prices(symbol)
        asset_type = "CRYPTO" if symbol in CRYPTO else "STOCK"
        print(f"  Labeling {symbol:<6} {len(prices):>6} prices...", end="", flush=True)
        count = 0

        for i in range(MIN_PRICES, len(prices) - LOOKAHEAD):
            window       = prices[:i]
            current_price = prices[i - 1]
            future_price  = prices[i + LOOKAHEAD - 1]

            features = get_features(window)
            signal   = features["signal"]

            # Skip HOLDs — not useful for training buy/sell model
            if signal == "HOLD":
                continue

            # Label: was this a correct call?
            price_change = (future_price - current_price) / current_price * 100
            if signal in ("BUY", "STRONG BUY"):
                label = 1 if price_change > 0 else 0
            else:  # SELL / STRONG SELL
                label = 1 if price_change < 0 else 0

            rows.append({
                "symbol":        symbol,
                "asset_type":    asset_type,
                "price":         round(current_price, 4),
                "future_price":  round(future_price, 4),
                "price_change":  round(price_change, 4),
                **features,
                "label":         label,
            })
            count += 1

        print(f" {count} samples")
        total += count

    # Write CSV
    fieldnames = ["symbol","asset_type","price","future_price","price_change",
                  "short_ma","long_ma","ma_cross_pct","momentum","volatility",
                  "price_vs_long","score","signal","label"]

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Done! {total:,} labeled samples saved to training_data.csv")
    print(f"  This is your ML training dataset.\n")

if __name__ == "__main__":
    print("\n  Generating labeled training data...\n")
    main()
