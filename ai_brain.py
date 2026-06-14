#!/usr/bin/env python3
"""
ai_brain.py — AI-powered signal generator
Uses the trained ML model instead of hardcoded rules.

Usage:
    python3 ai_brain.py
"""

import os, sys, pickle
import pandas as pd

PROJECT_DIR = "/Users/mythreeboyz/pythonuh/ai trader"
DATA_DIR    = "/Users/mythreeboyz/pythonuh/ai trader/data"
MODEL_PATH  = os.path.join(PROJECT_DIR, "model.pkl")

FEATURES = ["ma_cross_pct","momentum","volatility","price_vs_long","score"]
SHORT_MA  = 5
LONG_MA   = 20

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","MATIC","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

def load_prices(symbol):
    path = os.path.join(DATA_DIR, f"history_{symbol}.txt")
    if not os.path.isfile(path): return []
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

    return {
        "ma_cross_pct":  ma_cross,
        "momentum":      momentum,
        "volatility":    volatility,
        "price_vs_long": price_vs_long,
        "score":         score,
        "short_ma":      short_ma,
        "long_ma":       long_ma,
        "price":         prices[-1],
    }

def ai_signal(model, features, old_score):
    X = pd.DataFrame([{f: features[f] for f in FEATURES}])
    prob = model.predict_proba(X)[0][1]  # probability signal is correct

    # Map probability + old score to a signal
    if old_score >= 2 and prob >= 0.60:   return "STRONG BUY",  prob
    elif old_score >= 2 and prob >= 0.52: return "BUY",         prob
    elif old_score <= -2 and prob >= 0.60: return "STRONG SELL", prob
    elif old_score <= -2 and prob >= 0.52: return "SELL",        prob
    else:                                  return "HOLD",         prob

def main():
    print("\n  Loading AI model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("  Model loaded.\n")

    results = []
    for symbol in CRYPTO + STOCKS:
        prices = load_prices(symbol)
        if len(prices) < 25: continue
        features = get_features(prices)
        signal, confidence = ai_signal(model, features, features["score"])
        results.append({
            "symbol":     symbol,
            "type":       "CRYPTO" if symbol in CRYPTO else "STOCK",
            "price":      features["price"],
            "signal":     signal,
            "confidence": confidence,
            "momentum":   features["momentum"],
            "volatility": features["volatility"],
        })

    # Sort by confidence
    results.sort(key=lambda x: x["confidence"], reverse=True)

    print("  " + "="*70)
    print("  STERLIN'S AI BRAIN — ML-POWERED SIGNALS")
    print("  " + "="*70)
    print(f"  {'SYMBOL':<7} {'TYPE':<7} {'PRICE':<14} {'SIGNAL':<13} {'CONFIDENCE'}")
    print("  " + "-"*70)

    for r in results:
        conf_bar = "█" * int(r["confidence"] * 20)
        sig_color = {
            "STRONG BUY": "🟢", "BUY": "🟩",
            "STRONG SELL": "🔴", "SELL": "🟥", "HOLD": "⬜"
        }.get(r["signal"], "⬜")
        print(f"  {r['symbol']:<7} {r['type']:<7} ${r['price']:<13,.4f} "
              f"{sig_color} {r['signal']:<11} {r['confidence']*100:.1f}%")

    print("  " + "-"*70)
    buys  = [r for r in results if "BUY"  in r["signal"]]
    sells = [r for r in results if "SELL" in r["signal"]]
    holds = [r for r in results if r["signal"] == "HOLD"]
    print(f"\n  Bullish: {len(buys)}  Bearish: {len(sells)}  Neutral: {len(holds)}")

    if buys:
        print(f"\n  Top AI picks:")
        for r in buys[:3]:
            print(f"    {r['symbol']} — {r['signal']} ({r['confidence']*100:.1f}% confidence)")
    print()

if __name__ == "__main__":
    main()
