#!/usr/bin/env python3
"""
regime_classifier.py — Layer 1 Market Regime Model
Alphai Trader | alphaitrader.com

Labels market as BULL / BEAR / SIDEWAYS using:
  - SPY MA crossovers (50 vs 200) + short momentum (5 vs 20)
  - VIX fear level
  - BTC trend (risk appetite proxy)

Output: logs/regime.json
"""

import json, os
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd

LOG_DIR     = "logs"
OUTPUT_FILE = os.path.join(LOG_DIR, "regime.json")

VIX_HIGH       = 25
VIX_EXTREME    = 35
MA_FAST        = 50
MA_SLOW        = 200
MOM_FAST       = 5
MOM_SLOW       = 20
BTC_MOM_WINDOW = 14
MIN_HISTORY    = 210

def fetch(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [WARN] No data for {ticker}")
        return df
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return pd.DataFrame()

def classify_spy(df):
    if df.empty or len(df) < MIN_HISTORY:
        return {"score": 0, "detail": "insufficient data"}
    close   = df["Close"].squeeze()
    ma50    = close.rolling(MA_FAST).mean()
    ma200   = close.rolling(MA_SLOW).mean()
    ma5     = close.rolling(MOM_FAST).mean()
    ma20    = close.rolling(MOM_SLOW).mean()
    golden  = float(ma50.iloc[-1])  > float(ma200.iloc[-1])
    mom_up  = float(ma5.iloc[-1])   > float(ma20.iloc[-1])
    score   = (2 if golden else -2) + (1 if mom_up else -1)
    return {"score": score, "golden_cross": golden, "momentum_up": mom_up,
            "ma50": round(float(ma50.iloc[-1]),2), "ma200": round(float(ma200.iloc[-1]),2),
            "spy_price": round(float(close.iloc[-1]),2)}

def classify_vix(df):
    if df.empty:
        return {"score": 0, "vix": None, "level": "unknown"}
    vix = round(float(df["Close"].squeeze().iloc[-1]), 2)
    if vix < VIX_HIGH:   score, level = 1,  "LOW"
    elif vix < VIX_EXTREME: score, level = -1, "HIGH"
    else:                score, level = -3, "EXTREME"
    return {"score": score, "vix": vix, "level": level}

def classify_btc(df):
    if df.empty or len(df) < BTC_MOM_WINDOW + 2:
        return {"score": 0, "btc_price": None, "risk_on": None}
    close   = df["Close"].squeeze()
    ma14    = close.rolling(BTC_MOM_WINDOW).mean()
    price   = round(float(close.iloc[-1]), 2)
    ma_val  = round(float(ma14.iloc[-1]),  2)
    risk_on = price > ma_val
    return {"score": 1 if risk_on else -1, "btc_price": price,
            "btc_ma14": ma_val, "risk_on": risk_on}

def determine_regime(score):
    if score >= 2:  return "BULL"
    if score <= -2: return "BEAR"
    return "SIDEWAYS"

def run():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("=" * 50)
    print("  Alphai Regime Classifier")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 50)

    print("\n[1/3] Fetching SPY (2y)...")
    spy = classify_spy(fetch("SPY", period="2y"))
    print("[2/3] Fetching VIX...")
    vix = classify_vix(fetch("^VIX", period="5d"))
    print("[3/3] Fetching BTC...")
    btc = classify_btc(fetch("BTC-USD", period="60d"))

    total   = spy["score"] + vix["score"] + btc["score"]
    regime  = determine_regime(total)
    blocked = regime == "BEAR"

    result = {
        "regime": regime, "total_score": total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signals": {"spy": spy, "vix": vix, "btc": btc},
        "rules": {"buy_signals_blocked": blocked,
                  "reason": "BEAR regime: BUY suppressed" if blocked else "Normal signal processing"}
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  SPY  : ${spy.get('spy_price','?')}  |  Golden Cross: {spy.get('golden_cross','?')}")
    print(f"  VIX  : {vix.get('vix','?')}  →  {vix.get('level','?')} fear")
    print(f"  BTC  : ${btc.get('btc_price','?')}  Risk-On: {btc.get('risk_on','?')}")
    print(f"\n  SCORE  : {total:+d}")
    print(f"  REGIME : {regime}")
    print(f"  BUY BLOCKED : {blocked}")
    print(f"\n  Saved → {OUTPUT_FILE}")
    print("=" * 50)
    return result

if __name__ == "__main__":
    run()
