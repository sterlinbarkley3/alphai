#!/usr/bin/env python3
"""
label_data_v3.py — Generates high quality labeled training data from OHLCV
Uses volume, ATR, OBV, and all advanced features
Outputs separate crypto and stock datasets
"""

import os, csv
import pandas as pd
import numpy as np

PROJECT_DIR = "/Users/mythreeboyz/pythonuh/ai trader"
OHLCV_DIR   = os.path.join(PROJECT_DIR, "ohlcv")

LOOKAHEAD  = 10
MIN_CHANGE = 0.5

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

# ── Feature engineering ──────────────────────────────────────────────────────

def compute_features(df):
    """
    df: DataFrame with columns Open, High, Low, Close, Volume
    Returns a dict of features using the full row history up to this point
    """
    if len(df) < 50:
        return None

    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    volume = df["Volume"].values

    c = close[-1]

    # Moving averages
    ma5  = close[-5:].mean()
    ma10 = close[-10:].mean()
    ma20 = close[-20:].mean()
    ma50 = close[-50:].mean()

    # MA cross signals (as % diff)
    ma5_cross  = (ma5  - ma20) / ma20 * 100
    ma10_cross = (ma10 - ma20) / ma20 * 100
    ma20_cross = (ma20 - ma50) / ma50 * 100
    price_vs_ma20 = (c - ma20) / ma20 * 100
    price_vs_ma50 = (c - ma50) / ma50 * 100

    # Momentum
    mom5  = (c - close[-6])  / close[-6]  * 100 if close[-6]  != 0 else 0
    mom10 = (c - close[-11]) / close[-11] * 100 if close[-11] != 0 else 0
    mom20 = (c - close[-21]) / close[-21] * 100 if close[-21] != 0 else 0

    # Volatility
    vol10 = (close[-10:].max() - close[-10:].min()) / close[-10:].min() * 100 if close[-10:].min() != 0 else 0
    vol20 = (close[-20:].max() - close[-20:].min()) / close[-20:].min() * 100 if close[-20:].min() != 0 else 0
    vol_ratio = vol10 / vol20 if vol20 != 0 else 1

    # RSI
    deltas = pd.Series(close).diff()
    gain   = deltas.clip(lower=0).rolling(14).mean()
    loss   = (-deltas.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi_s  = 100 - (100 / (1 + rs))
    rsi    = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

    # MACD
    s         = pd.Series(close)
    ema12     = s.ewm(span=12).mean()
    ema26     = s.ewm(span=26).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9).mean()
    macd_cross = float(macd_line.iloc[-1] - sig_line.iloc[-1])
    macd_hist  = float(macd_line.iloc[-1] - sig_line.iloc[-1])

    # Bollinger Band position
    bb_mid = close[-20:].mean()
    bb_std = close[-20:].std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    bb_position = (c - bb_mid) / (bb_range / 2 + 1e-10)

    # Rate of change
    roc5  = (c - close[-6])  / close[-6]  * 100 if close[-6]  != 0 else 0
    roc10 = (c - close[-11]) / close[-11] * 100 if close[-11] != 0 else 0

    # ATR — Average True Range (measures volatility including gaps)
    tr_list = []
    for i in range(-14, 0):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i]  - close[i-1])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list)
    atr_pct = atr / c * 100 if c != 0 else 0

    # OBV — On Balance Volume (volume confirms price direction)
    obv = 0.0
    for i in range(1, min(20, len(close))):
        if close[-i] > close[-i-1]:
            obv += volume[-i]
        elif close[-i] < close[-i-1]:
            obv -= volume[-i]
    # Normalize OBV as % of average volume
    avg_vol = volume[-20:].mean()
    obv_pct = obv / avg_vol if avg_vol != 0 else 0

    # Volume trend — is volume increasing or decreasing?
    vol_ma5  = volume[-5:].mean()
    vol_ma20 = volume[-20:].mean()
    vol_trend = (vol_ma5 - vol_ma20) / vol_ma20 * 100 if vol_ma20 != 0 else 0

    # High/Low position — where is price in recent range?
    recent_high = high[-20:].max()
    recent_low  = low[-20:].min()
    hl_range    = recent_high - recent_low
    hl_position = (c - recent_low) / hl_range * 100 if hl_range != 0 else 50

    # Old rule score (keep as feature)
    score = 0
    if ma5 > ma20:    score += 2
    elif ma5 < ma20:  score -= 2
    if mom5 > 2:      score += 2
    elif mom5 > 0.5:  score += 1
    elif mom5 < -2:   score -= 2
    elif mom5 < -0.5: score -= 1
    if vol10 > 10:    score -= 1

    return {
        "ma5_cross":    round(ma5_cross,    4),
        "ma10_cross":   round(ma10_cross,   4),
        "ma20_cross":   round(ma20_cross,   4),
        "price_vs_ma20":round(price_vs_ma20,4),
        "price_vs_ma50":round(price_vs_ma50,4),
        "mom5":         round(mom5,         4),
        "mom10":        round(mom10,        4),
        "mom20":        round(mom20,        4),
        "vol10":        round(vol10,        4),
        "vol20":        round(vol20,        4),
        "vol_ratio":    round(vol_ratio,    4),
        "rsi":          round(rsi,          4),
        "macd_cross":   round(macd_cross,   4),
        "macd_hist":    round(macd_hist,    4),
        "bb_position":  round(bb_position,  4),
        "roc5":         round(roc5,         4),
        "roc10":        round(roc10,        4),
        "atr_pct":      round(atr_pct,      4),
        "obv_pct":      round(obv_pct,      4),
        "vol_trend":    round(vol_trend,    4),
        "hl_position":  round(hl_position,  4),
        "score":        score,
    }

def get_signal(f):
    buy_signals  = 0
    sell_signals = 0

    if f["ma5_cross"] > 0:         buy_signals  += 1
    else:                           sell_signals += 1

    if f["mom5"] > 1:               buy_signals  += 1
    elif f["mom5"] < -1:            sell_signals += 1

    if f["rsi"] < 40:               buy_signals  += 1
    elif f["rsi"] > 60:             sell_signals += 1

    if f["macd_cross"] > 0:         buy_signals  += 1
    else:                           sell_signals += 1

    if f["bb_position"] < -0.5:     buy_signals  += 1
    elif f["bb_position"] > 0.5:    sell_signals += 1

    if f["obv_pct"] > 0.5:          buy_signals  += 1
    elif f["obv_pct"] < -0.5:       sell_signals += 1

    if f["vol_trend"] > 10:         buy_signals  += 1
    elif f["vol_trend"] < -10:      sell_signals += 1

    if f["hl_position"] < 30:       buy_signals  += 1
    elif f["hl_position"] > 70:     sell_signals += 1

    # 8 possible signals, require 5 to agree
    if buy_signals >= 5:    return "BUY"
    elif sell_signals >= 5: return "SELL"
    else:                   return "HOLD"

# ── Label generation ─────────────────────────────────────────────────────────

def generate_labels(df, symbol, asset_type, rows):
    count = 0
    closes = df["Close"].values

    for i in range(50, len(df) - LOOKAHEAD):
        window        = df.iloc[:i]
        current_price = closes[i - 1]
        future_price  = closes[i + LOOKAHEAD - 1]

        feats = compute_features(window)
        if feats is None:
            continue

        signal = get_signal(feats)
        if signal == "HOLD":
            continue

        price_change = (future_price - current_price) / current_price * 100
        if abs(price_change) < MIN_CHANGE:
            continue

        label = 1 if (
            (signal == "BUY"  and price_change > 0) or
            (signal == "SELL" and price_change < 0)
        ) else 0

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

# ── Main ─────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "symbol","asset_type","price","future_price","price_change","signal","label",
    "ma5_cross","ma10_cross","ma20_cross","price_vs_ma20","price_vs_ma50",
    "mom5","mom10","mom20","vol10","vol20","vol_ratio",
    "rsi","macd_cross","macd_hist","bb_position","roc5","roc10",
    "atr_pct","obv_pct","vol_trend","hl_position","score",
]

def main():
    print("\n  Generating labeled training data from OHLCV...\n")

    crypto_rows = []
    stock_rows  = []

    print("  CRYPTO — daily + hourly")
    print("  " + "-"*50)
    for symbol in CRYPTO:
        total = 0

        daily_path  = os.path.join(OHLCV_DIR, f"{symbol}_daily.parquet")
        hourly_path = os.path.join(OHLCV_DIR, f"{symbol}_hourly.parquet")

        if os.path.exists(daily_path):
            df = pd.read_parquet(daily_path)
            n  = generate_labels(df, symbol, "CRYPTO", crypto_rows)
            total += n
            print(f"  {symbol:<6} daily:  {n:>6,} samples")

        if os.path.exists(hourly_path):
            df = pd.read_parquet(hourly_path)
            n  = generate_labels(df, symbol, "CRYPTO", crypto_rows)
            total += n
            print(f"  {symbol:<6} hourly: {n:>6,} samples")

        print(f"  {symbol:<6} TOTAL:  {total:>6,}\n")

    print("\n  STOCKS — daily")
    print("  " + "-"*50)
    for symbol in STOCKS:
        daily_path = os.path.join(OHLCV_DIR, f"{symbol}_daily.parquet")
        if os.path.exists(daily_path):
            df = pd.read_parquet(daily_path)
            n  = generate_labels(df, symbol, "STOCK", stock_rows)
            print(f"  {symbol:<6} {n:>6,} samples")

    # Save
    crypto_path   = os.path.join(PROJECT_DIR, "training_crypto_v3.csv")
    stock_path    = os.path.join(PROJECT_DIR, "training_stocks_v3.csv")
    combined_path = os.path.join(PROJECT_DIR, "training_data_v3.csv")

    for path, rows in [
        (crypto_path,   crypto_rows),
        (stock_path,    stock_rows),
        (combined_path, crypto_rows + stock_rows),
    ]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    total = len(crypto_rows) + len(stock_rows)
    print(f"\n  {'='*50}")
    print(f"  Crypto samples:  {len(crypto_rows):>10,}")
    print(f"  Stock samples:   {len(stock_rows):>10,}")
    print(f"  Total samples:   {total:>10,}")
    print(f"\n  Saved:")
    print(f"    training_crypto_v3.csv")
    print(f"    training_stocks_v3.csv")
    print(f"    training_data_v3.csv")
    print(f"\n  Next: run train_model_v2.py\n")

if __name__ == "__main__":
    main()
