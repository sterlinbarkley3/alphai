#!/usr/bin/env python3
"""
label_data_v4.py — Adds macro context features to training data
New features: VIX level, SPY daily return, BTC/ETH ratio (dominance proxy),
              asset vs SPY relative strength
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

# ── Load macro data once ─────────────────────────────────────────────────────

def load_macro():
    print("  Loading macro data (VIX, SPY, ETH)...")

    vix = pd.read_parquet(os.path.join(OHLCV_DIR, "VIX_daily.parquet"))
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix = vix["vix"]

    spy = pd.read_parquet(os.path.join(OHLCV_DIR, "SPY_daily.parquet"))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_ret = spy["Close"].pct_change() * 100  # daily % return

    btc = pd.read_parquet(os.path.join(OHLCV_DIR, "BTC_daily.parquet"))
    btc.index = pd.to_datetime(btc.index).tz_localize(None)

    eth = pd.read_parquet(os.path.join(OHLCV_DIR, "ETH_daily.parquet"))
    eth.index = pd.to_datetime(eth.index).tz_localize(None)

    # BTC/ETH ratio as dominance proxy
    btc_eth = (btc["Close"] / eth["Close"].replace(0, np.nan)).rename("btc_eth_ratio")

    print(f"  VIX: {len(vix):,} days  SPY returns: {len(spy_ret):,} days  BTC/ETH: {len(btc_eth):,} days\n")
    return vix, spy_ret, spy["Close"], btc_eth

def get_macro_features(date, vix, spy_ret, spy_close, btc_eth, asset_close_series):
    """Get macro features for a specific date"""
    result = {
        "vix":           50.0,
        "vix_high":      0,
        "spy_ret_1d":    0.0,
        "spy_ret_5d":    0.0,
        "spy_trend":     0,
        "btc_eth_ratio": 0.0,
        "rel_strength":  0.0,
    }

    try:
        # VIX on this date
        vix_val = vix.asof(date)
        if not pd.isna(vix_val):
            result["vix"]      = round(float(vix_val), 2)
            result["vix_high"] = 1 if vix_val > 25 else 0

        # SPY 1-day and 5-day return
        spy_1d = spy_ret.asof(date)
        if not pd.isna(spy_1d):
            result["spy_ret_1d"] = round(float(spy_1d), 4)

        spy_loc = spy_close.index.searchsorted(date)
        if spy_loc >= 5:
            spy_5d = (spy_close.iloc[spy_loc] - spy_close.iloc[spy_loc-5]) / spy_close.iloc[spy_loc-5] * 100
            result["spy_ret_5d"] = round(float(spy_5d), 4)
            result["spy_trend"]  = 1 if spy_5d > 0 else -1

        # BTC/ETH ratio
        ratio = btc_eth.asof(date)
        if not pd.isna(ratio):
            result["btc_eth_ratio"] = round(float(ratio), 4)

        # Relative strength vs SPY (asset 5d return vs SPY 5d return)
        asset_loc = asset_close_series.index.searchsorted(date)
        if asset_loc >= 5 and spy_loc >= 5:
            asset_5d = (asset_close_series.iloc[asset_loc] - asset_close_series.iloc[asset_loc-5]) / asset_close_series.iloc[asset_loc-5] * 100
            rel_str  = float(asset_5d) - float(result["spy_ret_5d"])
            result["rel_strength"] = round(rel_str, 4)

    except Exception:
        pass

    return result

# ── Core feature engineering ─────────────────────────────────────────────────

def compute_all_features(df, date, vix, spy_ret, spy_close, btc_eth):
    if len(df) < 50:
        return None

    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    volume = df["Volume"].values
    c = close[-1]

    ma5  = close[-5:].mean()
    ma10 = close[-10:].mean()
    ma20 = close[-20:].mean()
    ma50 = close[-50:].mean()

    ma5_cross     = (ma5  - ma20) / ma20 * 100
    ma10_cross    = (ma10 - ma20) / ma20 * 100
    ma20_cross    = (ma20 - ma50) / ma50 * 100
    price_vs_ma20 = (c - ma20) / ma20 * 100
    price_vs_ma50 = (c - ma50) / ma50 * 100

    mom5  = (c - close[-6])  / close[-6]  * 100 if close[-6]  != 0 else 0
    mom10 = (c - close[-11]) / close[-11] * 100 if close[-11] != 0 else 0
    mom20 = (c - close[-21]) / close[-21] * 100 if close[-21] != 0 else 0

    vol10     = (close[-10:].max() - close[-10:].min()) / close[-10:].min() * 100 if close[-10:].min() != 0 else 0
    vol20     = (close[-20:].max() - close[-20:].min()) / close[-20:].min() * 100 if close[-20:].min() != 0 else 0
    vol_ratio = vol10 / vol20 if vol20 != 0 else 1

    deltas = pd.Series(close).diff()
    gain   = deltas.clip(lower=0).rolling(14).mean()
    loss   = (-deltas.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi_s  = 100 - (100 / (1 + rs))
    rsi    = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

    s          = pd.Series(close)
    macd_line  = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sig_line   = macd_line.ewm(span=9).mean()
    macd_cross = float(macd_line.iloc[-1] - sig_line.iloc[-1])
    macd_hist  = macd_cross

    bb_mid      = close[-20:].mean()
    bb_std      = close[-20:].std()
    bb_position = (c - bb_mid) / (bb_std * 2 / 2 + 1e-10)

    roc5  = mom5
    roc10 = mom10

    tr_list = [max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])) for i in range(-14,0)]
    atr_pct = np.mean(tr_list) / c * 100 if c != 0 else 0

    obv     = sum(volume[-i] if close[-i] > close[-i-1] else -volume[-i] if close[-i] < close[-i-1] else 0 for i in range(1, min(20, len(close))))
    avg_vol = volume[-20:].mean()
    obv_pct = obv / avg_vol if avg_vol != 0 else 0

    vol_ma5   = volume[-5:].mean()
    vol_trend = (vol_ma5 - avg_vol) / avg_vol * 100 if avg_vol != 0 else 0

    recent_high = high[-20:].max()
    recent_low  = low[-20:].min()
    hl_range    = recent_high - recent_low
    hl_position = (c - recent_low) / hl_range * 100 if hl_range != 0 else 50

    score = 0
    if ma5 > ma20:    score += 2
    elif ma5 < ma20:  score -= 2
    if mom5 > 2:      score += 2
    elif mom5 > 0.5:  score += 1
    elif mom5 < -2:   score -= 2
    elif mom5 < -0.5: score -= 1
    if vol10 > 10:    score -= 1

    # Macro features
    asset_close_series = df["Close"]
    macro = get_macro_features(date, vix, spy_ret, spy_close, btc_eth, asset_close_series)

    return {
        "ma5_cross": round(ma5_cross,4), "ma10_cross": round(ma10_cross,4),
        "ma20_cross": round(ma20_cross,4), "price_vs_ma20": round(price_vs_ma20,4),
        "price_vs_ma50": round(price_vs_ma50,4),
        "mom5": round(mom5,4), "mom10": round(mom10,4), "mom20": round(mom20,4),
        "vol10": round(vol10,4), "vol20": round(vol20,4), "vol_ratio": round(vol_ratio,4),
        "rsi": round(rsi,4), "macd_cross": round(macd_cross,4), "macd_hist": round(macd_hist,4),
        "bb_position": round(bb_position,4), "roc5": round(roc5,4), "roc10": round(roc10,4),
        "atr_pct": round(atr_pct,4), "obv_pct": round(obv_pct,4),
        "vol_trend": round(vol_trend,4), "hl_position": round(hl_position,4),
        "score": score,
        **macro,
    }

def get_signal(f):
    buy_signals = sell_signals = 0

    if f["ma5_cross"] > 0:          buy_signals  += 1
    else:                            sell_signals += 1
    if f["mom5"] > 1:                buy_signals  += 1
    elif f["mom5"] < -1:             sell_signals += 1
    if f["rsi"] < 40:                buy_signals  += 1
    elif f["rsi"] > 60:              sell_signals += 1
    if f["macd_cross"] > 0:          buy_signals  += 1
    else:                            sell_signals += 1
    if f["bb_position"] < -0.5:      buy_signals  += 1
    elif f["bb_position"] > 0.5:     sell_signals += 1
    if f["obv_pct"] > 0.5:           buy_signals  += 1
    elif f["obv_pct"] < -0.5:        sell_signals += 1
    if f["vol_trend"] > 10:          buy_signals  += 1
    elif f["vol_trend"] < -10:       sell_signals += 1
    if f["hl_position"] < 30:        buy_signals  += 1
    elif f["hl_position"] > 70:      sell_signals += 1
    # Macro boost
    if f["spy_trend"] == 1:          buy_signals  += 1
    elif f["spy_trend"] == -1:       sell_signals += 1
    if f["vix_high"] == 1:           sell_signals += 1
    if f["rel_strength"] > 2:        buy_signals  += 1
    elif f["rel_strength"] < -2:     sell_signals += 1

    if buy_signals >= 6:    return "BUY"
    elif sell_signals >= 6: return "SELL"
    else:                   return "HOLD"

# ── Label generation ─────────────────────────────────────────────────────────

def generate_labels(df, symbol, asset_type, rows, vix, spy_ret, spy_close, btc_eth):
    closes = df["Close"].values
    dates  = df.index
    count  = 0

    for i in range(50, len(df) - LOOKAHEAD):
        window        = df.iloc[:i]
        current_price = closes[i - 1]
        future_price  = closes[i + LOOKAHEAD - 1]
        date          = dates[i - 1]

        feats = compute_all_features(window, date, vix, spy_ret, spy_close, btc_eth)
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
            "symbol": symbol, "asset_type": asset_type,
            "price": round(current_price, 6),
            "future_price": round(future_price, 6),
            "price_change": round(price_change, 4),
            "signal": signal, "label": label,
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
    "vix","vix_high","spy_ret_1d","spy_ret_5d","spy_trend","btc_eth_ratio","rel_strength",
]

def main():
    print("\n  Generating training data with macro features...\n")

    vix, spy_ret, spy_close, btc_eth = load_macro()

    crypto_rows = []
    stock_rows  = []

    print("  CRYPTO — daily + hourly")
    print("  " + "-"*50)
    for symbol in CRYPTO:
        total = 0
        for interval in ["daily", "hourly"]:
            path = os.path.join(OHLCV_DIR, f"{symbol}_{interval}.parquet")
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            n = generate_labels(df, symbol, "CRYPTO", crypto_rows, vix, spy_ret, spy_close, btc_eth)
            print(f"  {symbol:<6} {interval:<8} {n:>6,} samples")
            total += n
        print(f"  {symbol:<6} TOTAL    {total:>6,}\n")

    print("\n  STOCKS — daily")
    print("  " + "-"*50)
    for symbol in STOCKS:
        path = os.path.join(OHLCV_DIR, f"{symbol}_daily.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        n = generate_labels(df, symbol, "STOCK", stock_rows, vix, spy_ret, spy_close, btc_eth)
        print(f"  {symbol:<6} {n:>6,} samples")

    crypto_path   = os.path.join(PROJECT_DIR, "training", "training_crypto_v4.csv")
    stock_path    = os.path.join(PROJECT_DIR, "training", "training_stocks_v4.csv")

    for path, rows in [(crypto_path, crypto_rows), (stock_path, stock_rows)]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n  {'='*50}")
    print(f"  Crypto samples:  {len(crypto_rows):>10,}")
    print(f"  Stock samples:   {len(stock_rows):>10,}")
    print(f"  Total:           {len(crypto_rows)+len(stock_rows):>10,}")
    print(f"\n  Saved to training/ folder")
    print(f"  Next: run scripts/train_model_v3.py\n")

if __name__ == "__main__":
    main()
