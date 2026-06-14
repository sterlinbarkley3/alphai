#!/usr/bin/env python3
"""
ai_brain_v2.py — AI signals using v3 models with macro features
"""

import os, pickle
import pandas as pd
import numpy as np

PROJECT_DIR = "/Users/mythreeboyz/pythonuh/ai trader"
OHLCV_DIR   = os.path.join(PROJECT_DIR, "ohlcv")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

FEATURES = [
    "ma5_cross","ma10_cross","ma20_cross","price_vs_ma20","price_vs_ma50",
    "mom5","mom10","mom20","vol10","vol20","vol_ratio",
    "rsi","macd_cross","macd_hist","bb_position",
    "roc5","roc10","atr_pct","obv_pct","vol_trend","hl_position","score",
    "vix","vix_high","spy_ret_1d","spy_ret_5d","spy_trend","btc_eth_ratio","rel_strength",
]

def load_sentiment():
    """Load today's sentiment scores"""
    import json
    sentiment_file = os.path.join(PROJECT_DIR, "logs", "sentiment.json")
    if not os.path.exists(sentiment_file):
        return {}, 50
    with open(sentiment_file) as f:
        data = json.load(f)
    scores = {sym: info["combined_score"] for sym, info in data["assets"].items()}
    return scores, data.get("fear_greed", 50)

def load_models():
    try:
        with open(os.path.join(MODELS_DIR, "model_crypto.pkl"), "rb") as f: cm = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "model_stocks.pkl"), "rb") as f: sm = pickle.load(f)
        print("  Models loaded (v3 — with macro features)\n")
        return cm, sm
    except Exception as e:
        print(f"  Could not load models: {e}")
        return None, None

def load_macro():
    vix = pd.read_parquet(os.path.join(OHLCV_DIR, "VIX_daily.parquet"))
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix = vix["vix"]

    spy = pd.read_parquet(os.path.join(OHLCV_DIR, "SPY_daily.parquet"))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_ret   = spy["Close"].pct_change() * 100
    spy_close = spy["Close"]

    btc = pd.read_parquet(os.path.join(OHLCV_DIR, "BTC_daily.parquet"))
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    eth = pd.read_parquet(os.path.join(OHLCV_DIR, "ETH_daily.parquet"))
    eth.index = pd.to_datetime(eth.index).tz_localize(None)
    btc_eth = (btc["Close"] / eth["Close"].replace(0, np.nan))

    return vix, spy_ret, spy_close, btc_eth

def get_macro_features(date, vix, spy_ret, spy_close, btc_eth, asset_close):
    result = {"vix": 50.0, "vix_high": 0, "spy_ret_1d": 0.0,
              "spy_ret_5d": 0.0, "spy_trend": 0, "btc_eth_ratio": 0.0, "rel_strength": 0.0}
    try:
        vix_val = vix.asof(date)
        if not pd.isna(vix_val):
            result["vix"]      = round(float(vix_val), 2)
            result["vix_high"] = 1 if vix_val > 25 else 0

        spy_1d = spy_ret.asof(date)
        if not pd.isna(spy_1d):
            result["spy_ret_1d"] = round(float(spy_1d), 4)

        spy_loc = spy_close.index.searchsorted(date)
        if spy_loc >= 5:
            spy_5d = (spy_close.iloc[spy_loc] - spy_close.iloc[spy_loc-5]) / spy_close.iloc[spy_loc-5] * 100
            result["spy_ret_5d"] = round(float(spy_5d), 4)
            result["spy_trend"]  = 1 if spy_5d > 0 else -1

        ratio = btc_eth.asof(date)
        if not pd.isna(ratio):
            result["btc_eth_ratio"] = round(float(ratio), 4)

        asset_loc = asset_close.index.searchsorted(date)
        if asset_loc >= 5 and spy_loc >= 5:
            asset_5d = (asset_close.iloc[asset_loc] - asset_close.iloc[asset_loc-5]) / asset_close.iloc[asset_loc-5] * 100
            result["rel_strength"] = round(float(asset_5d) - result["spy_ret_5d"], 4)
    except:
        pass
    return result

def compute_features(df, vix, spy_ret, spy_close, btc_eth):
    if len(df) < 50:
        return None

    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    volume = df["Volume"].values
    c      = close[-1]
    date   = df.index[-1]

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

    tr_list = [max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])) for i in range(-14, 0)]
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

    macro = get_macro_features(date, vix, spy_ret, spy_close, btc_eth, df["Close"])

    return {
        "ma5_cross": ma5_cross, "ma10_cross": ma10_cross, "ma20_cross": ma20_cross,
        "price_vs_ma20": price_vs_ma20, "price_vs_ma50": price_vs_ma50,
        "mom5": mom5, "mom10": mom10, "mom20": mom20,
        "vol10": vol10, "vol20": vol20, "vol_ratio": vol_ratio,
        "rsi": rsi, "macd_cross": macd_cross, "macd_hist": macd_hist,
        "bb_position": bb_position, "roc5": mom5, "roc10": mom10,
        "atr_pct": atr_pct, "obv_pct": obv_pct, "vol_trend": vol_trend,
        "hl_position": hl_position, "score": score,
        "price": c,
        **macro,
    }

def get_ai_signal(model, feats, threshold=0.55, sentiment_score=0.0):
    X    = pd.DataFrame([{f: feats[f] for f in FEATURES}]).fillna(0)
    prob = model.predict_proba(X)[0][1]
    score = feats["score"]

    # Sentiment nudge — positive sentiment lowers buy threshold slightly
    # negative sentiment lowers sell threshold slightly
    buy_threshold  = threshold - (sentiment_score * 0.05)
    sell_threshold = threshold + (sentiment_score * 0.05)
    buy_threshold  = max(0.50, min(0.65, buy_threshold))
    sell_threshold = max(0.50, min(0.65, sell_threshold))

    if score >= 2 and prob >= 0.65:         return "STRONG BUY",  prob
    elif score >= 2 and prob >= buy_threshold:  return "BUY",     prob
    elif score <= -2 and prob >= 0.65:      return "STRONG SELL", prob
    elif score <= -2 and prob >= sell_threshold: return "SELL",   prob
    else:                                   return "HOLD",         prob

def main():
    print("\n" + "="*65)
    print("  STERLIN AI BRAIN v2 — v3 MODELS (MACRO FEATURES)")
    print("="*65 + "\n")

    crypto_model, stocks_model = load_models()
    if not crypto_model or not stocks_model:
        return

    print("  Loading macro data...")
    vix, spy_ret, spy_close, btc_eth = load_macro()
    print(f"  VIX today: {vix.iloc[-1]:.1f}  SPY 1d: {spy_ret.iloc[-1]:+.2f}%\n")

    sentiment_scores, fear_greed = load_sentiment()
    print(f"  Sentiment loaded — Fear & Greed: {fear_greed}/100\n")

    results = []
    for symbol in CRYPTO + STOCKS:
        is_crypto  = symbol in CRYPTO
        model      = crypto_model if is_crypto else stocks_model
        asset_type = "CRYPTO" if is_crypto else "STOCK"

        path = os.path.join(OHLCV_DIR, f"{symbol}_daily.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        if len(df) < 50:
            continue

        feats = compute_features(df, vix, spy_ret, spy_close, btc_eth)
        if feats is None:
            continue

        sent_score = sentiment_scores.get(symbol, 0.0)
        signal, prob = get_ai_signal(model, feats, sentiment_score=sent_score)
        results.append({
            "symbol": symbol, "type": asset_type,
            "price":  feats["price"], "signal": signal, "prob": prob,
            "rsi":    feats["rsi"],   "mom5":   feats["mom5"],
            "vix":    feats["vix"],   "spy5d":  feats["spy_ret_5d"],
        })

    results.sort(key=lambda x: x["prob"], reverse=True)

    icons = {"STRONG BUY":"🟢","BUY":"🟩","STRONG SELL":"🔴","SELL":"🟥","HOLD":"⬜"}

    print(f"  {'SYMBOL':<7} {'TYPE':<7} {'PRICE':<14} {'SIGNAL':<13} {'CONF':<8} {'RSI':<6} {'MOM'}")
    print("  " + "-"*72)
    for r in results:
        price_str = f"${r['price']:,.4f}" if r["price"] < 100 else f"${r['price']:,.2f}"
        icon = icons.get(r["signal"], "⬜")
        print(f"  {r['symbol']:<7} {r['type']:<7} {price_str:<14} {icon} {r['signal']:<11} {r['prob']*100:.1f}%   {r['rsi']:.0f}    {r['mom5']:+.2f}%")

    print("  " + "-"*72)
    buys  = [r for r in results if "BUY"  in r["signal"]]
    sells = [r for r in results if "SELL" in r["signal"]]
    holds = [r for r in results if r["signal"] == "HOLD"]
    print(f"\n  Bullish: {len(buys)}   Bearish: {len(sells)}   Neutral: {len(holds)}")
    if buys:
        print(f"\n  Top picks:")
        for r in buys[:3]:
            print(f"    {r['symbol']} — {r['signal']} ({r['prob']*100:.1f}% confidence, RSI {r['rsi']:.0f})")
    print()

if __name__ == "__main__":
    main()
