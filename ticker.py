#!/usr/bin/env python3
"""
ticker.py — Universal ticker lookup
Search any stock or crypto, get AI signal, stats, and sentiment
Usage: python3 ticker.py TSLA
       python3 ticker.py DOGE-USD
"""

import os, sys, pickle
import pandas as pd
import numpy as np
import yfinance as yf

PROJECT_DIR = "/root/alphai"
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
OHLCV_DIR   = os.path.join(PROJECT_DIR, "ohlcv")

FEATURES = [
    "ma5_cross","ma10_cross","ma20_cross","price_vs_ma20","price_vs_ma50",
    "mom5","mom10","mom20","vol10","vol20","vol_ratio",
    "rsi","macd_cross","macd_hist","bb_position",
    "roc5","roc10","atr_pct","obv_pct","vol_trend","hl_position","score",
    "vix","vix_high","spy_ret_1d","spy_ret_5d","spy_trend","btc_eth_ratio","rel_strength",
]

CRYPTO_SUFFIXES = ["-USD", "-USDT", "-BTC"]

def is_crypto(symbol):
    return any(symbol.endswith(s) for s in CRYPTO_SUFFIXES) or symbol in [
        "BTC","ETH","XRP","SOL","DOGE","ADA","DOT","AVAX","MATIC","ATOM",
        "LINK","HBAR","XLM","LTC","BCH","UNI","AAVE","FIL","THETA"
    ]

def fetch_ticker_data(symbol):
    """Fetch 200 days of OHLCV for any ticker"""
    # Auto-add -USD for known crypto without suffix
    yahoo_sym = symbol
    if is_crypto(symbol) and "-" not in symbol:
        yahoo_sym = f"{symbol}-USD"

    try:
        ticker = yf.Ticker(yahoo_sym)
        hist   = ticker.history(period="200d", interval="1d", auto_adjust=True)
        info   = ticker.fast_info

        if hist.empty:
            return None, None, None

        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist = hist[["Open","High","Low","Close","Volume"]].dropna()

        name = getattr(info, "company_name", yahoo_sym) or yahoo_sym

        meta = {
            "symbol":    symbol.upper(),
            "yahoo_sym": yahoo_sym,
            "name":      name,
            "price":     float(info.last_price),
            "high_200":  float(hist["High"].max()),
            "low_200":   float(hist["Low"].min()),
            "avg_vol":   float(hist["Volume"].mean()),
            "asset_type": "CRYPTO" if is_crypto(symbol) else "STOCK",
        }

        return hist, meta, ticker

    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None, None, None

def load_models():
    try:
        with open(os.path.join(MODELS_DIR, "model_crypto.pkl"), "rb") as f:
            cm = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "model_stocks.pkl"), "rb") as f:
            sm = pickle.load(f)
        return cm, sm
    except:
        return None, None

def load_macro():
    try:
        vix = pd.read_parquet(os.path.join(OHLCV_DIR, "VIX_daily.parquet"))
        vix.index = pd.to_datetime(vix.index).tz_localize(None)

        spy = pd.read_parquet(os.path.join(OHLCV_DIR, "SPY_daily.parquet"))
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        spy_ret   = spy["Close"].pct_change() * 100
        spy_close = spy["Close"]

        btc = pd.read_parquet(os.path.join(OHLCV_DIR, "BTC_daily.parquet"))
        btc.index = pd.to_datetime(btc.index).tz_localize(None)
        eth = pd.read_parquet(os.path.join(OHLCV_DIR, "ETH_daily.parquet"))
        eth.index = pd.to_datetime(eth.index).tz_localize(None)
        btc_eth = btc["Close"] / eth["Close"].replace(0, np.nan)

        return vix["vix"], spy_ret, spy_close, btc_eth
    except Exception as e:
        print(f"  Macro load error: {e}")
        return None, None, None, None

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

    # Macro
    macro = {"vix": 50.0, "vix_high": 0, "spy_ret_1d": 0.0,
             "spy_ret_5d": 0.0, "spy_trend": 0, "btc_eth_ratio": 0.0, "rel_strength": 0.0}
    try:
        if vix is not None:
            vix_val = vix.asof(date)
            if not pd.isna(vix_val):
                macro["vix"]      = round(float(vix_val), 2)
                macro["vix_high"] = 1 if vix_val > 25 else 0

        if spy_ret is not None:
            spy_1d = spy_ret.asof(date)
            if not pd.isna(spy_1d):
                macro["spy_ret_1d"] = round(float(spy_1d), 4)

        if spy_close is not None:
            spy_loc = spy_close.index.searchsorted(date)
            if spy_loc >= 5:
                spy_5d = (spy_close.iloc[spy_loc] - spy_close.iloc[spy_loc-5]) / spy_close.iloc[spy_loc-5] * 100
                macro["spy_ret_5d"] = round(float(spy_5d), 4)
                macro["spy_trend"]  = 1 if spy_5d > 0 else -1

        if btc_eth is not None:
            ratio = btc_eth.asof(date)
            if not pd.isna(ratio):
                macro["btc_eth_ratio"] = round(float(ratio), 4)

        if spy_close is not None:
            asset_loc = df["Close"].index.searchsorted(date)
            spy_loc   = spy_close.index.searchsorted(date)
            if asset_loc >= 5 and spy_loc >= 5:
                asset_5d = (df["Close"].iloc[asset_loc] - df["Close"].iloc[asset_loc-5]) / df["Close"].iloc[asset_loc-5] * 100
                macro["rel_strength"] = round(float(asset_5d) - macro["spy_ret_5d"], 4)
    except:
        pass

    return {
        "ma5_cross": ma5_cross, "ma10_cross": ma10_cross, "ma20_cross": ma20_cross,
        "price_vs_ma20": price_vs_ma20, "price_vs_ma50": price_vs_ma50,
        "mom5": mom5, "mom10": mom10, "mom20": mom20,
        "vol10": vol10, "vol20": vol20, "vol_ratio": vol_ratio,
        "rsi": rsi, "macd_cross": macd_cross, "macd_hist": macd_hist,
        "bb_position": bb_position, "roc5": mom5, "roc10": mom10,
        "atr_pct": atr_pct, "obv_pct": obv_pct, "vol_trend": vol_trend,
        "hl_position": hl_position, "score": score, "price": c,
        **macro,
    }

def get_signal(model, feats, threshold=0.55):
    X    = pd.DataFrame([{f: feats[f] for f in FEATURES}]).fillna(0)
    prob = model.predict_proba(X)[0][1]
    score = feats["score"]
    if score >= 2 and prob >= 0.65:     return "STRONG BUY",  prob
    elif score >= 2 and prob >= threshold: return "BUY",      prob
    elif score <= -2 and prob >= 0.65:  return "STRONG SELL", prob
    elif score <= -2 and prob >= threshold: return "SELL",    prob
    else:                               return "HOLD",         prob

def get_news_sentiment(symbol):
    import requests, re
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        r   = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        titles = re.findall(r'<title>(.*?)</title>', r.text)
        headlines = [t.strip() for t in titles[1:6] if len(t.strip()) > 10]

        positive = ["surge","soar","rally","gain","rise","jump","bull","record",
                    "high","growth","profit","beat","strong","buy","upgrade","bullish"]
        negative = ["crash","plunge","fall","drop","loss","bear","low","decline",
                    "sell","downgrade","weak","risk","fear","warning","bearish"]

        score = 0.0
        for h in headlines:
            hl = h.lower()
            pos = sum(1 for w in positive if w in hl)
            neg = sum(1 for w in negative if w in hl)
            if pos + neg > 0:
                score += (pos - neg) / (pos + neg)

        sentiment = round(score / len(headlines), 3) if headlines else 0.0
        return sentiment, headlines
    except:
        return 0.0, []

def analyze(symbol):
    symbol = symbol.upper().strip()
    print(f"\n  Fetching data for {symbol}...")

    df, meta, ticker_obj = fetch_ticker_data(symbol)
    if df is None:
        print(f"  Could not find ticker: {symbol}")
        print("  Make sure it's a valid Yahoo Finance symbol.")
        print("  Examples: TSLA, GOOGL, DOGE-USD, ETH-USD")
        return

    crypto_model, stocks_model = load_models()
    model = crypto_model if meta["asset_type"] == "CRYPTO" else stocks_model

    vix, spy_ret, spy_close, btc_eth = load_macro()
    feats = compute_features(df, vix, spy_ret, spy_close, btc_eth)

    if feats is None or model is None:
        print("  Not enough data to analyze.")
        return

    signal, prob = get_signal(model, feats)
    sentiment, headlines = get_news_sentiment(symbol)

    # Price change
    daily_change = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
    weekly_change = (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / df["Close"].iloc[-5] * 100

    # Format price
    price = meta["price"]
    price_str = f"${price:,.5f}" if price < 1 else f"${price:,.2f}"

    # Signal styling
    sig_icons = {"STRONG BUY":"🟢","BUY":"🟩","STRONG SELL":"🔴","SELL":"🟥","HOLD":"⬜"}
    icon = sig_icons.get(signal, "⬜")

    print("\n" + "="*55)
    print(f"  {symbol}  —  {meta['name'][:30]}")
    print(f"  {meta['asset_type']}")
    print("="*55)
    print(f"  Price:          {price_str}")
    print(f"  24h Change:     {daily_change:+.2f}%")
    print(f"  7d Change:      {weekly_change:+.2f}%")
    print(f"  200d High:      ${meta['high_200']:,.4f}")
    print(f"  200d Low:       ${meta['low_200']:,.4f}")
    print("-"*55)
    print(f"  AI Signal:      {icon} {signal}")
    print(f"  Confidence:     {prob*100:.1f}%")
    print(f"  RSI:            {feats['rsi']:.1f}")
    print(f"  Momentum (5d):  {feats['mom5']:+.2f}%")
    print(f"  Volatility:     {feats['vol10']:.2f}%")
    print(f"  VIX today:      {feats['vix']:.1f}")
    print(f"  vs Market:      {feats['rel_strength']:+.2f}%")
    print("-"*55)

    sent_icon = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "⬜"
    print(f"  News Sentiment: {sent_icon} {sentiment:+.3f}")
    if headlines:
        print(f"  Latest news:")
        for h in headlines[:3]:
            print(f"    • {h[:55]}...")
    else:
        print("  No recent headlines found.")

    print("="*55 + "\n")

def main():
    if len(sys.argv) < 2:
        print("\n  Usage: python3 ticker.py SYMBOL")
        print("  Examples:")
        print("    python3 ticker.py TSLA")
        print("    python3 ticker.py DOGE-USD")
        print("    python3 ticker.py GOOGL\n")
        return
    analyze(sys.argv[1])

if __name__ == "__main__":
    main()
