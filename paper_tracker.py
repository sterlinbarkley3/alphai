#!/usr/bin/env python3
"""
paper_tracker.py — Tracks AI signals and measures real forward accuracy
Every time ai_brain_v2.py runs, signals get logged here.
10 days later, we check if the call was correct and record the result.

Usage:
    python3 paper_tracker.py --log      log today's signals
    python3 paper_tracker.py --grade    grade signals that are 10+ days old
    python3 paper_tracker.py --report   show full performance report
    python3 paper_tracker.py            do all three
"""

import os, json, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROJECT_DIR  = "/root/alphai"
OHLCV_DIR    = os.path.join(PROJECT_DIR, "ohlcv")
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
TRACKER_FILE = os.path.join(PROJECT_DIR, "logs", "paper_trades.json")

GRADE_AFTER_DAYS = 10
MIN_MOVE_PCT     = 0.5   # ignore signals where price barely moved

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

# ── Load / Save ──────────────────────────────────────────────────────────────

def load_tracker():
    if not os.path.exists(TRACKER_FILE):
        return {"signals": [], "graded": []}
    with open(TRACKER_FILE) as f:
        return json.load(f)

def save_tracker(data):
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ── Get current AI signals ───────────────────────────────────────────────────

def get_current_signals():
    """Run ai_brain_v2 and capture signals"""
    import pickle
    import importlib.util

    # Load models
    try:
        with open(os.path.join(MODELS_DIR, "model_crypto.pkl"), "rb") as f:
            crypto_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "model_stocks.pkl"), "rb") as f:
            stocks_model = pickle.load(f)
    except Exception as e:
        print(f"  Could not load models: {e}")
        return []

    # Load macro
    try:
        vix = pd.read_parquet(os.path.join(OHLCV_DIR, "VIX_daily.parquet"))
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        vix_series = vix["vix"]

        spy = pd.read_parquet(os.path.join(OHLCV_DIR, "SPY_daily.parquet"))
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        spy_ret   = spy["Close"].pct_change() * 100
        spy_close = spy["Close"]

        btc = pd.read_parquet(os.path.join(OHLCV_DIR, "BTC_daily.parquet"))
        btc.index = pd.to_datetime(btc.index).tz_localize(None)
        eth = pd.read_parquet(os.path.join(OHLCV_DIR, "ETH_daily.parquet"))
        eth.index = pd.to_datetime(eth.index).tz_localize(None)
        btc_eth = btc["Close"] / eth["Close"].replace(0, np.nan)
    except Exception as e:
        print(f"  Could not load macro data: {e}")
        return []

    # Import brain functions
    spec = importlib.util.spec_from_file_location(
        "ai_brain_v2",
        os.path.join(PROJECT_DIR, "ai_brain_v2.py")
    )
    brain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(brain)

    signals = []
    today   = datetime.now().strftime("%Y-%m-%d")

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

        feats = brain.compute_features(df, vix_series, spy_ret, spy_close, btc_eth)
        if feats is None:
            continue

        signal, prob = brain.get_ai_signal(model, feats)
        if signal == "HOLD":
            continue  # only track actionable signals

        signals.append({
            "date":        today,
            "symbol":      symbol,
            "asset_type":  asset_type,
            "signal":      signal,
            "confidence":  round(prob * 100, 2),
            "entry_price": round(feats["price"], 6),
            "graded":      False,
            "correct":     None,
            "exit_price":  None,
            "price_change":None,
            "grade_date":  None,
        })

    return signals

# ── Log today's signals ──────────────────────────────────────────────────────

def log_signals():
    print("\n  Logging today's AI signals...")
    tracker = load_tracker()
    today   = datetime.now().strftime("%Y-%m-%d")

    # Check if already logged today
    already_logged = [s for s in tracker["signals"] if s["date"] == today]
    if already_logged and "--force" not in sys.argv:
        print(f"  Already logged {len(already_logged)} signals for {today}. Use --force to re-log.")
        return

    signals = get_current_signals()
    if not signals:
        print("  No actionable signals today.")
        return

    tracker["signals"].extend(signals)
    save_tracker(tracker)

    print(f"  Logged {len(signals)} signals for {today}:")
    for s in signals:
        icon = "🟢" if "BUY" in s["signal"] else "🔴"
        print(f"    {icon} {s['symbol']:<6} {s['signal']:<13} @ ${s['entry_price']:,.4f}  ({s['confidence']}% confidence)")

# ── Grade old signals ────────────────────────────────────────────────────────

def grade_signals():
    print("\n  Grading matured signals...")
    tracker  = load_tracker()
    today    = datetime.now()
    graded   = 0
    skipped  = 0

    for signal in tracker["signals"]:
        if signal["graded"]:
            continue

        signal_date = datetime.strptime(signal["date"], "%Y-%m-%d")
        age_days    = (today - signal_date).days

        if age_days < GRADE_AFTER_DAYS:
            skipped += 1
            continue

        # Get current price from OHLCV
        symbol = signal["symbol"]
        path   = os.path.join(OHLCV_DIR, f"{symbol}_daily.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # Find price GRADE_AFTER_DAYS after signal
        target_date = signal_date + timedelta(days=GRADE_AFTER_DAYS)
        future_prices = df[df.index >= target_date]
        if future_prices.empty:
            continue

        exit_price   = float(future_prices["Close"].iloc[0])
        entry_price  = signal["entry_price"]
        price_change = (exit_price - entry_price) / entry_price * 100

        # Was the signal correct?
        if abs(price_change) < MIN_MOVE_PCT:
            # Price barely moved — inconclusive
            correct = None
        elif "BUY" in signal["signal"]:
            correct = price_change > 0
        else:  # SELL
            correct = price_change < 0

        signal["graded"]      = True
        signal["correct"]     = correct
        signal["exit_price"]  = round(exit_price, 6)
        signal["price_change"]= round(price_change, 4)
        signal["grade_date"]  = today.strftime("%Y-%m-%d")

        result_icon = "✅" if correct else "❌" if correct is False else "➖"
        print(f"  {result_icon} {symbol:<6} {signal['signal']:<13} entry=${entry_price:,.4f} exit=${exit_price:,.4f} ({price_change:+.2f}%)")
        graded += 1

    save_tracker(tracker)
    print(f"\n  Graded: {graded}  Still pending: {skipped}")

# ── Performance report ───────────────────────────────────────────────────────

def performance_report():
    tracker = load_tracker()
    all_signals = tracker["signals"]
    graded = [s for s in all_signals if s["graded"] and s["correct"] is not None]

    print("\n" + "="*60)
    print("  PAPER TRADING PERFORMANCE REPORT")
    print("="*60)
    print(f"  Total signals logged:  {len(all_signals)}")
    print(f"  Graded (conclusive):   {len(graded)}")
    pending = [s for s in all_signals if not s["graded"]]
    print(f"  Pending grading:       {len(pending)}")

    if not graded:
        print("\n  No graded signals yet — check back in 10 days.")
        print("="*60)
        return

    wins   = [s for s in graded if s["correct"]]
    losses = [s for s in graded if not s["correct"]]
    win_rate = len(wins) / len(graded) * 100

    print(f"\n  Win rate:    {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")

    # By signal type
    buys  = [s for s in graded if "BUY"  in s["signal"]]
    sells = [s for s in graded if "SELL" in s["signal"]]
    if buys:
        buy_wr = sum(1 for s in buys if s["correct"]) / len(buys) * 100
        print(f"  BUY signals: {buy_wr:.1f}% win rate ({len(buys)} total)")
    if sells:
        sell_wr = sum(1 for s in sells if s["correct"]) / len(sells) * 100
        print(f"  SELL signals:{sell_wr:.1f}% win rate ({len(sells)} total)")

    # By asset type
    crypto_g = [s for s in graded if s["asset_type"] == "CRYPTO"]
    stocks_g = [s for s in graded if s["asset_type"] == "STOCK"]
    if crypto_g:
        c_wr = sum(1 for s in crypto_g if s["correct"]) / len(crypto_g) * 100
        print(f"  Crypto:      {c_wr:.1f}% win rate ({len(crypto_g)} signals)")
    if stocks_g:
        s_wr = sum(1 for s in stocks_g if s["correct"]) / len(stocks_g) * 100
        print(f"  Stocks:      {s_wr:.1f}% win rate ({len(stocks_g)} signals)")

    # Average return on winning trades
    if wins:
        avg_win_return = np.mean([abs(s["price_change"]) for s in wins])
        print(f"\n  Avg gain on wins:  +{avg_win_return:.2f}%")
    if losses:
        avg_loss_return = np.mean([abs(s["price_change"]) for s in losses])
        print(f"  Avg loss on losses: -{avg_loss_return:.2f}%")

    # Best and worst calls
    if graded:
        best  = max(graded, key=lambda s: s["price_change"] if "BUY" in s["signal"] else -s["price_change"])
        worst = min(graded, key=lambda s: s["price_change"] if "BUY" in s["signal"] else -s["price_change"])
        print(f"\n  Best call:  {best['symbol']} {best['signal']} ({best['price_change']:+.2f}%)")
        print(f"  Worst call: {worst['symbol']} {worst['signal']} ({worst['price_change']:+.2f}%)")

    # Per-asset breakdown
    print(f"\n  {'SYMBOL':<8} {'SIGNALS':<9} {'WINS':<6} {'WIN%':<8} {'AVG MOVE'}")
    print("  " + "-"*45)
    symbols = sorted(set(s["symbol"] for s in graded))
    for sym in symbols:
        sym_g  = [s for s in graded if s["symbol"] == sym]
        sym_w  = [s for s in sym_g  if s["correct"]]
        sym_wr = len(sym_w) / len(sym_g) * 100
        avg_mv = np.mean([abs(s["price_change"]) for s in sym_g])
        print(f"  {sym:<8} {len(sym_g):<9} {len(sym_w):<6} {sym_wr:<8.1f} {avg_mv:+.2f}%")

    print("="*60)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--log" in args:
        log_signals()
    elif "--grade" in args:
        grade_signals()
    elif "--report" in args:
        performance_report()
    else:
        # Do all three
        log_signals()
        grade_signals()
        performance_report()

if __name__ == "__main__":
    main()
