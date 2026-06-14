#!/usr/bin/env python3
"""
auto_trader.py — Autonomous paper trading agent
Runs AI signals and automatically executes trades based on confidence
Tracks hold time for tax-aware ROI
Runs via cron after ai_brain_v2.py generates signals

Usage:
    python3 auto_trader.py          run once
    python3 auto_trader.py --report show performance with tax
"""

import os, sys, json, pickle
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_DIR   = "/Users/mythreeboyz/pythonuh/ai trader"
MODELS_DIR    = os.path.join(PROJECT_DIR, "models")
OHLCV_DIR     = os.path.join(PROJECT_DIR, "ohlcv")
PORTFOLIO_FILE = os.path.join(PROJECT_DIR, "logs", "auto_portfolio.json")

STARTING_CASH = 10_000.0
MAX_POSITIONS = 5        # max number of assets held at once
MIN_CONFIDENCE = 0.65    # only trade above 65% confidence

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

FEATURES = [
    "ma5_cross","ma10_cross","ma20_cross","price_vs_ma20","price_vs_ma50",
    "mom5","mom10","mom20","vol10","vol20","vol_ratio",
    "rsi","macd_cross","macd_hist","bb_position",
    "roc5","roc10","atr_pct","obv_pct","vol_trend","hl_position","score",
    "vix","vix_high","spy_ret_1d","spy_ret_5d","spy_trend","btc_eth_ratio","rel_strength",
]

# ── Portfolio management ─────────────────────────────────────────────────────

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return {
            "cash":       STARTING_CASH,
            "starting_cash": STARTING_CASH,
            "positions":  {},   # symbol -> {qty, entry_price, entry_date, cost_basis}
            "trades":     [],   # completed trades
            "created":    datetime.now().isoformat(),
        }
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

def position_size(confidence, signal, portfolio_value):
    """
    Returns dollar amount to invest based on confidence
    Higher confidence = bigger position
    """
    if confidence >= 0.80:
        pct = 0.20
    elif confidence >= 0.75:
        pct = 0.15
    elif confidence >= 0.70:
        pct = 0.10
    else:
        pct = 0.05

    # Bonus for strong signals
    if "STRONG" in signal:
        pct += 0.05

    # Never risk more than 25%
    pct = min(pct, 0.25)

    return portfolio_value * pct

# ── Tax calculation ──────────────────────────────────────────────────────────

def calculate_tax(gain, entry_date_str, exit_date_str, bracket=0.32):
    """
    Calculate tax on a gain based on hold time
    Short term (<1 year): taxed at income bracket rate
    Long term (>=1 year): 15% for most people
    """
    if gain <= 0:
        return 0.0, "loss"

    entry = datetime.fromisoformat(entry_date_str)
    exit  = datetime.fromisoformat(exit_date_str)
    days  = (exit - entry).days

    if days >= 365:
        tax_rate = 0.15   # long term capital gains
        tax_type = "long-term (15%)"
    else:
        tax_rate = bracket  # short term = ordinary income
        tax_type = f"short-term ({int(bracket*100)}%)"

    tax = gain * tax_rate
    return round(tax, 2), tax_type

# ── Signal generation ────────────────────────────────────────────────────────

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
    except:
        return None, None, None, None

def load_sentiment_scores():
    import json
    path = os.path.join(PROJECT_DIR, "logs", "sentiment.json")
    if not os.path.exists(path):
        return {}, 50
    with open(path) as f:
        data = json.load(f)
    return {sym: info["combined_score"] for sym, info in data["assets"].items()}, data.get("fear_greed", 50)

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
        "rsi": rsi, "macd_cross": macd_cross, "macd_hist": macd_cross,
        "bb_position": bb_position, "roc5": mom5, "roc10": mom10,
        "atr_pct": atr_pct, "obv_pct": obv_pct, "vol_trend": vol_trend,
        "hl_position": hl_position, "score": score, "price": c,
        **macro,
    }

def get_signal(model, feats, sentiment=0.0):
    X    = pd.DataFrame([{f: feats[f] for f in FEATURES}]).fillna(0)
    prob = model.predict_proba(X)[0][1]
    score = feats["score"]
    threshold = 0.55 - (sentiment * 0.05)
    threshold = max(0.50, min(0.65, threshold))

    if score >= 2 and prob >= 0.65:         return "STRONG BUY",  prob
    elif score >= 2 and prob >= threshold:  return "BUY",         prob
    elif score <= -2 and prob >= 0.65:      return "STRONG SELL", prob
    elif score <= -2 and prob >= threshold: return "SELL",        prob
    else:                                   return "HOLD",         prob

# ── Trade execution ──────────────────────────────────────────────────────────

def execute_trades(signals, portfolio):
    now       = datetime.now().isoformat()
    today     = datetime.now().strftime("%Y-%m-%d")
    executed  = []

    # Current portfolio value
    total_value = portfolio["cash"]
    for sym, pos in portfolio["positions"].items():
        # Get latest price
        path = os.path.join(OHLCV_DIR, f"{sym}_daily.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            latest_price = float(df["Close"].iloc[-1])
            total_value += pos["qty"] * latest_price

    for sig in signals:
        symbol   = sig["symbol"]
        signal   = sig["signal"]
        price    = sig["price"]
        prob     = sig["prob"]

        # ── SELL / STRONG SELL ───────────────────────────────────────────────
        if "SELL" in signal and symbol in portfolio["positions"]:
            pos       = portfolio["positions"][symbol]
            qty       = pos["qty"]
            proceeds  = qty * price
            cost      = pos["cost_basis"]
            gain      = proceeds - cost
            hold_days = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days

            tax, tax_type = calculate_tax(gain, pos["entry_date"], now)
            net_gain      = gain - tax

            portfolio["cash"] += proceeds
            del portfolio["positions"][symbol]

            trade = {
                "type":        "SELL",
                "symbol":      symbol,
                "signal":      signal,
                "confidence":  round(prob * 100, 2),
                "qty":         round(qty, 6),
                "price":       round(price, 4),
                "proceeds":    round(proceeds, 2),
                "cost_basis":  round(cost, 2),
                "gross_gain":  round(gain, 2),
                "tax":         round(tax, 2),
                "net_gain":    round(net_gain, 2),
                "tax_type":    tax_type,
                "hold_days":   hold_days,
                "date":        today,
                "timestamp":   now,
            }
            portfolio["trades"].append(trade)
            executed.append(trade)
            print(f"  🔴 SOLD  {symbol:<6} @ ${price:,.4f}  gain=${gain:+,.2f}  tax=${tax:.2f}  net=${net_gain:+,.2f}  ({tax_type})")

        # ── BUY / STRONG BUY ─────────────────────────────────────────────────
        elif "BUY" in signal and symbol not in portfolio["positions"]:
            # Check position limit
            if len(portfolio["positions"]) >= MAX_POSITIONS:
                print(f"  ⚠️  {symbol}: position limit reached ({MAX_POSITIONS}), skipping")
                continue

            invest = position_size(prob, signal, total_value)
            invest = min(invest, portfolio["cash"] * 0.95)  # never use more than 95% of cash

            if invest < 10:
                print(f"  ⚠️  {symbol}: not enough cash (${portfolio['cash']:.2f}), skipping")
                continue

            qty  = invest / price
            portfolio["cash"] -= invest
            portfolio["positions"][symbol] = {
                "qty":         round(qty, 6),
                "entry_price": round(price, 4),
                "entry_date":  now,
                "cost_basis":  round(invest, 2),
                "signal":      signal,
                "confidence":  round(prob * 100, 2),
            }

            trade = {
                "type":       "BUY",
                "symbol":     symbol,
                "signal":     signal,
                "confidence": round(prob * 100, 2),
                "qty":        round(qty, 6),
                "price":      round(price, 4),
                "invested":   round(invest, 2),
                "date":       today,
                "timestamp":  now,
            }
            portfolio["trades"].append(trade)
            executed.append(trade)
            pct = invest / total_value * 100
            print(f"  🟢 BOUGHT {symbol:<6} @ ${price:,.4f}  invested=${invest:,.2f} ({pct:.1f}% of portfolio)  confidence={prob*100:.1f}%")

    return executed

# ── Performance report ───────────────────────────────────────────────────────

def performance_report(portfolio):
    trades    = portfolio["trades"]
    sells     = [t for t in trades if t["type"] == "SELL"]
    buys      = [t for t in trades if t["type"] == "BUY"]
    positions = portfolio["positions"]

    # Current value of open positions
    open_value = 0.0
    for sym, pos in positions.items():
        path = os.path.join(OHLCV_DIR, f"{sym}_daily.parquet")
        if os.path.exists(path):
            df           = pd.read_parquet(path)
            latest_price = float(df["Close"].iloc[-1])
            open_value  += pos["qty"] * latest_price

    total_value  = portfolio["cash"] + open_value
    total_return = (total_value - STARTING_CASH) / STARTING_CASH * 100

    print("\n" + "="*60)
    print("  AUTONOMOUS TRADER — PERFORMANCE REPORT")
    print("="*60)
    print(f"  Starting cash:    ${STARTING_CASH:>10,.2f}")
    print(f"  Current cash:     ${portfolio['cash']:>10,.2f}")
    print(f"  Open positions:   ${open_value:>10,.2f}")
    print(f"  Total value:      ${total_value:>10,.2f}")
    print(f"  Total return:     {total_return:>+10.2f}%")

    if sells:
        total_gross = sum(t["gross_gain"] for t in sells)
        total_tax   = sum(t["tax"]        for t in sells)
        total_net   = sum(t["net_gain"]   for t in sells)
        wins        = [t for t in sells if t["gross_gain"] > 0]
        losses      = [t for t in sells if t["gross_gain"] <= 0]
        win_rate    = len(wins) / len(sells) * 100

        print(f"\n  Completed trades: {len(sells)}")
        print(f"  Win rate:         {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"  Gross P&L:        ${total_gross:>+10,.2f}")
        print(f"  Total taxes paid: ${total_tax:>10,.2f}")
        print(f"  Net P&L:          ${total_net:>+10,.2f}")

        # Tax breakdown
        short_trades = [t for t in sells if "short" in t.get("tax_type","")]
        long_trades  = [t for t in sells if "long"  in t.get("tax_type","")]
        if short_trades:
            st_tax = sum(t["tax"] for t in short_trades)
            print(f"\n  Short-term trades: {len(short_trades)}  taxes: ${st_tax:,.2f}")
        if long_trades:
            lt_tax = sum(t["tax"] for t in long_trades)
            print(f"  Long-term trades:  {len(long_trades)}  taxes: ${lt_tax:,.2f}")

        # ROI after tax
        net_roi = (STARTING_CASH + total_net - STARTING_CASH) / STARTING_CASH * 100
        print(f"\n  After-tax ROI on closed trades: {net_roi:+.2f}%")

    if positions:
        print(f"\n  Open positions ({len(positions)}):")
        for sym, pos in positions.items():
            path = os.path.join(OHLCV_DIR, f"{sym}_daily.parquet")
            if os.path.exists(path):
                df           = pd.read_parquet(path)
                latest_price = float(df["Close"].iloc[-1])
                current_val  = pos["qty"] * latest_price
                unrealized   = current_val - pos["cost_basis"]
                hold_days    = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days
                tax_status   = "long-term" if hold_days >= 365 else f"short-term ({hold_days}d)"
                print(f"    {sym:<6} cost=${pos['cost_basis']:,.2f}  value=${current_val:,.2f}  unrealized={unrealized:+,.2f}  {tax_status}")
    else:
        print("\n  No open positions.")

    print("="*60 + "\n")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if "--report" in sys.argv:
        portfolio = load_portfolio()
        performance_report(portfolio)
        return

    from datetime import date as _date
    _project_start = _date(2026, 6, 2)
    _days_old = (_date.today() - _project_start).days
    _now_str = datetime.now().strftime("%b %d %Y  %I:%M %p")
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  ALPHAI TRADER — {_now_str:<34}║")
    print(f"║  Project age: {_days_old} days  (live since Jun 02, 2026)       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Load models
    try:
        with open(os.path.join(MODELS_DIR, "model_crypto.pkl"), "rb") as f:
            crypto_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "model_stocks.pkl"), "rb") as f:
            stocks_model = pickle.load(f)
    except Exception as e:
        print(f"  Could not load models: {e}")
        return

    vix, spy_ret, spy_close, btc_eth = load_macro()
    sentiment_scores, fear_greed     = load_sentiment_scores()
    portfolio                        = load_portfolio()

    from datetime import date as _date2
    _days_old2 = (_date2.today() - _date2(2026, 6, 2)).days
    _fg_label = "EXTREME FEAR" if fear_greed <= 25 else "FEAR" if fear_greed <= 40 else "NEUTRAL" if fear_greed <= 60 else "GREED" if fear_greed <= 75 else "EXTREME GREED"
    print(f"  📅 Day {_days_old2} of operation  |  Market: {_fg_label} ({fear_greed}/100)")
    print(f"  💵 Cash: ${portfolio['cash']:,.2f}  |  Positions: {len(portfolio['positions'])}/{MAX_POSITIONS}\n")

    # Generate signals for all assets
    signals = []
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

        sent  = sentiment_scores.get(symbol, 0.0)
        signal, prob = get_signal(model, feats, sentiment=sent)

        if signal != "HOLD" and prob >= MIN_CONFIDENCE:
            signals.append({
                "symbol":     symbol,
                "asset_type": asset_type,
                "signal":     signal,
                "prob":       prob,
                "price":      feats["price"],
            })

    # Sort by confidence — execute highest confidence first
    signals.sort(key=lambda x: x["prob"], reverse=True)

    print(f"  {len(signals)} actionable signals above {MIN_CONFIDENCE*100:.0f}% confidence\n")

    if signals:
        executed = execute_trades(signals, portfolio)
        print(f"\n  Executed {len(executed)} trades")
    else:
        print("  No trades executed — no signals above confidence threshold")

    save_portfolio(portfolio)
    print()
    performance_report(portfolio)

if __name__ == "__main__":
    main()
