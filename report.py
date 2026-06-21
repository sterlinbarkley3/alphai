#!/usr/bin/env python3
"""
report.py — Alphai Daily Dashboard
One clean read showing everything: screener picks, signals,
price targets, portfolio status, and win rate.
Run anytime: python3 report.py
"""

import os, json, pickle, csv, re
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import anthropic

PROJECT_DIR  = "/root/alphai"
OHLCV_DIR    = os.path.join(PROJECT_DIR, "ohlcv")
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
LOG_DIR      = os.path.join(PROJECT_DIR, "logs")

CORE_CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]

FEATURES = [
    "ma5_cross","ma10_cross","ma20_cross","price_vs_ma20","price_vs_ma50",
    "mom5","mom10","mom20","vol10","vol20","vol_ratio",
    "rsi","macd_cross","macd_hist","bb_position",
    "roc5","roc10","atr_pct","obv_pct","vol_trend","hl_position","score",
    "vix","vix_high","spy_ret_1d","spy_ret_5d","spy_trend","btc_eth_ratio","rel_strength",
]


def load_json(path, default={}):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return default


def load_macro():
    try:
        import yfinance as yf
        vix = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)["Close"].squeeze()
        spy = yf.download("SPY",  period="10d", auto_adjust=True, progress=False)["Close"].squeeze()
        btc = yf.download("BTC-USD", period="10d", auto_adjust=True, progress=False)["Close"].squeeze()
        eth = yf.download("ETH-USD", period="10d", auto_adjust=True, progress=False)["Close"].squeeze()
        spy_ret = spy.pct_change()
        return vix, spy_ret, spy, btc / eth
    except:
        return None, None, None, None


def compute_features(df, vix, spy_ret, spy_close, btc_eth):
    try:
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        vol   = df["Volume"].squeeze()

        ma5   = close.rolling(5).mean()
        ma10  = close.rolling(10).mean()
        ma20  = close.rolling(20).mean()
        ma50  = close.rolling(50).mean()

        feats = {
            "ma5_cross":      float(close.iloc[-1] > ma5.iloc[-1]),
            "ma10_cross":     float(close.iloc[-1] > ma10.iloc[-1]),
            "ma20_cross":     float(close.iloc[-1] > ma20.iloc[-1]),
            "price_vs_ma20":  float((close.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]),
            "price_vs_ma50":  float((close.iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1]) if ma50.iloc[-1] > 0 else 0,
            "mom5":           float(close.pct_change(5).iloc[-1]),
            "mom10":          float(close.pct_change(10).iloc[-1]),
            "mom20":          float(close.pct_change(20).iloc[-1]),
            "vol10":          float(vol.rolling(10).mean().iloc[-1]),
            "vol20":          float(vol.rolling(20).mean().iloc[-1]),
            "vol_ratio":      float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]),
            "rsi":            0.0,
            "macd_cross":     0.0,
            "macd_hist":      0.0,
            "bb_position":    0.0,
            "roc5":           float(close.pct_change(5).iloc[-1]),
            "roc10":          float(close.pct_change(10).iloc[-1]),
            "atr_pct":        float(((high - low).rolling(14).mean() / close).iloc[-1]),
            "obv_pct":        0.0,
            "vol_trend":      float(vol.pct_change(5).iloc[-1]),
            "hl_position":    float((close.iloc[-1] - low.iloc[-1]) / (high.iloc[-1] - low.iloc[-1] + 1e-9)),
            "score":          0.0,
            "vix":            float(vix.iloc[-1]) if vix is not None else 20.0,
            "vix_high":       float(vix.iloc[-1] > 25) if vix is not None else 0.0,
            "spy_ret_1d":     float(spy_ret.iloc[-1]) if spy_ret is not None else 0.0,
            "spy_ret_5d":     float(spy_ret.rolling(5).sum().iloc[-1]) if spy_ret is not None else 0.0,
            "spy_trend":      float(spy_close.iloc[-1] > spy_close.rolling(20).mean().iloc[-1]) if spy_close is not None else 1.0,
            "btc_eth_ratio":  float(btc_eth.iloc[-1]) if btc_eth is not None else 15.0,
            "rel_strength":   0.0,
        }

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        feats["rsi"] = float((100 - 100 / (1 + rs)).iloc[-1])

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9).mean()
        feats["macd_cross"] = float(macd.iloc[-1] > sig.iloc[-1])
        feats["macd_hist"]  = float(macd.iloc[-1] - sig.iloc[-1])

        # Bollinger
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        feats["bb_position"] = float((close.iloc[-1] - bb_mid.iloc[-1]) / (2 * bb_std.iloc[-1] + 1e-9))

        return pd.DataFrame([feats])[FEATURES]
    except:
        return None


def price_targets(df, signal, confidence):
    """
    Calculate realistic price targets using ATR-based projection.
    BUY  = project upward
    SELL = project downward
    """
    try:
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()

        price   = float(close.iloc[-1])
        atr     = float((high - low).rolling(14).mean().iloc[-1])
        atr_pct = atr / price

        # Confidence multiplier — higher confidence = bigger target
        conf_mult = 0.5 + confidence  # ranges 0.5 to 1.5

        direction = 1 if signal == "BUY" else -1

        week  = price * (1 + direction * atr_pct * 1.5 * conf_mult)
        month = price * (1 + direction * atr_pct * 4.0 * conf_mult)
        year  = price * (1 + direction * atr_pct * 15.0 * conf_mult)

        week_pct  = (week  - price) / price * 100
        month_pct = (month - price) / price * 100
        year_pct  = (year  - price) / price * 100

        return {
            "current": round(price, 4),
            "week":    round(week, 4),
            "month":   round(month, 4),
            "year":    round(year, 4),
            "week_pct":  round(week_pct, 1),
            "month_pct": round(month_pct, 1),
            "year_pct":  round(year_pct, 1),
        }
    except:
        return None


def get_ai_summary(signals_data: list) -> str:
    """Ask Claude for a plain English market summary."""
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        signals_str = "\n".join([
            f"{s['symbol']} — {s['signal']} ({s['conf']*100:.0f}% confidence, RSI {s['rsi']:.0f})"
            for s in signals_data[:8]
        ])

        prompt = f"""You are a friendly trading assistant explaining today's signals to a beginner investor.

Today's signals:
{signals_str}

Write a SHORT plain English summary (5-7 sentences max) explaining:
1. What the market looks like today overall
2. Which 2-3 signals look most interesting and why
3. Any risks to watch out for

Write like you're explaining to a smart 18-year-old who's new to investing. No jargon. Be direct and honest."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"AI summary unavailable: {e}"


def load_win_rate():
    """Load win rate from paper trades (correct structure: dict with 'signals' key)."""
    try:
        with open(os.path.join(LOG_DIR, "paper_trades.json")) as f:
            data = json.load(f)
        all_signals = data.get("signals", [])
        graded = [s for s in all_signals if s.get("graded") and s.get("correct") is not None]
        wins   = [s for s in graded if s.get("correct")]
        return len(wins), len(graded), len(all_signals)
    except Exception as e:
        print(f"  [WARN] Could not load win rate: {e}")
        return 0, 0, 0


def run():
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  ALPHAI DAILY REPORT — {datetime.now().strftime('%b %d %Y  %I:%M %p'):<37}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Regime ───────────────────────────────────────────────────────────
    regime_data = load_json(os.path.join(LOG_DIR, "regime.json"))
    regime      = regime_data.get("regime", "UNKNOWN")
    score       = regime_data.get("total_score", 0)
    blocked     = regime_data.get("rules", {}).get("buy_signals_blocked", False)
    signals_r   = regime_data.get("signals", {})
    vix_val     = signals_r.get("vix", {}).get("vix", "?")
    spy_price   = signals_r.get("spy", {}).get("spy_price", "?")

    regime_icon = "🟢" if regime == "BULL" else "🔴" if regime == "BEAR" else "🟡"
    print(f"\n{'─'*64}")
    print(f"  MARKET REGIME")
    print(f"{'─'*64}")
    print(f"  {regime_icon} Regime    : {regime}  (score: {score:+d})")
    print(f"  📊 VIX      : {vix_val}  {'⚠️  HIGH FEAR' if isinstance(vix_val, float) and vix_val > 25 else '✅ Calm'}")
    print(f"  📈 SPY      : ${spy_price}")
    print(f"  🚫 BUY Block: {'YES — No buys allowed today' if blocked else 'NO — Buys allowed'}")

    # ── Screener picks ────────────────────────────────────────────────────
    screener = load_json(os.path.join(LOG_DIR, "screener_picks.json"))
    print(f"\n{'─'*64}")
    print(f"  TODAY'S SCREENER PICKS  (what the AI chose from the whole market)")
    print(f"{'─'*64}")

    top_stocks = screener.get("screened", {}).get("top_stocks", [])
    top_crypto = screener.get("screened", {}).get("top_crypto", [])

    print(f"\n  📦 TOP STOCKS:")
    for s in top_stocks[:10]:
        ai  = s.get("ai_boost", 0)
        bar = "🟢" if ai > 0 else "🔴" if ai < 0 else "⬜"
        print(f"    {bar} {s['ticker']:<8} +{s.get('change_pct',0):.1f}% today  "
              f"vol={s.get('vol_surge',0):.1f}x avg  AI: {'Bullish' if ai > 0 else 'Bearish' if ai < 0 else 'Neutral'}")

    print(f"\n  🪙 TOP CRYPTO:")
    for c in top_crypto[:10]:
        ai  = c.get("ai_boost", 0)
        bar = "🟢" if ai > 0 else "🔴" if ai < 0 else "⬜"
        print(f"    {bar} {c['ticker']:<8} {c.get('mom5',0):+.1f}% (5d)  "
              f"vol={c.get('vol_surge',0):.1f}x avg  AI: {'Bullish' if ai > 0 else 'Bearish' if ai < 0 else 'Neutral'}")

    # ── Signals + Price targets ───────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  TODAY'S SIGNALS + PRICE TARGETS")
    print(f"{'─'*64}")

    # Load models and macro
    try:
        with open(os.path.join(MODELS_DIR, "model_crypto.pkl"), "rb") as f:
            crypto_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "model_stocks.pkl"), "rb") as f:
            stocks_model = pickle.load(f)
    except Exception as e:
        print(f"  Could not load models: {e}")
        return

    vix, spy_ret, spy_close, btc_eth = load_macro()
    active_crypto = screener.get("crypto", CORE_CRYPTO)
    active_stocks = screener.get("stocks", [])

    signals_data = []
    buy_signals  = []
    sell_signals = []

    for symbol in active_crypto + active_stocks:
        is_crypto = symbol in active_crypto
        model     = crypto_model if is_crypto else stocks_model

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

        prob    = float(model.predict_proba(feats)[0][1])
        signal  = "BUY" if prob >= 0.58 else "SELL" if prob <= 0.42 else "HOLD"
        rsi     = feats["rsi"].iloc[0] if hasattr(feats["rsi"], "iloc") else float(feats["rsi"])
        targets = price_targets(df, signal, prob)
        price   = targets["current"] if targets else 0

        entry = {
            "symbol":  symbol,
            "signal":  signal,
            "conf":    prob,
            "rsi":     float(feats["rsi"].iloc[0]) if hasattr(feats["rsi"], "iloc") else 0,
            "targets": targets,
            "type":    "CRYPTO" if is_crypto else "STOCK",
        }
        signals_data.append(entry)

        if signal == "BUY":
            buy_signals.append(entry)
        elif signal == "SELL":
            sell_signals.append(entry)

    # Print BUY signals
    if buy_signals:
        print(f"\n  🟢 BUY SIGNALS ({len(buy_signals)}):")
        for s in sorted(buy_signals, key=lambda x: x["conf"], reverse=True):
            t = s["targets"]
            print(f"\n    {s['symbol']} ({s['type']})")
            print(f"      Confidence : {s['conf']*100:.1f}%  |  RSI: {s['rsi']:.0f}")
            if t:
                print(f"      Price now  : ${t['current']:,.4f}")
                print(f"      1 week     : ${t['week']:,.4f}  ({t['week_pct']:+.1f}%)")
                print(f"      1 month    : ${t['month']:,.4f}  ({t['month_pct']:+.1f}%)")
                print(f"      1 year     : ${t['year']:,.4f}  ({t['year_pct']:+.1f}%)")
    else:
        print(f"\n  🟢 BUY SIGNALS: None today")

    # Print SELL signals
    if sell_signals:
        print(f"\n  🔴 SELL SIGNALS ({len(sell_signals)}):")
        for s in sorted(sell_signals, key=lambda x: x["conf"], reverse=True)[:5]:
            t = s["targets"]
            print(f"\n    {s['symbol']} ({s['type']})")
            print(f"      Confidence : {s['conf']*100:.1f}%  |  RSI: {s['rsi']:.0f}")
            if t:
                print(f"      Price now  : ${t['current']:,.4f}")
                print(f"      1 week     : ${t['week']:,.4f}  ({t['week_pct']:+.1f}%)")
                print(f"      1 month    : ${t['month']:,.4f}  ({t['month_pct']:+.1f}%)")

    # ── Portfolio ─────────────────────────────────────────────────────────
    portfolio = load_json(os.path.join(LOG_DIR, "auto_portfolio.json"))
    cash      = portfolio.get("cash", 10000)
    positions = portfolio.get("positions", {})
    start     = portfolio.get("starting_cash", 10000)

    total_value = cash
    print(f"\n{'─'*64}")
    print(f"  PORTFOLIO STATUS")
    print(f"{'─'*64}")
    print(f"  💵 Cash      : ${cash:,.2f}")

    if positions:
        print(f"  📂 Positions :")
        for sym, pos in positions.items():
            cost  = pos.get("cost_basis", 0)
            entry = pos.get("entry_price", 0)
            qty   = pos.get("qty", 0)

            path = os.path.join(OHLCV_DIR, f"{sym}_daily.parquet")
            try:
                df    = pd.read_parquet(path)
                curr  = float(df["Close"].squeeze().iloc[-1])
                value = curr * qty
                pnl   = value - cost
                pct   = (pnl / cost) * 100
                total_value += value
                pnl_icon = "📈" if pnl >= 0 else "📉"
                print(f"    {pnl_icon} {sym:<8} {qty:.4f} shares @ ${entry:.4f}  "
                      f"value=${value:,.2f}  P&L: ${pnl:+.2f} ({pct:+.1f}%)")
            except:
                total_value += cost
                print(f"    {sym:<8} cost=${cost:.2f}")
    else:
        print(f"  📂 Positions : None")

    total_return = (total_value - start) / start * 100
    print(f"  💰 Total     : ${total_value:,.2f}  ({total_return:+.2f}% return)")

    # ── Win rate ──────────────────────────────────────────────────────────
    wins, graded, total = load_win_rate()
    win_rate = (wins / graded * 100) if graded > 0 else 0

    print(f"\n{'─'*64}")
    print(f"  TRACK RECORD")
    print(f"{'─'*64}")
    print(f"  🏆 Win Rate  : {win_rate:.1f}%  ({wins}W / {graded-wins}L of {graded} graded)")
    print(f"  📋 Total signals logged: {total}")

    # ── AI plain English summary ──────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  CLAUDE'S TAKE — Plain English Summary")
    print(f"{'─'*64}")
    summary = get_ai_summary(signals_data)
    # Strip markdown formatting (bold, headers) for clean terminal display
    summary = re.sub(r'\*\*(.*?)\*\*', r'\1', summary)  # remove **bold**
    summary = re.sub(r'^#+\s*', '', summary, flags=re.MULTILINE)  # remove # headers
    summary = summary.replace('*', '')  # remove stray asterisks
    # Word wrap at 60 chars
    words = summary.split()
    line  = "  "
    for word in words:
        if len(line) + len(word) > 62:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)

    print(f"\n{'═'*64}")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    run()
