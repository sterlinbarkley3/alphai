#!/usr/bin/env python3
"""
screener.py — True Dynamic Market Scanner for Alphai Trader
Scans the entire US market using yfinance's built-in screener.
No hardcoded lists — finds the best stocks and crypto daily.
Saves to logs/screener_picks.json for ai_brain_v2.py to read.
Runs at 7:45am ET via cron.
"""

import os, json
import yfinance as yf
from yfinance.screener.screener import screen
from yfinance.screener.query import EquityQuery
from datetime import datetime, timezone
import anthropic

PROJECT_DIR = "/root/alphai"
LOG_DIR     = os.path.join(PROJECT_DIR, "logs")
OUTPUT_FILE = os.path.join(LOG_DIR, "screener_picks.json")

CORE_STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]
CORE_CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]

CRYPTO_CANDIDATES = [
    "ETH","BNB","DOGE","SHIB","LTC","BCH","AAVE",
    "ARB","OP","INJ","FET","WIF","TIA",
    "SEI","JTO","PYTH","JUP"
]

MAX_STOCK_PICKS  = 10
MAX_CRYPTO_PICKS = 10


def scan_stocks() -> list:
    """
    Use yfinance screener to find top momentum stocks
    from the entire US market right now.
    """
    print("  Scanning full US market...")
    try:
        # High momentum + high volume stocks
        q = EquityQuery('and', [
            EquityQuery('gt', ['eodprice', 5]),
            EquityQuery('gt', ['dayvolume', 1000000]),
            EquityQuery('is-in', ['region', 'us']),
            EquityQuery('gt', ['percentchange', 1]),
        ])
        result = screen(q, sortField='dayvolume', sortAsc=False)
        quotes = result.get('quotes', [])
        print(f"  Found {result.get('total', 0)} qualifying stocks today")

        scored = []
        for q in quotes:
            symbol  = q.get('symbol', '')
            if not symbol or symbol in CORE_STOCKS:
                continue
            # Skip non-US or weird symbols
            if '.' in symbol or len(symbol) > 5:
                continue

            price      = q.get('regularMarketPrice', 0)
            change_pct = q.get('regularMarketChangePercent', 0)
            volume     = q.get('regularMarketVolume', 0)
            avg_vol    = q.get('averageDailyVolume3Month', 1)
            week52_chg = q.get('fiftyTwoWeekChangePercent', 0)
            mkt_cap    = q.get('marketCap', 0)

            vol_surge  = volume / avg_vol if avg_vol > 0 else 1.0

            # Composite score
            score = (
                (change_pct / 100) * 0.40 +
                (min(vol_surge, 5) / 5) * 0.35 +
                (week52_chg / 100) * 0.25
            )

            scored.append({
                "ticker":     symbol,
                "price":      round(price, 2),
                "change_pct": round(change_pct, 2),
                "vol_surge":  round(vol_surge, 2),
                "week52_chg": round(week52_chg, 2),
                "mkt_cap":    mkt_cap,
                "score":      round(score, 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:30]  # top 30 for AI review

    except Exception as e:
        print(f"  [ERROR] Stock scan failed: {e}")
        return []


def scan_crypto() -> list:
    """
    Score crypto candidates using yfinance.
    """
    print("  Scanning crypto candidates...")
    scored = []

    for ticker in CRYPTO_CANDIDATES:
        if ticker in CORE_CRYPTO:
            continue
        try:
            df = yf.download(f"{ticker}-USD", period="30d",
                           auto_adjust=True, progress=False)
            if df.empty or len(df) < 10:
                continue

            close     = df["Close"].squeeze()
            volume    = df["Volume"].squeeze()
            price     = float(close.iloc[-1])
            mom5      = (price - float(close.iloc[-5]))   / float(close.iloc[-5])
            mom20     = (price - float(close.iloc[-min(20, len(close)-1)])) / float(close.iloc[-min(20, len(close)-1)])
            avg_vol   = float(volume.tail(10).mean())
            vol_surge = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

            score = (mom20 * 0.40) + (mom5 * 0.35) + (min(vol_surge, 4) / 4 * 0.25)

            scored.append({
                "ticker":    ticker,
                "price":     round(price, 4),
                "mom5":      round(mom5 * 100, 2),
                "mom20":     round(mom20 * 100, 2),
                "vol_surge": round(vol_surge, 2),
                "score":     round(score, 4),
            })
        except:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:15]


def get_ai_verdict(candidates: list, asset_type: str) -> dict:
    """Ask Claude to review top candidates using news awareness."""
    if not candidates:
        return {}
    try:
        client  = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        tickers = ", ".join([c["ticker"] for c in candidates[:15]])

        prompt = f"""You are a trading analyst. Today is {datetime.now().strftime('%B %d, %Y')}.

Review these {asset_type} candidates: {tickers}

Based on recent news, earnings, macro trends, and market sentiment rate each:
+1 = bullish (good news, strong setup, worth buying today)
 0 = neutral
-1 = bearish (bad news, avoid)

Respond ONLY with valid JSON, no explanation, no markdown backticks:
{{"TICKER": score}}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  [WARN] AI verdict failed: {e}")
        return {}


def run():
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 54)
    print("  Alphai Dynamic Market Screener")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 54)

    # ── Scan ─────────────────────────────────────────────────
    print("\n[1/4] Scanning stocks...")
    stock_candidates = scan_stocks()

    print(f"\n[2/4] Scanning crypto...")
    crypto_candidates = scan_crypto()

    # ── AI verdict ────────────────────────────────────────────
    print(f"\n[3/4] Getting Claude AI verdict on top stocks...")
    stock_verdict = get_ai_verdict(stock_candidates[:15], "stock")

    print(f"[4/4] Getting Claude AI verdict on top crypto...")
    crypto_verdict = get_ai_verdict(crypto_candidates[:10], "crypto")

    # ── Apply AI boost ────────────────────────────────────────
    for s in stock_candidates:
        boost = stock_verdict.get(s["ticker"], 0)
        s["ai_boost"]    = boost
        s["final_score"] = round(s["score"] + (boost * 0.15), 4)

    for c in crypto_candidates:
        boost = crypto_verdict.get(c["ticker"], 0)
        c["ai_boost"]    = boost
        c["final_score"] = round(c["score"] + (boost * 0.15), 4)

    # ── Sort and pick winners ─────────────────────────────────
    stock_candidates.sort(key=lambda x: x.get("final_score", x["score"]), reverse=True)
    crypto_candidates.sort(key=lambda x: x.get("final_score", x["score"]), reverse=True)

    top_stocks = [s["ticker"] for s in stock_candidates[:MAX_STOCK_PICKS]]
    top_crypto = [c["ticker"] for c in crypto_candidates[:MAX_CRYPTO_PICKS]]

    final_stocks = list(dict.fromkeys(CORE_STOCKS + top_stocks))
    final_crypto = list(dict.fromkeys(CORE_CRYPTO + top_crypto))

    # ── Save ─────────────────────────────────────────────────
    result = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "stocks":      final_stocks,
        "crypto":      final_crypto,
        "screened": {
            "top_stocks": stock_candidates[:MAX_STOCK_PICKS],
            "top_crypto": crypto_candidates[:MAX_CRYPTO_PICKS],
        }
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n── Today's Picks ────────────────────────────────")
    print(f"  Top stocks  : {top_stocks}")
    print(f"  Top crypto  : {top_crypto}")
    print(f"\n  Stock details:")
    for s in stock_candidates[:MAX_STOCK_PICKS]:
        print(f"    {s['ticker']:<8} +{s['change_pct']:.1f}%  vol={s['vol_surge']:.1f}x  AI:{s.get('ai_boost',0):+d}  score={s.get('final_score', s['score']):.3f}")
    print(f"\n  Crypto details:")
    for c in crypto_candidates[:MAX_CRYPTO_PICKS]:
        print(f"    {c['ticker']:<8} mom5={c['mom5']:+.1f}%  vol={c['vol_surge']:.1f}x  AI:{c.get('ai_boost',0):+d}  score={c.get('final_score', c['score']):.3f}")
    print(f"\n  Total stocks tracked : {len(final_stocks)}")
    print(f"  Total crypto tracked : {len(final_crypto)}")
    print(f"  Saved → {OUTPUT_FILE}")
    print("=" * 54)

    return result


if __name__ == "__main__":
    run()
