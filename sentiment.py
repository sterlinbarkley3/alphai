#!/usr/bin/env python3
"""
sentiment.py — Pulls Fear & Greed index and news sentiment for all assets
Outputs a sentiment score per asset used as a feature in the ML model
"""

import os, json, requests
from datetime import datetime

PROJECT_DIR  = "/Users/mythreeboyz/pythonuh/ai trader"
SENTIMENT_FILE = os.path.join(PROJECT_DIR, "logs", "sentiment.json")

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

# Map symbols to search terms for news
SEARCH_TERMS = {
    "BTC":  "Bitcoin", "XRP":  "XRP Ripple", "SOL":  "Solana",
    "LINK": "Chainlink", "HBAR": "Hedera HBAR", "XLM":  "Stellar XLM",
    "ADA":  "Cardano", "DOT":  "Polkadot", "AVAX": "Avalanche AVAX",
    "ATOM": "Cosmos ATOM", "LMT":  "Lockheed Martin", "ABTC": "American Bitcoin Corp",
    "PFE":  "Pfizer stock", "ORCL": "Oracle stock", "AAPL": "Apple stock",
    "NVDA": "Nvidia stock", "MSFT": "Microsoft stock", "AMZN": "Amazon stock",
    "JPM":  "JPMorgan stock", "SPY":  "S&P 500", "QQQ":  "Nasdaq QQQ",
}

# ── Fear & Greed Index ───────────────────────────────────────────────────────

def get_fear_greed():
    """
    Pulls the crypto Fear & Greed index (0=extreme fear, 100=extreme greed)
    Free API, no key needed
    """
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()
        value = int(data["data"][0]["value"])
        label = data["data"][0]["value_classification"]
        return value, label
    except Exception as e:
        print(f"  Fear & Greed error: {e}")
        return 50, "Neutral"

# ── News sentiment ───────────────────────────────────────────────────────────

def get_news_headlines(symbol):
    """
    Pulls recent headlines from Google News RSS
    No API key needed
    """
    import re as re2
    search = SEARCH_TERMS.get(symbol, symbol)
    try:
        url = f"https://news.google.com/rss/search?q={search}&hl=en-US&gl=US&ceid=US:en"
        r   = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        titles = re2.findall(r'<title>(.*?)</title>', r.text)
        headlines = [t.strip() for t in titles[1:8] if t.strip() and len(t.strip()) > 10]
        return headlines
    except Exception as e:
        return []

def score_headlines(headlines):
    """
    Simple keyword-based sentiment scoring
    Returns score from -1.0 (very negative) to +1.0 (very positive)
    """
    if not headlines:
        return 0.0

    positive_words = [
        "surge", "soar", "rally", "gain", "rise", "jump", "bull", "record",
        "high", "growth", "profit", "beat", "strong", "buy", "upgrade",
        "bullish", "breakout", "recover", "up", "positive", "boost",
        "outperform", "opportunity", "momentum", "milestone"
    ]
    negative_words = [
        "crash", "plunge", "fall", "drop", "loss", "bear", "low", "decline",
        "sell", "downgrade", "weak", "risk", "fear", "warning", "concern",
        "bearish", "breakdown", "down", "negative", "trouble", "miss",
        "underperform", "lawsuit", "fine", "cut", "layoff", "recession"
    ]

    total_score = 0.0
    for headline in headlines:
        h = headline.lower()
        pos = sum(1 for w in positive_words if w in h)
        neg = sum(1 for w in negative_words if w in h)
        if pos + neg > 0:
            total_score += (pos - neg) / (pos + neg)

    return round(total_score / len(headlines), 4)

# ── Main sentiment engine ────────────────────────────────────────────────────

def get_all_sentiment():
    print("\n  Running sentiment analysis...\n")

    # Fear & Greed
    fg_score, fg_label = get_fear_greed()
    print(f"  Fear & Greed Index: {fg_score}/100 — {fg_label}")

    # Normalize to -1 to +1
    fg_normalized = round((fg_score - 50) / 50, 4)

    results = {
        "timestamp":       datetime.now().isoformat(),
        "fear_greed":      fg_score,
        "fear_greed_label":fg_label,
        "fear_greed_norm": fg_normalized,
        "assets":          {}
    }

    print(f"\n  {'SYMBOL':<8} {'HEADLINES':<12} {'SENTIMENT':<12} {'SCORE'}")
    print("  " + "-"*50)

    for symbol in CRYPTO + STOCKS:
        headlines = get_news_headlines(symbol)
        score     = score_headlines(headlines)
        count     = len(headlines)

        # For crypto, also factor in fear & greed
        if symbol in CRYPTO:
            combined = round((score + fg_normalized) / 2, 4)
        else:
            combined = score

        icon = "🟢" if combined > 0.1 else "🔴" if combined < -0.1 else "⬜"
        print(f"  {symbol:<8} {count:<12} {icon} {combined:+.4f}   {headlines[0][:40] + '...' if headlines else 'no headlines'}")

        results["assets"][symbol] = {
            "headline_count": count,
            "headline_score": score,
            "combined_score": combined,
            "headlines":      headlines[:3],
        }

    # Save to disk
    with open(SENTIMENT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to logs/sentiment.json")
    return results

def load_sentiment():
    """Load most recent sentiment scores"""
    if not os.path.exists(SENTIMENT_FILE):
        return None
    with open(SENTIMENT_FILE) as f:
        return json.load(f)

def get_asset_sentiment(symbol):
    """Get sentiment score for a single asset — returns 0.0 if unavailable"""
    data = load_sentiment()
    if not data:
        return 0.0, 50
    asset = data.get("assets", {}).get(symbol, {})
    return asset.get("combined_score", 0.0), data.get("fear_greed", 50)

if __name__ == "__main__":
    get_all_sentiment()
