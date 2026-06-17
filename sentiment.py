#!/usr/bin/env python3
"""
sentiment.py — Pulls Fear & Greed index and news sentiment for all assets
Uses Claude API for AI-powered news scoring
"""

import os, json, requests
from datetime import datetime
import anthropic

PROJECT_DIR    = "/root/alphai"
SENTIMENT_FILE = os.path.join(PROJECT_DIR, "logs", "sentiment.json")

CRYPTO = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","ATOM"]
STOCKS = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]

SEARCH_TERMS = {
    "BTC":  "Bitcoin", "XRP":  "XRP Ripple", "SOL":  "Solana",
    "LINK": "Chainlink", "HBAR": "Hedera HBAR", "XLM":  "Stellar XLM",
    "ADA":  "Cardano", "DOT":  "Polkadot", "AVAX": "Avalanche AVAX",
    "ATOM": "Cosmos ATOM", "LMT":  "Lockheed Martin", "ABTC": "American Bitcoin Corp",
    "PFE":  "Pfizer stock", "ORCL": "Oracle stock", "AAPL": "Apple stock",
    "NVDA": "Nvidia stock", "MSFT": "Microsoft stock", "AMZN": "Amazon stock",
    "JPM":  "JPMorgan stock", "SPY":  "S&P 500", "QQQ":  "Nasdaq QQQ",
}

def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()
        value = int(data["data"][0]["value"])
        label = data["data"][0]["value_classification"]
        return value, label
    except Exception as e:
        print(f"  Fear & Greed error: {e}")
        return 50, "Neutral"

def get_news_headlines(symbol):
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

def score_headlines(headlines, symbol):
    if not headlines:
        return 0.0, "LOW"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"  No API key found for {symbol}")
        return 0.0, "LOW"

    client = anthropic.Anthropic(api_key=api_key)
    headlines_text = "\n".join(f"- {h}" for h in headlines)

    prompt = f"""You are a financial news analyst scoring headlines for {symbol}.

Headlines:
{headlines_text}

Respond with ONLY a JSON object, no markdown, no explanation:
{{"score": 0.0, "risk_label": "LOW", "reason": "one sentence"}}

score: -1.0 = extremely bearish, 0.0 = neutral, 1.0 = extremely bullish
risk_label: LOW, MEDIUM, HIGH, or EXTREME"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        score = max(-1.0, min(1.0, float(result.get("score", 0.0))))
        risk_label = result.get("risk_label", "LOW")
        if risk_label not in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
            risk_label = "LOW"
        return score, risk_label
    except Exception as e:
        print(f"  Claude scoring error for {symbol}: {e}")
        return 0.0, "LOW"

def get_all_sentiment():
    print("\n  Running sentiment analysis (Claude AI scoring)...\n")

    fg_score, fg_label = get_fear_greed()
    print(f"  Fear & Greed Index: {fg_score}/100 — {fg_label}")
    fg_normalized = round((fg_score - 50) / 50, 4)

    results = {
        "timestamp":        datetime.now().isoformat(),
        "fear_greed":       fg_score,
        "fear_greed_label": fg_label,
        "fear_greed_norm":  fg_normalized,
        "assets":           {}
    }

    print(f"\n  {'SYMBOL':<8} {'HEADLINES':<12} {'RISK':<10} {'SCORE'}")
    print("  " + "-"*55)

    for symbol in CRYPTO + STOCKS:
        headlines = get_news_headlines(symbol)
        score, risk_label = score_headlines(headlines, symbol)
        count = len(headlines)

        if symbol in CRYPTO:
            combined = round((score + fg_normalized) / 2, 4)
        else:
            combined = score

        icon = "🟢" if combined > 0.1 else "🔴" if combined < -0.1 else "⬜"
        risk_icon = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🔴", "EXTREME": "💀"}.get(risk_label, "⬜")
        print(f"  {symbol:<8} {count:<12} {risk_icon} {risk_label:<8} {icon} {combined:+.4f}")

        results["assets"][symbol] = {
            "headline_count": count,
            "headline_score": score,
            "combined_score": combined,
            "risk_label":     risk_label,
            "headlines":      headlines[:3],
        }

    with open(SENTIMENT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to logs/sentiment.json")
    return results

def load_sentiment():
    if not os.path.exists(SENTIMENT_FILE):
        return None
    with open(SENTIMENT_FILE) as f:
        return json.load(f)

def get_asset_sentiment(symbol):
    data = load_sentiment()
    if not data:
        return 0.0, 50, "LOW"
    asset = data.get("assets", {}).get(symbol, {})
    return (
        asset.get("combined_score", 0.0),
        data.get("fear_greed", 50),
        asset.get("risk_label", "LOW")
    )

if __name__ == "__main__":
    get_all_sentiment()
