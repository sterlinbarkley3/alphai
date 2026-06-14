#!/usr/bin/env python3
"""
sentiment_history.py — Logs daily sentiment scores to build historical dataset
Run daily via cron alongside sentiment.py
After 60 days this becomes a training feature
"""

import os, json, csv
from datetime import datetime

PROJECT_DIR   = "/Users/mythreeboyz/pythonuh/ai trader"
SENTIMENT_FILE = os.path.join(PROJECT_DIR, "logs", "sentiment.json")
HISTORY_FILE   = os.path.join(PROJECT_DIR, "logs", "sentiment_history.csv")

FIELDNAMES = ["date","symbol","fear_greed","fear_greed_norm","combined_score","headline_score"]

def log_today():
    if not os.path.exists(SENTIMENT_FILE):
        print("  No sentiment.json found — run sentiment.py first")
        return

    with open(SENTIMENT_FILE) as f:
        data = json.load(f)

    today = datetime.now().strftime("%Y-%m-%d")

    # Check if already logged today
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            existing = f.read()
        if today in existing:
            print(f"  Already logged sentiment for {today}")
            return

    rows = []
    for symbol, info in data["assets"].items():
        rows.append({
            "date":             today,
            "symbol":           symbol,
            "fear_greed":       data["fear_greed"],
            "fear_greed_norm":  data["fear_greed_norm"],
            "combined_score":   info["combined_score"],
            "headline_score":   info["headline_score"],
        })

    # Append to history file
    write_header = not os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  Logged sentiment for {today} — {len(rows)} assets")
    print(f"  Fear & Greed: {data['fear_greed']} ({data['fear_greed_label']})")

def get_sentiment_for_date(symbol, date_str):
    """Look up historical sentiment for a symbol on a specific date"""
    if not os.path.exists(HISTORY_FILE):
        return 0.0, 50

    with open(HISTORY_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["symbol"] == symbol and row["date"] == date_str:
                return float(row["combined_score"]), int(row["fear_greed"])

    return 0.0, 50

def get_latest_sentiment(symbol):
    """Get most recent sentiment score for a symbol"""
    if not os.path.exists(HISTORY_FILE):
        return 0.0, 50

    last_match = None
    with open(HISTORY_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["symbol"] == symbol:
                last_match = row

    if last_match:
        return float(last_match["combined_score"]), int(last_match["fear_greed"])
    return 0.0, 50

def show_history():
    if not os.path.exists(HISTORY_FILE):
        print("  No history yet.")
        return

    import pandas as pd
    df = pd.read_csv(HISTORY_FILE)
    print(f"\n  Sentiment history: {len(df)} records across {df['date'].nunique()} days")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n  Average sentiment by asset:")
    avg = df.groupby("symbol")["combined_score"].mean().sort_values(ascending=False)
    for sym, score in avg.items():
        icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⬜"
        print(f"    {sym:<8} {icon} {score:+.3f}")

if __name__ == "__main__":
    import sys
    if "--history" in sys.argv:
        show_history()
    else:
        log_today()
