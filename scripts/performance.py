import os

LOG_FILE = "decision_log.txt"


def read_log():
    trades = []
    if not os.path.exists(LOG_FILE):
        print("No decision log found yet. Run main.py a few times first.")
        return trades

    with open(LOG_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                entry = {
                    "time": parts[0],
                    "price": float(parts[1]),
                    "decision": parts[2],
                    "trend": parts[3],
                    "confidence": float(parts[4]),
                    "risk": parts[5],
                    "momentum": float(parts[6]),
                    "volatility": float(parts[7]),
                    "week": float(parts[8]),
                    "month": float(parts[9]),
                    "year": float(parts[10]),
                }
                trades.append(entry)
            except (ValueError, IndexError):
                continue

    return trades


def analyze_performance():
    trades = read_log()

    if not trades:
        return

    total = len(trades)
    buys = [t for t in trades if t["decision"] == "BUY"]
    sells = [t for t in trades if t["decision"] == "SELL"]
    holds = [t for t in trades if t["decision"] == "HOLD"]

    print("=" * 40)
    print("       PERFORMANCE REPORT")
    print("=" * 40)
    print(f"Total Decisions:      {total}")
    print(f"BUY Decisions:        {len(buys)}")
    print(f"SELL Decisions:       {len(sells)}")
    print(f"HOLD Decisions:       {len(holds)}")
    print("-" * 40)

    if buys:
        avg_buy_confidence = sum(t["confidence"] for t in buys) / len(buys)
        print(f"Avg BUY Confidence:   {round(avg_buy_confidence, 2)}%")

    if sells:
        avg_sell_confidence = sum(t["confidence"] for t in sells) / len(sells)
        print(f"Avg SELL Confidence:  {round(avg_sell_confidence, 2)}%")

    high_risk = [t for t in trades if t["risk"] == "HIGH"]
    med_risk = [t for t in trades if t["risk"] == "MEDIUM"]
    low_risk = [t for t in trades if t["risk"] == "LOW"]

    print("-" * 40)
    print(f"HIGH Risk Periods:    {len(high_risk)}")
    print(f"MEDIUM Risk Periods:  {len(med_risk)}")
    print(f"LOW Risk Periods:     {len(low_risk)}")
    print("-" * 40)

    prices = [t["price"] for t in trades]
    print(f"Lowest BTC Seen:      ${min(prices):,.2f}")
    print(f"Highest BTC Seen:     ${max(prices):,.2f}")
    print(f"Price Range:          ${max(prices) - min(prices):,.2f}")
    print("-" * 40)

    up_trends = [t for t in trades if t["trend"] == "UP"]
    down_trends = [t for t in trades if t["trend"] == "DOWN"]
    sideways = [t for t in trades if t["trend"] == "SIDEWAYS"]

    print(f"UP Trends:            {len(up_trends)}")
    print(f"DOWN Trends:          {len(down_trends)}")
    print(f"SIDEWAYS Trends:      {len(sideways)}")
    print("=" * 40)
    print("NOTE: Win/loss tracking will activate")
    print("once buy and sell pairs are matched.")
    print("=" * 40)


if __name__ == "__main__":
    analyze_performance()
