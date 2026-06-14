from strategies import analyze_asset
from assets import CRYPTO_ASSETS, STOCK_ASSETS


def score_asset(symbol):
    analysis = analyze_asset(symbol)

    if analysis["trend"] == "UNKNOWN":
        return {
            "symbol": symbol,
            "score": 0,
            "signal": "WAIT",
            "reason": analysis["reason"],
            "analysis": analysis
        }

    score = 0
    reasons = []

    # Trend score
    if analysis["trend"] == "UP":
        score += 30
        reasons.append("trend up")
    elif analysis["trend"] == "DOWN":
        score -= 30
        reasons.append("trend down")

    # Confidence score
    if analysis["confidence"] >= 70:
        score += 30
        reasons.append("high confidence")
    elif analysis["confidence"] >= 50:
        score += 15
        reasons.append("moderate confidence")
    elif analysis["confidence"] >= 30:
        score += 5
        reasons.append("low confidence")
    else:
        score -= 10
        reasons.append("very low confidence")

    # Momentum score
    if analysis["momentum"] > 0:
        score += 20
        reasons.append("positive momentum")
    else:
        score -= 20
        reasons.append("negative momentum")

    # Risk penalty
    if analysis["risk"] == "HIGH":
        score -= 40
        reasons.append("high risk penalty")
    elif analysis["risk"] == "MEDIUM":
        score -= 10
        reasons.append("medium risk penalty")
    else:
        score += 10
        reasons.append("low risk bonus")

    # Projection bonus
    if analysis["week_prediction"] > 2:
        score += 10
        reasons.append("strong week projection")
    elif analysis["week_prediction"] < -2:
        score -= 10
        reasons.append("weak week projection")

    # Determine signal
    if score >= 60:
        signal = "STRONG BUY"
    elif score >= 30:
        signal = "BUY"
    elif score <= -60:
        signal = "STRONG SELL"
    elif score <= -30:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol": symbol,
        "score": score,
        "signal": signal,
        "reason": ", ".join(reasons),
        "analysis": analysis
    }


def run_brain():
    all_assets = (
        [(s, "crypto") for s in CRYPTO_ASSETS] +
        [(s, "stock") for s in STOCK_ASSETS]
    )

    if "BTC" not in CRYPTO_ASSETS:
        all_assets = [("BTC", "crypto")] + all_assets

    results = []

    for symbol, asset_type in all_assets:
        result = score_asset(symbol)
        results.append(result)

    # Sort by score highest to lowest
    results.sort(key=lambda x: x["score"], reverse=True)

    print("=" * 50)
    print("         BRAIN ANALYSIS REPORT")
    print("=" * 50)

    print("\n TOP OPPORTUNITIES")
    print("-" * 50)
    top = [r for r in results if r["signal"] in ("STRONG BUY", "BUY")]
    if top:
        for r in top:
            print(f"{r['symbol']:8} Score: {r['score']:+d}  Signal: {r['signal']}")
            print(f"         {r['reason']}")
    else:
        print("No strong buy signals right now.")

    print("\n ASSETS TO AVOID")
    print("-" * 50)
    bottom = [r for r in results if r["signal"] in ("STRONG SELL", "SELL")]
    if bottom:
        for r in bottom:
            print(f"{r['symbol']:8} Score: {r['score']:+d}  Signal: {r['signal']}")
            print(f"         {r['reason']}")
    else:
        print("No strong sell signals right now.")

    print("\n HOLDING")
    print("-" * 50)
    holding = [r for r in results if r["signal"] == "HOLD"]
    for r in holding:
        print(f"{r['symbol']:8} Score: {r['score']:+d}  Signal: HOLD")

    print("\n WAITING FOR DATA")
    print("-" * 50)
    waiting = [r for r in results if r["signal"] == "WAIT"]
    for r in waiting:
        print(f"{r['symbol']:8} {r['reason']}")

    print("=" * 50)


if __name__ == "__main__":
    run_brain()
