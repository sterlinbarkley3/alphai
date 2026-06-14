from market import load_prices


def analyze_asset(symbol):

    prices = load_prices(symbol)

    if len(prices) < 10:
        return {
            "symbol": symbol,
            "trend": "UNKNOWN",
            "confidence": 0,
            "risk": "HIGH",
            "momentum": 0,
            "volatility": 0,
            "week_prediction": 0,
            "month_prediction": 0,
            "year_prediction": 0,
            "reason": f"Not enough data yet ({len(prices)}/10 prices collected)"
        }

    current_price = prices[-1]
    short_ma = sum(prices[-5:]) / 5
    long_ma = sum(prices[-10:]) / 10
    momentum = current_price - prices[-5]
    volatility = max(prices[-10:]) - min(prices[-10:])
    trend_strength = ((short_ma - long_ma) / long_ma) * 100

    if short_ma > long_ma:
        trend = "UP"
    elif short_ma < long_ma:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"

    momentum_score = abs(momentum) / current_price * 100
    volatility_penalty = volatility / current_price * 100

    confidence = (
        abs(trend_strength) * 10
        + momentum_score * 20
        - volatility_penalty * 15
    )

    if confidence > 100:
        confidence = 100
    if confidence < 0:
        confidence = 0

    if volatility / current_price > 0.05:
        risk = "HIGH"
    elif volatility / current_price > 0.02:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    if risk == "HIGH":
        reason = "Market volatility too high"
    elif trend == "UP" and confidence >= 50:
        reason = "Short MA above long MA with strong confidence"
    elif trend == "DOWN" and confidence >= 50:
        reason = "Short MA below long MA with strong confidence"
    elif trend == "UP":
        reason = "Trend is up but confidence too low"
    elif trend == "DOWN":
        reason = "Trend is down but confidence too low"
    else:
        reason = "No clear trend"

    week_prediction = round(trend_strength * 1.5, 2)
    month_prediction = round(trend_strength * 4, 2)
    year_prediction = round(trend_strength * 12, 2)

    return {
        "symbol": symbol,
        "price": current_price,
        "trend": trend,
        "confidence": round(confidence, 2),
        "risk": risk,
        "momentum": round(momentum, 2),
        "volatility": round(volatility, 2),
        "week_prediction": week_prediction,
        "month_prediction": month_prediction,
        "year_prediction": year_prediction,
        "reason": reason
    }


# Keep backward compatibility
def analyze_market():
    return analyze_asset("BTC")