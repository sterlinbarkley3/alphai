from core.market import load_prices

def get_daily_prices(symbol):
    all_prices = load_prices(symbol)
    if len(all_prices) < 2:
        return all_prices
    daily = [all_prices[0]]
    for price in all_prices[1:]:
        if price != daily[-1]:
            daily.append(price)
    return daily

def analyze_asset(symbol):
    prices = get_daily_prices(symbol)

    if len(prices) < 10:
        return {
            "symbol": symbol,
            "price": 0,
            "trend": "UNKNOWN",
            "confidence": 0,
            "risk": "HIGH",
            "momentum": 0,
            "volatility": 0,
            "week_prediction": 0,
            "month_prediction": 0,
            "year_prediction": 0,
            "reason": "Not enough data yet"
        }

    prices = prices[-200:]

    current_price = prices[-1]
    short_ma = sum(prices[-5:]) / 5
    long_ma = sum(prices[-20:]) / 20
    momentum = current_price - prices[-5]
    volatility = max(prices[-20:]) - min(prices[-20:])
    trend_strength = ((short_ma - long_ma) / long_ma) * 100

    if short_ma > long_ma:
        trend = "UP"
    elif short_ma < long_ma:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"

    # Normalized scores — all percentage based so stocks and crypto are equal
    trend_score = abs(trend_strength) * 25
    momentum_score = abs(momentum) / current_price * 100 * 15
    volatility_penalty = (volatility / current_price * 100) * 2

    confidence = trend_score + momentum_score - volatility_penalty

    if confidence > 100:
        confidence = 100
    if confidence < 0:
        confidence = 0

    # Risk based on volatility as percentage of price
    vol_pct = volatility / current_price * 100
    if vol_pct > 15:
        risk = "HIGH"
    elif vol_pct > 5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    if risk == "HIGH":
        reason = "Volatility too high"
    elif trend == "UP" and confidence >= 50:
        reason = "Strong upward trend"
    elif trend == "DOWN" and confidence >= 50:
        reason = "Strong downward trend"
    elif trend == "UP":
        reason = "Trend up but low confidence"
    elif trend == "DOWN":
        reason = "Trend down but low confidence"
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

def analyze_market():
    return analyze_asset("BTC")
