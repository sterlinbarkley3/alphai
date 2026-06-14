from core.market import get_price
from core.strategies import analyze_asset
from wallet import show_wallet
from assets import CRYPTO_ASSETS, STOCK_ASSETS
from datetime import datetime

# Assets to show in terminal (bot still tracks ALL assets)
DISPLAY_ASSETS = ["XRP", "LMT", "BTC", "PFE"]

print("=" * 50)
print("        AI TRADING SYSTEM ONLINE")
print("=" * 50)

all_assets = (
    [(s, "crypto") for s in CRYPTO_ASSETS] +
    [(s, "stock") for s in STOCK_ASSETS]
)

# Make sure BTC is included even though its not in crypto list
if "BTC" not in CRYPTO_ASSETS:
    all_assets = [("BTC", "crypto")] + all_assets

for symbol, asset_type in all_assets:

    price = get_price(symbol, asset_type)

    if price is None:
        continue

    analysis = analyze_asset(symbol)

    if analysis["risk"] == "HIGH":
        decision = "HOLD"
    elif analysis["trend"] == "UP" and analysis["confidence"] >= 50:
        decision = "BUY"
    elif analysis["trend"] == "DOWN" and analysis["confidence"] >= 50:
        decision = "SELL"
    else:
        decision = "HOLD"

    with open("decision_log.txt", "a") as log:
        log.write(
            f"{datetime.now()},"
            f"{symbol},"
            f"{price},"
            f"{decision},"
            f"{analysis['trend']},"
            f"{analysis['confidence']},"
            f"{analysis['risk']},"
            f"{analysis['momentum']},"
            f"{analysis['volatility']},"
            f"{analysis['week_prediction']},"
            f"{analysis['month_prediction']},"
            f"{analysis['year_prediction']},"
            f"{analysis['reason']}\n"
        )

    if symbol in DISPLAY_ASSETS:
        print(f"\n--- {symbol} ---")
        if price < 100:
            print(f"Price:         ${price:,.4f}")
        else:
            print(f"Price:         ${price:,.2f}")
        print(f"Trend:         {analysis['trend']}")
        print(f"Confidence:    {analysis['confidence']}%")
        print(f"Risk:          {analysis['risk']}")
        print(f"Reason:        {analysis['reason']}")
        print(f"1 Week:        {analysis['week_prediction']}%")
        print(f"1 Month:       {analysis['month_prediction']}%")
        print(f"1 Year:        {analysis['year_prediction']}%")
        print(f"Decision:      {decision}")

print("\n" + "=" * 50)
print("           WALLET STATUS")
print("=" * 50)
show_wallet(0)
