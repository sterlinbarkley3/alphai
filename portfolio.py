import json
import os
from datetime import datetime

PORTFOLIO_FILE = "portfolio.json"
STARTING_CASH = 1_000_000.0

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "positions": {}, "history": []}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

def buy_asset(symbol, price, amount_usd=None, shares=None):
    portfolio = load_portfolio()
    cash = portfolio["cash"]
    if amount_usd is not None:
        cost = amount_usd
        qty = cost / price
    elif shares is not None:
        qty = shares
        cost = qty * price
    else:
        return {"success": False, "message": "No amount specified"}
    if cost > cash:
        return {"success": False, "message": f"Insufficient cash. You have ${cash:,.2f}"}
    if cost <= 0:
        return {"success": False, "message": "Amount must be greater than 0"}
    portfolio["cash"] -= cost
    if symbol in portfolio["positions"]:
        existing = portfolio["positions"][symbol]
        total_qty = existing["qty"] + qty
        total_cost = existing["avg_price"] * existing["qty"] + cost
        portfolio["positions"][symbol] = {"qty": total_qty, "avg_price": total_cost / total_qty, "total_invested": existing["total_invested"] + cost}
    else:
        portfolio["positions"][symbol] = {"qty": qty, "avg_price": price, "total_invested": cost}
    portfolio["history"].append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "action": "BUY", "symbol": symbol, "qty": round(qty, 6), "price": price, "total": round(cost, 2)})
    save_portfolio(portfolio)
    return {"success": True, "message": f"Bought {round(qty, 6)} {symbol} for ${cost:,.2f}"}

def sell_asset(symbol, price, amount_usd=None, shares=None, sell_all=False):
    portfolio = load_portfolio()
    if symbol not in portfolio["positions"]:
        return {"success": False, "message": f"You don't own any {symbol}"}
    position = portfolio["positions"][symbol]
    if sell_all:
        qty = position["qty"]
    elif shares is not None:
        qty = shares
    elif amount_usd is not None:
        qty = amount_usd / price
    else:
        return {"success": False, "message": "No amount specified"}
    if qty > position["qty"]:
        return {"success": False, "message": f"You only own {round(position['qty'], 6)} {symbol}"}
    proceeds = qty * price
    portfolio["cash"] += proceeds
    if qty >= position["qty"]:
        del portfolio["positions"][symbol]
    else:
        position["qty"] -= qty
        portfolio["positions"][symbol] = position
    portfolio["history"].append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "action": "SELL", "symbol": symbol, "qty": round(qty, 6), "price": price, "total": round(proceeds, 2)})
    save_portfolio(portfolio)
    return {"success": True, "message": f"Sold {round(qty, 6)} {symbol} for ${proceeds:,.2f}"}

def get_portfolio_summary(current_prices):
    portfolio = load_portfolio()
    cash = portfolio["cash"]
    positions = portfolio["positions"]
    total_invested = 0
    total_current = 0
    position_details = []
    for symbol, pos in positions.items():
        current_price = current_prices.get(symbol, pos["avg_price"])
        current_value = pos["qty"] * current_price
        invested = pos["total_invested"]
        gain_loss = current_value - invested
        gain_loss_pct = (gain_loss / invested * 100) if invested > 0 else 0
        total_invested += invested
        total_current += current_value
        position_details.append({"symbol": symbol, "qty": round(pos["qty"], 6), "avg_price": round(pos["avg_price"], 4), "current_price": round(current_price, 4), "current_value": round(current_value, 2), "invested": round(invested, 2), "gain_loss": round(gain_loss, 2), "gain_loss_pct": round(gain_loss_pct, 2)})
    total_portfolio = cash + total_current
    total_gain_loss = total_current - total_invested
    total_gain_loss_pct = (total_gain_loss / total_invested * 100) if total_invested > 0 else 0
    return {"cash": round(cash, 2), "total_invested": round(total_invested, 2), "total_current": round(total_current, 2), "total_portfolio": round(total_portfolio, 2), "total_gain_loss": round(total_gain_loss, 2), "total_gain_loss_pct": round(total_gain_loss_pct, 2), "positions": position_details, "history": portfolio["history"][-20:]}
