import os
import json
from datetime import datetime

WALLET_FILE = "wallet.json"

def load_wallet():
    if os.path.exists(WALLET_FILE):
        with open(WALLET_FILE, "r") as f:
            return json.load(f)
    return {"cash": 1000.0, "bitcoin": 0.0}

def save_wallet(cash, bitcoin):
    with open(WALLET_FILE, "w") as f:
        json.dump({"cash": cash, "bitcoin": bitcoin}, f)

def log_trade(action, price):
    with open("trades.txt", "a") as file:
        file.write(f"{action} at {price}\n")

def buy(price):
    wallet = load_wallet()
    cash = wallet["cash"]
    bitcoin = wallet["bitcoin"]
    if cash <= 0:
        print("Cannot buy — no cash available")
        return
    if bitcoin > 0:
        print("Cannot buy — already holding Bitcoin")
        return
    bitcoin = cash / price
    cash = 0.0
    save_wallet(cash, bitcoin)
    print("Bought Bitcoin")
    log_trade("BUY", price)

def sell(price):
    wallet = load_wallet()
    cash = wallet["cash"]
    bitcoin = wallet["bitcoin"]
    if bitcoin <= 0:
        print("Cannot sell — no Bitcoin to sell")
        return
    cash = bitcoin * price
    bitcoin = 0.0
    save_wallet(cash, bitcoin)
    print("Sold Bitcoin")
    log_trade("SELL", price)

def show_wallet(price):
    wallet = load_wallet()
    cash = wallet["cash"]
    bitcoin = wallet["bitcoin"]
    total = cash + (bitcoin * price)
    print("Cash:", cash)
    print("Bitcoin:", bitcoin)
    print("Wallet Value:", total)
