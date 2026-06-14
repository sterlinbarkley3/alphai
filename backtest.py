#!/usr/bin/env python3
"""
backtest.py — Sterlin's AI Trading Dashboard Backtester
Replays historical price data through the strategy engine and measures performance.

Usage:
    python3 backtest.py                    # backtest all assets
    python3 backtest.py --symbol BTC       # single asset
    python3 backtest.py --top 5            # show top 5 performers
    python3 backtest.py --export           # save results to backtest_results.json
"""

import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# ── Config ──────────────────────────────────────────────────────────────────
SHORT_MA   = 5
LONG_MA    = 20
MIN_PRICES = 25          # minimum data points needed to generate a signal
STEP       = 1           # check every N prices (1 = every tick, 5 = every 5th, etc.)
START_CASH = 10_000.0    # starting cash per asset simulation

# ── Asset lists (mirrors assets.py) ─────────────────────────────────────────
CRYPTO_ASSETS = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","MATIC","ATOM"]
STOCK_ASSETS  = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]
ALL_ASSETS    = CRYPTO_ASSETS + STOCK_ASSETS


# ── Data loader ──────────────────────────────────────────────────────────────
def load_prices(symbol: str) -> list[float]:
    path = os.path.join(PROJECT_DIR, f"history_{symbol}.txt")
    if not os.path.isfile(path):
        return []
    prices = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    prices.append(float(line))
                except ValueError:
                    pass
    return prices


# ── Strategy (mirrors strategies.py logic) ──────────────────────────────────
def moving_average(prices: list[float], window: int) -> float:
    if len(prices) < window:
        return 0.0
    return sum(prices[-window:]) / window

def get_signal(prices: list[float]) -> str:
    """Reproduce the brain.py scoring logic on a price window."""
    if len(prices) < MIN_PRICES:
        return "HOLD"

    short_ma  = moving_average(prices, SHORT_MA)
    long_ma   = moving_average(prices, LONG_MA)
    momentum  = (prices[-1] - prices[-SHORT_MA]) / prices[-SHORT_MA] * 100 if prices[-SHORT_MA] != 0 else 0
    recent    = prices[-LONG_MA:]
    volatility = (max(recent) - min(recent)) / min(recent) * 100 if min(recent) != 0 else 0

    score = 0

    # MA crossover
    if short_ma > long_ma:
        score += 2
    elif short_ma < long_ma:
        score -= 2

    # Momentum
    if momentum > 2:
        score += 2
    elif momentum > 0.5:
        score += 1
    elif momentum < -2:
        score -= 2
    elif momentum < -0.5:
        score -= 1

    # Volatility penalty
    if volatility > 10:
        score -= 1

    if score >= 4:
        return "STRONG BUY"
    elif score >= 2:
        return "BUY"
    elif score <= -4:
        return "STRONG SELL"
    elif score <= -2:
        return "SELL"
    else:
        return "HOLD"


# ── Backtester ───────────────────────────────────────────────────────────────
def backtest_asset(symbol: str, prices: list[float]) -> dict:
    cash     = START_CASH
    holdings = 0.0
    trades   = []
    last_signal = "HOLD"

    for i in range(MIN_PRICES, len(prices), STEP):
        window = prices[:i]
        price  = prices[i - 1]
        signal = get_signal(window)

        # Only act on signal changes to avoid thrashing
        if signal == last_signal:
            continue

        if signal in ("BUY", "STRONG BUY") and cash > 0:
            quantity  = cash / price
            holdings += quantity
            cash      = 0.0
            trades.append({
                "type":     "BUY",
                "signal":   signal,
                "price":    price,
                "quantity": quantity,
                "index":    i,
            })

        elif signal in ("SELL", "STRONG SELL") and holdings > 0:
            proceeds  = holdings * price
            pnl       = proceeds - (trades[-1]["price"] * holdings) if trades else 0
            cash     += proceeds
            trades.append({
                "type":     "SELL",
                "signal":   signal,
                "price":    price,
                "quantity": holdings,
                "pnl":      pnl,
                "index":    i,
            })
            holdings = 0.0

        last_signal = signal

    # Close any open position at last price
    final_price = prices[-1]
    if holdings > 0:
        proceeds = holdings * final_price
        pnl      = proceeds - (trades[-1]["price"] * holdings) if trades else 0
        cash    += proceeds
        trades.append({
            "type":     "CLOSE",
            "signal":   "END",
            "price":    final_price,
            "quantity": holdings,
            "pnl":      pnl,
            "index":    len(prices) - 1,
        })
        holdings = 0.0

    # ── Stats ────────────────────────────────────────────────────────────────
    sell_trades  = [t for t in trades if t["type"] in ("SELL", "CLOSE")]
    total_trades = len(sell_trades)
    wins         = [t for t in sell_trades if t.get("pnl", 0) > 0]
    losses       = [t for t in sell_trades if t.get("pnl", 0) <= 0]
    total_pnl    = sum(t.get("pnl", 0) for t in sell_trades)
    win_rate     = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win      = sum(t["pnl"] for t in wins)    / len(wins)   if wins   else 0
    avg_loss     = sum(t["pnl"] for t in losses)  / len(losses) if losses else 0
    profit_factor= abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else float("inf")
    total_return = (cash - START_CASH) / START_CASH * 100
    buy_hold_return = (prices[-1] - prices[MIN_PRICES]) / prices[MIN_PRICES] * 100

    return {
        "symbol":          symbol,
        "asset_type":      "CRYPTO" if symbol in CRYPTO_ASSETS else "STOCK",
        "data_points":     len(prices),
        "total_trades":    total_trades,
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        round(win_rate, 1),
        "total_pnl":       round(total_pnl, 2),
        "total_return":    round(total_return, 2),
        "buy_hold_return": round(buy_hold_return, 2),
        "vs_buy_hold":     round(total_return - buy_hold_return, 2),
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "profit_factor":   round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "final_cash":      round(cash, 2),
        "start_cash":      START_CASH,
        "trades":          trades[-10:],   # last 10 trades for reference
    }


# ── Report printer ───────────────────────────────────────────────────────────
def print_report(results: list[dict], top_n: int = None):
    results_sorted = sorted(results, key=lambda x: x["total_return"], reverse=True)
    if top_n:
        results_sorted = results_sorted[:top_n]

    print("\n" + "═" * 90)
    print("  STERLIN'S AI TRADING DASHBOARD — BACKTEST RESULTS")
    print("═" * 90)
    print(f"  {'SYMBOL':<8} {'TYPE':<7} {'TRADES':<8} {'WIN%':<7} {'RETURN%':<10} {'B&H%':<10} {'VS B&H':<10} {'PNL':>10}")
    print("─" * 90)

    for r in results_sorted:
        vs   = r["vs_buy_hold"]
        ret  = r["total_return"]
        bh   = r["buy_hold_return"]
        flag = "✅" if vs > 0 else "❌"
        print(f"  {r['symbol']:<8} {r['asset_type']:<7} {r['total_trades']:<8} "
              f"{r['win_rate']:<7} {ret:>+8.1f}%  {bh:>+8.1f}%  {vs:>+8.1f}%  "
              f"${r['total_pnl']:>9,.2f}  {flag}")

    print("─" * 90)

    # Summary
    avg_return   = sum(r["total_return"]   for r in results) / len(results)
    avg_bh       = sum(r["buy_hold_return"] for r in results) / len(results)
    beats_market = sum(1 for r in results if r["vs_buy_hold"] > 0)

    print(f"\n  Assets analyzed:      {len(results)}")
    print(f"  Beats buy & hold:     {beats_market}/{len(results)}")
    print(f"  Avg strategy return:  {avg_return:+.1f}%")
    print(f"  Avg buy & hold:       {avg_bh:+.1f}%")
    print(f"  Avg edge vs market:   {avg_return - avg_bh:+.1f}%")
    print("═" * 90 + "\n")

    # Best and worst
    best  = max(results, key=lambda x: x["total_return"])
    worst = min(results, key=lambda x: x["total_return"])
    print(f"  🏆 Best:  {best['symbol']} — {best['total_return']:+.1f}% return, {best['win_rate']}% win rate")
    print(f"  💀 Worst: {worst['symbol']} — {worst['total_return']:+.1f}% return, {worst['win_rate']}% win rate")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sterlin's Backtester")
    parser.add_argument("--symbol", type=str, help="Backtest a single symbol (e.g. BTC)")
    parser.add_argument("--top",    type=int, help="Show only top N performers")
    parser.add_argument("--export", action="store_true", help="Save results to backtest_results.json")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else ALL_ASSETS
    results = []

    print(f"\n  Running backtest on {len(symbols)} asset(s)...")

    for symbol in symbols:
        prices = load_prices(symbol)
        if len(prices) < MIN_PRICES + 10:
            print(f"  ⚠️  {symbol}: not enough data ({len(prices)} points), skipping.")
            continue
        print(f"  📊 {symbol:<6} — {len(prices):,} prices", end="", flush=True)
        result = backtest_asset(symbol, prices)
        results.append(result)
        print(f"  →  {result['total_return']:+.1f}% return  |  {result['win_rate']}% win rate  |  {result['total_trades']} trades")

    if not results:
        print("  No results. Check your history_*.txt files.")
        return

    print_report(results, top_n=args.top)

    if args.export:
        out_path = os.path.join(PROJECT_DIR, "backtest_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to backtest_results.json\n")


if __name__ == "__main__":
    main()