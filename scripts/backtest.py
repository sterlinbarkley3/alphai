#!/usr/bin/env python3
import os, sys, json, argparse

PROJECT_DIR = "/root/alphai/data"
sys.path.insert(0, PROJECT_DIR)

SHORT_MA   = 5
LONG_MA    = 20
MIN_PRICES = 25
STEP       = 1
START_CASH = 10_000.0

CRYPTO_ASSETS = ["BTC","XRP","SOL","LINK","HBAR","XLM","ADA","DOT","AVAX","MATIC","ATOM"]
STOCK_ASSETS  = ["LMT","ABTC","PFE","ORCL","AAPL","NVDA","MSFT","AMZN","JPM","SPY","QQQ"]
ALL_ASSETS    = CRYPTO_ASSETS + STOCK_ASSETS

def load_prices(symbol):
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

def moving_average(prices, window):
    if len(prices) < window:
        return 0.0
    return sum(prices[-window:]) / window

def get_signal(prices):
    if len(prices) < MIN_PRICES:
        return "HOLD"
    short_ma   = moving_average(prices, SHORT_MA)
    long_ma    = moving_average(prices, LONG_MA)
    momentum   = (prices[-1] - prices[-SHORT_MA]) / prices[-SHORT_MA] * 100 if prices[-SHORT_MA] != 0 else 0
    recent     = prices[-LONG_MA:]
    volatility = (max(recent) - min(recent)) / min(recent) * 100 if min(recent) != 0 else 0
    score = 0
    if short_ma > long_ma:   score += 2
    elif short_ma < long_ma: score -= 2
    if momentum > 2:         score += 2
    elif momentum > 0.5:     score += 1
    elif momentum < -2:      score -= 2
    elif momentum < -0.5:    score -= 1
    if volatility > 10:      score -= 1
    if score >= 4:   return "STRONG BUY"
    elif score >= 2: return "BUY"
    elif score <= -4: return "STRONG SELL"
    elif score <= -2: return "SELL"
    else:            return "HOLD"

def backtest_asset(symbol, prices):
    cash, holdings, trades, last_signal = START_CASH, 0.0, [], "HOLD"
    for i in range(MIN_PRICES, len(prices), STEP):
        window = prices[:i]
        price  = prices[i - 1]
        signal = get_signal(window)
        if signal == last_signal:
            continue
        if signal in ("BUY","STRONG BUY") and cash > 0:
            quantity   = cash / price
            holdings  += quantity
            cash       = 0.0
            trades.append({"type":"BUY","signal":signal,"price":price,"quantity":quantity})
        elif signal in ("SELL","STRONG SELL") and holdings > 0:
            proceeds  = holdings * price
            pnl       = proceeds - (trades[-1]["price"] * holdings) if trades else 0
            cash     += proceeds
            trades.append({"type":"SELL","signal":signal,"price":price,"quantity":holdings,"pnl":pnl})
            holdings  = 0.0
        last_signal = signal
    if holdings > 0:
        proceeds = holdings * prices[-1]
        pnl      = proceeds - (trades[-1]["price"] * holdings) if trades else 0
        cash    += proceeds
        trades.append({"type":"CLOSE","price":prices[-1],"quantity":holdings,"pnl":pnl})
        holdings = 0.0
    sell_trades  = [t for t in trades if t["type"] in ("SELL","CLOSE")]
    total_trades = len(sell_trades)
    wins         = [t for t in sell_trades if t.get("pnl",0) > 0]
    losses       = [t for t in sell_trades if t.get("pnl",0) <= 0]
    total_pnl    = sum(t.get("pnl",0) for t in sell_trades)
    win_rate     = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win      = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss     = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    total_return     = (cash - START_CASH) / START_CASH * 100
    buy_hold_return  = (prices[-1] - prices[MIN_PRICES]) / prices[MIN_PRICES] * 100
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
        "final_cash":      round(cash, 2),
    }

def print_report(results):
    results_sorted = sorted(results, key=lambda x: x["total_return"], reverse=True)
    print("\n" + "="*90)
    print("  STERLIN'S AI TRADING DASHBOARD - BACKTEST RESULTS")
    print("="*90)
    print(f"  {'SYMBOL':<8} {'TYPE':<7} {'TRADES':<8} {'WIN%':<7} {'RETURN%':<10} {'B&H%':<10} {'VS B&H':<10} {'PNL':>10}")
    print("-"*90)
    for r in results_sorted:
        flag = "OK" if r["vs_buy_hold"] > 0 else "--"
        print(f"  {r['symbol']:<8} {r['asset_type']:<7} {r['total_trades']:<8} "
              f"{r['win_rate']:<7} {r['total_return']:>+8.1f}%  {r['buy_hold_return']:>+8.1f}%  "
              f"{r['vs_buy_hold']:>+8.1f}%  ${r['total_pnl']:>9,.2f}  {flag}")
    print("-"*90)
    avg_return = sum(r["total_return"]    for r in results) / len(results)
    avg_bh     = sum(r["buy_hold_return"] for r in results) / len(results)
    beats      = sum(1 for r in results if r["vs_buy_hold"] > 0)
    print(f"\n  Assets analyzed:     {len(results)}")
    print(f"  Beats buy & hold:    {beats}/{len(results)}")
    print(f"  Avg strategy return: {avg_return:+.1f}%")
    print(f"  Avg buy & hold:      {avg_bh:+.1f}%")
    print(f"  Avg edge vs market:  {avg_return - avg_bh:+.1f}%")
    best  = max(results, key=lambda x: x["total_return"])
    worst = min(results, key=lambda x: x["total_return"])
    print(f"\n  Best:  {best['symbol']} - {best['total_return']:+.1f}% return, {best['win_rate']}% win rate")
    print(f"  Worst: {worst['symbol']} - {worst['total_return']:+.1f}% return, {worst['win_rate']}% win rate")
    print("="*90 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    symbols = [args.symbol.upper()] if args.symbol else ALL_ASSETS
    results = []
    print(f"\n  Running backtest on {len(symbols)} asset(s)...")
    for symbol in symbols:
        prices = load_prices(symbol)
        if len(prices) < MIN_PRICES + 10:
            print(f"  SKIP {symbol}: only {len(prices)} data points")
            continue
        print(f"  {symbol:<6} {len(prices):>6} prices", end="", flush=True)
        result = backtest_asset(symbol, prices)
        results.append(result)
        print(f"  -> {result['total_return']:+.1f}% return | {result['win_rate']}% win rate | {result['total_trades']} trades")
    if not results:
        print("  No results.")
        return
    print_report(results)
    if args.export:
        with open("backtest_results.json","w") as f:
            json.dump(results, f, indent=2)
        print("  Saved to backtest_results.json\n")

if __name__ == "__main__":
    main()
