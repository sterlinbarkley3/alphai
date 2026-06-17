from core.market import load_prices
from core.strategies import analyze_asset
from assets import CRYPTO_ASSETS, STOCK_ASSETS
from portfolio import get_portfolio_summary
from datetime import datetime
import json, os, pickle, pandas as pd

PROJECT_DIR = "/root/alphai"
MODEL_PATH  = os.path.join(PROJECT_DIR, "models", "model_crypto.pkl")
FEATURES    = ["ma_cross_pct","momentum","volatility","price_vs_long","score"]

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except:
        return None

def ai_score(model, analysis):
    try:
        short_ma = analysis.get("short_ma", 0)
        long_ma  = analysis.get("long_ma", 0)
        momentum = analysis.get("momentum", 0)
        volatility = analysis.get("volatility", 0)
        price    = analysis.get("price", 0)
        ma_cross = (short_ma - long_ma) / long_ma * 100 if long_ma != 0 else 0
        price_vs_long = (price - long_ma) / long_ma * 100 if long_ma != 0 else 0

        old_score = 0
        if analysis["trend"] == "UP": old_score += 2
        elif analysis["trend"] == "DOWN": old_score -= 2
        if momentum > 2: old_score += 2
        elif momentum > 0.5: old_score += 1
        elif momentum < -2: old_score -= 2
        elif momentum < -0.5: old_score -= 1
        if volatility > 10: old_score -= 1

        X = pd.DataFrame([{
            "ma_cross_pct":  ma_cross,
            "momentum":      momentum,
            "volatility":    volatility,
            "price_vs_long": price_vs_long,
            "score":         old_score,
        }])
        prob = model.predict_proba(X)[0][1]

        # Map to signal + numeric score for sorting
        if old_score >= 2 and prob >= 0.60:   return "STRONG BUY",  80, prob
        elif old_score >= 2 and prob >= 0.52: return "BUY",         60, prob
        elif old_score <= -2 and prob >= 0.60: return "STRONG SELL",-80, prob
        elif old_score <= -2 and prob >= 0.52: return "SELL",       -60, prob
        else:                                  return "HOLD",          0, prob
    except:
        return "HOLD", 0, 0.5

def old_score(analysis):
    score = 0
    if analysis["trend"] == "UP": score += 30
    elif analysis["trend"] == "DOWN": score -= 30
    if analysis["confidence"] >= 70: score += 30
    elif analysis["confidence"] >= 50: score += 15
    elif analysis["confidence"] >= 30: score += 5
    else: score -= 10
    if analysis["momentum"] > 0: score += 20
    else: score -= 20
    if analysis["risk"] == "HIGH": score -= 40
    elif analysis["risk"] == "MEDIUM": score -= 10
    else: score += 10
    if analysis["week_prediction"] > 2: score += 10
    elif analysis["week_prediction"] < -2: score -= 10
    if score >= 60: decision = "STRONG BUY"
    elif score >= 30: decision = "BUY"
    elif score <= -60: decision = "STRONG SELL"
    elif score <= -30: decision = "SELL"
    else: decision = "HOLD"
    return decision, score

def build_dashboard():
    model = load_model()
    using_ai = model is not None
    print(f"  {'AI model loaded' if using_ai else 'No model found, using rules-based scoring'}")

    all_assets = (
        [("BTC", "crypto")] +
        [(s, "crypto") for s in CRYPTO_ASSETS] +
        [(s, "stock") for s in STOCK_ASSETS]
    )
    seen = set()
    unique_assets = []
    for item in all_assets:
        if item[0] not in seen:
            seen.add(item[0])
            unique_assets.append(item)
    all_assets = unique_assets

    asset_data = []
    current_prices = {}

    for symbol, asset_type in all_assets:
        analysis = analyze_asset(symbol)
        prices = load_prices(symbol)
        if analysis["price"] > 0:
            current_prices[symbol] = analysis["price"]

        if using_ai and analysis["trend"] != "UNKNOWN":
            decision, score, confidence = ai_score(model, analysis)
        else:
            decision, score = old_score(analysis)
            confidence = 0.5

        chart_prices = prices[-60:] if len(prices) >= 60 else prices
        daily_change_pct = 0
        if len(chart_prices) >= 2 and chart_prices[-2] != 0:
            daily_change_pct = round((chart_prices[-1] - chart_prices[-2]) / chart_prices[-2] * 100, 2)

        asset_data.append({
            "symbol": symbol, "type": asset_type,
            "price": analysis["price"], "trend": analysis["trend"],
            "confidence": analysis["confidence"], "risk": analysis["risk"],
            "momentum": analysis["momentum"], "volatility": analysis["volatility"],
            "week": analysis["week_prediction"], "month": analysis["month_prediction"],
            "year": analysis["year_prediction"], "reason": analysis["reason"],
            "decision": decision, "score": score,
            "ai_confidence": round(confidence * 100, 1),
            "chart": chart_prices, "daily_change_pct": daily_change_pct,
            "using_ai": using_ai,
        })

    asset_data.sort(key=lambda x: x["score"], reverse=True)
    portfolio = get_portfolio_summary(current_prices)

    strong_buys = [a for a in asset_data if a["decision"] == "STRONG BUY"]
    buys        = [a for a in asset_data if a["decision"] == "BUY"]
    holds       = [a for a in asset_data if a["decision"] == "HOLD"]
    sells       = [a for a in asset_data if a["decision"] == "SELL"]
    strong_sells= [a for a in asset_data if a["decision"] == "STRONG SELL"]
    bullish = len(strong_buys) + len(buys)
    bearish = len(sells) + len(strong_sells)
    neutral = len(holds)
    total   = len(asset_data)

    if bullish > bearish: overall_sentiment, sentiment_color = "BULLISH", "#3fb950"
    elif bearish > bullish: overall_sentiment, sentiment_color = "BEARISH", "#f85149"
    else: overall_sentiment, sentiment_color = "NEUTRAL", "#f9a825"

    top_picks = [a for a in asset_data if "BUY" in a["decision"]][:3] or asset_data[:3]

    def fmt_price(price):
        if price == 0: return "N/A"
        elif price < 1: return f"${price:,.5f}"
        elif price < 100: return f"${price:,.4f}"
        else: return f"${price:,.2f}"

    def make_row(a):
        price_str = fmt_price(a["price"])
        chg = a["daily_change_pct"]
        chg_color = "#3fb950" if chg >= 0 else "#f85149"
        chg_str = f"{'▲' if chg >= 0 else '▼'}{abs(chg):.2f}%"
        dec = a["decision"]
        dec_color = "#3fb950" if "BUY" in dec else "#f85149" if "SELL" in dec else "#f9a825"
        dec_bg = "#3fb95022" if "BUY" in dec else "#f8514922" if "SELL" in dec else "#f9a82522"
        chart_data = json.dumps(a["chart"])
        pj = json.dumps(a["price"])
        sj = json.dumps(a["symbol"])
        ai_badge = f'<span style="font-size:8px;color:#58a6ff;background:#58a6ff18;border:1px solid #58a6ff33;padding:1px 5px;border-radius:3px;margin-left:4px;">AI {a["ai_confidence"]}%</span>' if a["using_ai"] else ""
        def mj(v): return json.dumps(str(v))
        modal_args = f"'{a['symbol']}',{json.dumps(price_str)},{mj(a['trend'])},{mj(a['confidence'])},{mj(a['risk'])},{mj(a['momentum'])},{mj(a['volatility'])},{mj(a['week'])},{mj(a['month'])},{mj(a['year'])},{json.dumps(a['reason'])},{json.dumps(dec)},{mj(a['score'])},{pj}"
        return f"""
        <div class="asset-row" data-decision="{dec}" data-type="{a['type']}" data-symbol="{a['symbol']}">
            <div class="row-left" onclick="openModal({modal_args})">
                <div class="row-symbol-wrap">
                    <span class="row-symbol">{a['symbol']}</span>
                    <span class="row-type type-{a['type']}">{a['type'].upper()}</span>
                    {ai_badge}
                </div>
                <span class="row-reason">{a['reason']}</span>
            </div>
            <div class="row-chart-wrap" onclick="openModal({modal_args})">
                <canvas class="sparkline" id="spark_{a['symbol']}" data-prices="{chart_data.replace(chr(34),'&quot;')}" data-up="{'true' if chg >= 0 else 'false'}"></canvas>
            </div>
            <div class="row-right">
                <div onclick="openModal({modal_args})" style="cursor:pointer;display:flex;flex-direction:column;align-items:flex-end;gap:3px;">
                    <span class="row-price">{price_str}</span>
                    <span class="row-change" style="color:{chg_color}">{chg_str}</span>
                    <span class="row-signal" style="color:{dec_color};background:{dec_bg};border:1px solid {dec_color}">{dec}</span>
                </div>
                <div class="trade-btns">
                    <button class="trade-btn buy-btn" onclick="event.stopPropagation();openTradeModal('{a['symbol']}',{a['price']},'BUY')">BUY</button>
                    <button class="trade-btn sell-btn" onclick="event.stopPropagation();openTradeModal('{a['symbol']}',{a['price']},'SELL')">SELL</button>
                </div>
            </div>
        </div>"""

    rows_html = "".join(make_row(a) for a in asset_data)

    top_picks_html = ""
    for i, a in enumerate(top_picks):
        ai_conf = f" · AI {a['ai_confidence']}%" if a["using_ai"] else ""
        top_picks_html += f"""
        <div class="pick-card">
            <div class="pick-rank">#{i+1}</div>
            <div class="pick-info">
                <span class="pick-symbol">{a['symbol']}</span>
                <span class="pick-price">{fmt_price(a['price'])}</span>
            </div>
            <div class="pick-score" style="color:#3fb950">{a['decision']}{ai_conf}</div>
        </div>"""

    positions_html = ""
    if portfolio["positions"]:
        for pos in portfolio["positions"]:
            gl_color = "#3fb950" if pos["gain_loss"] >= 0 else "#f85149"
            gl_sign = "+" if pos["gain_loss"] >= 0 else ""
            positions_html += f"""
            <div class="pos-row">
                <div class="pos-left">
                    <span class="pos-symbol">{pos['symbol']}</span>
                    <span class="pos-qty">{pos['qty']} @ ${pos['avg_price']:,.4f}</span>
                </div>
                <div class="pos-right">
                    <span class="pos-value">${pos['current_value']:,.2f}</span>
                    <span class="pos-gl" style="color:{gl_color}">{gl_sign}${pos['gain_loss']:,.2f} ({gl_sign}{pos['gain_loss_pct']:.2f}%)</span>
                </div>
            </div>"""
    else:
        positions_html = '<div style="color:#484f58;font-size:12px;padding:16px 0;text-align:center;">No positions yet. Start trading!</div>'

    history_html = ""
    for h in reversed(portfolio["history"]):
        color = "#3fb950" if h["action"] == "BUY" else "#f85149"
        history_html += f"""
        <div class="hist-row">
            <span style="color:{color};font-weight:700;width:40px">{h['action']}</span>
            <span style="color:#fff;width:50px">{h['symbol']}</span>
            <span style="color:#8b949e;flex:1">{h['qty']} @ ${h['price']:,.4f}</span>
            <span style="color:#fff">${h['total']:,.2f}</span>
        </div>"""
    if not history_html:
        history_html = '<div style="color:#484f58;font-size:12px;padding:8px 0;text-align:center;">No trades yet.</div>'

    total_gl_color = "#3fb950" if portfolio["total_gain_loss"] >= 0 else "#f85149"
    total_gl_sign = "+" if portfolio["total_gain_loss"] >= 0 else ""
    ai_status = "🤖 AI-POWERED" if using_ai else "📊 RULES-BASED"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sterlin's Trading Dashboard</title>
<style>
:root{{--bg:#0d1117;--surface:#161b22;--surface2:#1c2333;--surface3:#21262d;--border:#21262d;--border2:#30363d;--text:#e6edf3;--text-muted:#8b949e;--text-dim:#484f58;--green:#3fb950;--green-dim:#3fb95018;--red:#f85149;--red-dim:#f8514918;--yellow:#f9a825;--yellow-dim:#f9a82518;--blue:#58a6ff;--blue-dim:#58a6ff18;--font:'SF Mono','Fira Code','Cascadia Code',monospace;}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;}}
.nav{{background:var(--surface);border-bottom:1px solid var(--border);padding:0 28px;height:58px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}}
.nav-brand{{display:flex;flex-direction:column;}}
.nav-title{{font-size:15px;font-weight:700;color:#fff;letter-spacing:1px;}}
.nav-sub{{font-size:10px;color:var(--text-dim);letter-spacing:2px;margin-top:1px;}}
.nav-right{{display:flex;align-items:center;gap:12px;}}
.live-pill{{display:flex;align-items:center;gap:6px;background:var(--surface2);border:1px solid var(--border2);padding:5px 12px;border-radius:20px;font-size:10px;color:var(--text-muted);letter-spacing:1px;}}
.live-dot{{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}
.nav-time{{font-size:11px;color:var(--text-dim);}}
.wallet-btn{{background:var(--blue-dim);border:1px solid var(--blue);color:var(--blue);padding:6px 16px;border-radius:6px;cursor:pointer;font-family:var(--font);font-size:11px;font-weight:700;letter-spacing:1px;transition:all 0.15s;}}
.wallet-btn:hover{{background:var(--blue);color:#0d1117;}}
.main{{max-width:1200px;margin:0 auto;padding:24px 20px;}}
.top-section{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}}
.sentiment-card,.picks-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;}}
.sentiment-title,.picks-title{{font-size:10px;color:var(--text-dim);letter-spacing:2px;margin-bottom:16px;}}
.sentiment-overall{{font-size:28px;font-weight:700;margin-bottom:16px;}}
.sentiment-bars{{display:flex;flex-direction:column;gap:8px;}}
.sent-row{{display:flex;align-items:center;gap:10px;font-size:11px;}}
.sent-label{{color:var(--text-muted);width:70px;}}
.sent-bar-bg{{flex:1;height:4px;background:var(--border2);border-radius:2px;overflow:hidden;}}
.sent-bar-fill{{height:100%;border-radius:2px;}}
.sent-num{{color:var(--text-muted);width:20px;text-align:right;}}
.summary-pills{{display:flex;gap:8px;margin-top:16px;flex-wrap:wrap;}}
.summary-pill{{padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:1px;}}
.pick-card{{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);}}
.pick-card:last-child{{border-bottom:none;}}
.pick-rank{{font-size:18px;font-weight:700;color:var(--text-dim);width:28px;}}
.pick-info{{flex:1;display:flex;flex-direction:column;gap:2px;}}
.pick-symbol{{font-size:15px;font-weight:700;color:#fff;}}
.pick-price{{font-size:12px;color:var(--text-muted);}}
.pick-score{{font-size:12px;font-weight:700;}}
.controls{{display:flex;align-items:center;gap:8px;margin-bottom:16px;flex-wrap:wrap;}}
.filter-label{{font-size:10px;color:var(--text-dim);letter-spacing:2px;margin-right:4px;}}
.filter-btn{{background:var(--surface);border:1px solid var(--border2);color:var(--text-muted);padding:5px 14px;border-radius:6px;cursor:pointer;font-family:var(--font);font-size:10px;letter-spacing:1px;transition:all 0.15s;}}
.filter-btn:hover{{border-color:var(--blue);color:var(--text);}}
.filter-btn.active{{background:var(--blue);border-color:var(--blue);color:#0d1117;font-weight:700;}}
.search-input{{margin-left:auto;background:var(--surface);border:1px solid var(--border2);color:var(--text);padding:5px 14px;border-radius:6px;font-family:var(--font);font-size:11px;outline:none;width:180px;transition:border 0.15s;}}
.search-input:focus{{border-color:var(--blue);}}
.search-input::placeholder{{color:var(--text-dim);}}
.watchlist{{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
.watchlist-header{{display:flex;justify-content:space-between;padding:12px 20px;border-bottom:1px solid var(--border);font-size:10px;color:var(--text-dim);letter-spacing:2px;}}
.asset-row{{display:flex;align-items:center;padding:12px 20px;border-bottom:1px solid var(--border);gap:12px;transition:background 0.1s;}}
.asset-row:last-child{{border-bottom:none;}}
.asset-row:hover{{background:var(--surface2);}}
.row-left{{flex:1;display:flex;flex-direction:column;gap:3px;min-width:100px;cursor:pointer;}}
.row-symbol-wrap{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}}
.row-symbol{{font-size:15px;font-weight:700;color:#fff;}}
.row-type{{font-size:8px;padding:2px 6px;border-radius:3px;letter-spacing:1px;}}
.type-crypto{{background:var(--blue-dim);color:var(--blue);border:1px solid #58a6ff33;}}
.type-stock{{background:#7c3aed18;color:#a78bfa;border:1px solid #7c3aed33;}}
.row-reason{{font-size:10px;color:var(--text-dim);}}
.row-chart-wrap{{width:90px;height:36px;cursor:pointer;}}
.sparkline{{width:90px!important;height:36px!important;display:block;}}
.row-right{{display:flex;align-items:center;gap:12px;}}
.row-price{{font-size:14px;font-weight:700;color:#fff;}}
.row-change{{font-size:11px;font-weight:600;}}
.row-signal{{font-size:9px;font-weight:700;padding:3px 8px;border-radius:4px;letter-spacing:1px;white-space:nowrap;}}
.trade-btns{{display:flex;gap:6px;}}
.trade-btn{{padding:5px 12px;border-radius:5px;font-family:var(--font);font-size:10px;font-weight:700;letter-spacing:1px;cursor:pointer;border:1px solid;transition:all 0.15s;}}
.buy-btn{{background:var(--green-dim);color:var(--green);border-color:var(--green);}}
.buy-btn:hover{{background:var(--green);color:#0d1117;}}
.sell-btn{{background:var(--red-dim);color:var(--red);border-color:var(--red);}}
.sell-btn:hover{{background:var(--red);color:#fff;}}
.modal-overlay{{display:none;position:fixed;inset:0;background:#00000099;z-index:200;align-items:center;justify-content:center;padding:20px;}}
.modal-overlay.open{{display:flex;}}
.modal{{background:var(--surface);border:1px solid var(--border2);border-radius:16px;padding:28px;width:100%;max-width:440px;position:relative;max-height:90vh;overflow-y:auto;}}
.modal-close{{position:absolute;top:16px;right:16px;background:var(--surface2);border:1px solid var(--border2);color:var(--text-muted);width:28px;height:28px;border-radius:6px;cursor:pointer;font-family:var(--font);font-size:14px;display:flex;align-items:center;justify-content:center;}}
.modal-close:hover{{color:#fff;}}
.modal-symbol{{font-size:26px;font-weight:700;color:#fff;}}
.modal-price{{font-size:20px;color:var(--blue);margin-top:4px;}}
.modal-signal{{display:inline-block;margin-top:8px;font-size:11px;font-weight:700;padding:4px 12px;border-radius:5px;letter-spacing:1px;}}
.modal-row{{display:flex;justify-content:space-between;padding:11px 0;border-bottom:1px solid var(--border);font-size:12px;}}
.modal-row:last-child{{border-bottom:none;}}
.modal-label{{color:var(--text-muted);}}
.modal-value{{color:#fff;font-weight:600;}}
.trade-modal-title{{font-size:18px;font-weight:700;color:#fff;margin-bottom:4px;}}
.trade-modal-price{{font-size:13px;color:var(--text-muted);margin-bottom:20px;}}
.toggle-wrap{{display:flex;background:var(--surface2);border:1px solid var(--border2);border-radius:8px;padding:3px;margin-bottom:16px;}}
.toggle-opt{{flex:1;padding:7px;text-align:center;border-radius:6px;font-family:var(--font);font-size:11px;letter-spacing:1px;cursor:pointer;color:var(--text-muted);border:none;background:none;transition:all 0.15s;}}
.toggle-opt.active{{background:var(--surface3);color:#fff;font-weight:700;}}
.trade-input-wrap{{margin-bottom:16px;}}
.trade-input-label{{font-size:10px;color:var(--text-dim);letter-spacing:1px;margin-bottom:6px;display:block;}}
.trade-input{{width:100%;background:var(--surface2);border:1px solid var(--border2);color:#fff;padding:10px 14px;border-radius:8px;font-family:var(--font);font-size:14px;outline:none;transition:border 0.15s;}}
.trade-input:focus{{border-color:var(--blue);}}
.trade-preview{{background:var(--surface2);border:1px solid var(--border2);border-radius:8px;padding:12px 14px;margin-bottom:16px;font-size:12px;color:var(--text-muted);min-height:40px;}}
.trade-submit{{width:100%;padding:12px;border-radius:8px;font-family:var(--font);font-size:13px;font-weight:700;letter-spacing:2px;cursor:pointer;border:none;transition:all 0.15s;}}
.trade-submit-buy{{background:var(--green);color:#0d1117;}}
.trade-submit-buy:hover{{background:#2ea043;}}
.trade-submit-sell{{background:var(--red);color:#fff;}}
.trade-submit-sell:hover{{background:#d93025;}}
.trade-result{{margin-top:12px;padding:10px;border-radius:6px;font-size:12px;text-align:center;display:none;}}
.wallet-modal{{max-width:560px;}}
.wallet-total{{font-size:32px;font-weight:700;color:#fff;margin-bottom:4px;}}
.wallet-sub{{font-size:13px;color:var(--text-muted);margin-bottom:20px;}}
.wallet-stats{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:20px;}}
.wallet-stat{{background:var(--surface2);border-radius:8px;padding:10px 12px;}}
.wallet-stat-label{{font-size:9px;color:var(--text-dim);letter-spacing:1px;margin-bottom:4px;}}
.wallet-stat-value{{font-size:13px;font-weight:700;}}
.section-title{{font-size:10px;color:var(--text-dim);letter-spacing:2px;margin-bottom:10px;margin-top:16px;}}
.pos-row{{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--border);}}
.pos-row:last-child{{border-bottom:none;}}
.pos-left{{display:flex;flex-direction:column;gap:2px;}}
.pos-symbol{{font-size:14px;font-weight:700;color:#fff;}}
.pos-qty{{font-size:10px;color:var(--text-dim);}}
.pos-right{{display:flex;flex-direction:column;align-items:flex-end;gap:2px;}}
.pos-value{{font-size:13px;font-weight:700;color:#fff;}}
.pos-gl{{font-size:11px;}}
.hist-row{{display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--border);font-size:11px;}}
.hist-row:last-child{{border-bottom:none;}}
@media(max-width:700px){{.top-section{{grid-template-columns:1fr;}}.nav-time{{display:none;}}.row-chart-wrap{{display:none;}}.trade-btns{{flex-direction:column;gap:4px;}}}}
</style>
</head>
<body>
<nav class="nav">
    <div class="nav-brand">
        <span class="nav-title">Sterlin's Trading Dashboard</span>
        <span class="nav-sub">{ai_status} · LIVE MARKET INTELLIGENCE</span>
    </div>
    <div class="nav-right">
        <span class="nav-time">{datetime.now().strftime("%b %d, %Y  %H:%M:%S")}</span>
        <div class="live-pill"><div class="live-dot"></div>LIVE</div>
        <button class="wallet-btn" onclick="document.getElementById('wallet-modal').classList.add('open')">💼 WALLET</button>
    </div>
</nav>
<div class="main">
    <div class="top-section">
        <div class="sentiment-card">
            <div class="sentiment-title">MARKET SENTIMENT</div>
            <div class="sentiment-overall" style="color:{sentiment_color}">{overall_sentiment}</div>
            <div class="sentiment-bars">
                <div class="sent-row"><span class="sent-label">BULLISH</span><div class="sent-bar-bg"><div class="sent-bar-fill" style="width:{int(bullish/total*100) if total else 0}%;background:#3fb950"></div></div><span class="sent-num" style="color:#3fb950">{bullish}</span></div>
                <div class="sent-row"><span class="sent-label">BEARISH</span><div class="sent-bar-bg"><div class="sent-bar-fill" style="width:{int(bearish/total*100) if total else 0}%;background:#f85149"></div></div><span class="sent-num" style="color:#f85149">{bearish}</span></div>
                <div class="sent-row"><span class="sent-label">NEUTRAL</span><div class="sent-bar-bg"><div class="sent-bar-fill" style="width:{int(neutral/total*100) if total else 0}%;background:#f9a825"></div></div><span class="sent-num" style="color:#f9a825">{neutral}</span></div>
            </div>
            <div class="summary-pills">
                <span class="summary-pill" style="background:#3fb95022;color:#3fb950;border:1px solid #3fb95044">{len(strong_buys)} STRONG BUY</span>
                <span class="summary-pill" style="background:#3fb95022;color:#3fb950;border:1px solid #3fb95044">{len(buys)} BUY</span>
                <span class="summary-pill" style="background:#f9a82522;color:#f9a825;border:1px solid #f9a82544">{neutral} HOLD</span>
                <span class="summary-pill" style="background:#f8514922;color:#f85149;border:1px solid #f8514944">{len(sells)+len(strong_sells)} SELL</span>
            </div>
        </div>
        <div class="picks-card">
            <div class="picks-title">TODAY'S TOP OPPORTUNITIES</div>
            {top_picks_html}
        </div>
    </div>
    <div class="controls">
        <span class="filter-label">FILTER</span>
        <button class="filter-btn active" onclick="filterCards('all',this)">ALL</button>
        <button class="filter-btn" onclick="filterCards('BUY',this)">BUY</button>
        <button class="filter-btn" onclick="filterCards('SELL',this)">SELL</button>
        <button class="filter-btn" onclick="filterCards('HOLD',this)">HOLD</button>
        <button class="filter-btn" onclick="filterCards('crypto',this)">CRYPTO</button>
        <button class="filter-btn" onclick="filterCards('stock',this)">STOCKS</button>
        <input class="search-input" type="text" placeholder="Search symbol..." oninput="searchCards(this.value)">
    </div>
    <div class="watchlist">
        <div class="watchlist-header"><span>ASSET</span><span>CHART</span><span>PRICE / ACTION</span></div>
        <div id="rows">{rows_html}</div>
    </div>
</div>

<!-- DETAIL MODAL -->
<div class="modal-overlay" id="detail-modal" onclick="if(event.target===this)this.classList.remove('open')">
    <div class="modal">
        <button class="modal-close" onclick="document.getElementById('detail-modal').classList.remove('open')">✕</button>
        <div style="margin-bottom:20px;">
            <div class="modal-symbol" id="m-symbol"></div>
            <div class="modal-price" id="m-price"></div>
            <div id="m-signal"></div>
        </div>
        <div id="m-rows"></div>
    </div>
</div>

<!-- TRADE MODAL -->
<div class="modal-overlay" id="trade-modal" onclick="if(event.target===this)this.classList.remove('open')">
    <div class="modal">
        <button class="modal-close" onclick="document.getElementById('trade-modal').classList.remove('open')">✕</button>
        <div class="trade-modal-title" id="t-title"></div>
        <div class="trade-modal-price" id="t-price-label"></div>
        <div class="toggle-wrap">
            <button class="toggle-opt active" id="toggle-usd" onclick="setMode('usd')">$ DOLLARS</button>
            <button class="toggle-opt" id="toggle-units" onclick="setMode('units')">SHARES / COINS</button>
        </div>
        <div class="trade-input-wrap">
            <label class="trade-input-label" id="t-input-label">AMOUNT (USD)</label>
            <input class="trade-input" type="number" id="t-amount" placeholder="0.00" oninput="updatePreview()">
        </div>
        <div class="trade-preview" id="t-preview">Enter an amount to see preview</div>
        <button class="trade-submit" id="t-submit" onclick="submitTrade()"></button>
        <div class="trade-result" id="t-result"></div>
    </div>
</div>

<!-- WALLET MODAL -->
<div class="modal-overlay" id="wallet-modal" onclick="if(event.target===this)this.classList.remove('open')">
    <div class="modal wallet-modal">
        <button class="modal-close" onclick="document.getElementById('wallet-modal').classList.remove('open')">✕</button>
        <div class="wallet-total">${portfolio['total_portfolio']:,.2f}</div>
        <div class="wallet-sub">Total Portfolio Value</div>
        <div class="wallet-stats">
            <div class="wallet-stat"><div class="wallet-stat-label">CASH</div><div class="wallet-stat-value" style="color:#58a6ff">${portfolio['cash']:,.2f}</div></div>
            <div class="wallet-stat"><div class="wallet-stat-label">INVESTED</div><div class="wallet-stat-value" style="color:#f9a825">${portfolio['total_current']:,.2f}</div></div>
            <div class="wallet-stat"><div class="wallet-stat-label">GAIN / LOSS</div><div class="wallet-stat-value" style="color:{total_gl_color}">{total_gl_sign}${portfolio['total_gain_loss']:,.2f}</div></div>
        </div>
        <div class="section-title">POSITIONS</div>
        <div>{positions_html}</div>
        <div class="section-title">RECENT TRADES</div>
        <div>{history_html}</div>
    </div>
</div>

<script>
var tradeSymbol='',tradePrice=0,tradeAction='',tradeMode='usd';
function filterCards(filter,btn){{
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    if(btn)btn.classList.add('active');
    document.querySelectorAll('.asset-row').forEach(row=>{{
        if(filter==='all')row.style.display='';
        else if(filter==='crypto'||filter==='stock')
            row.style.display=row.dataset.type===filter?'':'none';
        else row.style.display=row.dataset.decision.includes(filter)?'':'none';
    }});
}}
function searchCards(val){{
    val=val.toUpperCase();
    document.querySelectorAll('.asset-row').forEach(row=>{{
        row.style.display=row.dataset.symbol.includes(val)?'':'none';
    }});
}}
function openModal(symbol,price,trend,confidence,risk,momentum,volatility,week,month,year,reason,decision,score,rawPrice){{
    document.getElementById('m-symbol').textContent=symbol;
    document.getElementById('m-price').textContent=price;
    var dc=decision.includes('BUY')?'#3fb950':decision.includes('SELL')?'#f85149':'#f9a825';
    var db=decision.includes('BUY')?'#3fb95022':decision.includes('SELL')?'#f8514922':'#f9a82522';
    document.getElementById('m-signal').innerHTML='<span class="modal-signal" style="color:'+dc+';background:'+db+';border:1px solid '+dc+'">'+decision+'</span><span style="font-size:11px;color:#8b949e;margin-left:10px;">SCORE '+score+'</span>';
    var rows=[['TREND',trend],['SIGNAL STRENGTH',confidence+'%'],['RISK',risk],['MOMENTUM',momentum],['VOLATILITY',volatility],['1 WEEK',week+'%'],['1 MONTH',month+'%'],['1 YEAR',year+'%'],['REASON',reason]];
    document.getElementById('m-rows').innerHTML=rows.map(r=>'<div class="modal-row"><span class="modal-label">'+r[0]+'</span><span class="modal-value">'+r[1]+'</span></div>').join('');
    document.getElementById('detail-modal').classList.add('open');
}}
function openTradeModal(symbol,price,action){{
    tradeSymbol=symbol;tradePrice=price;tradeAction=action;tradeMode='usd';
    document.getElementById('t-title').textContent=action+' '+symbol;
    document.getElementById('t-price-label').textContent='Current Price: $'+price.toLocaleString();
    document.getElementById('t-submit').textContent=action+' '+symbol;
    document.getElementById('t-submit').className='trade-submit trade-submit-'+action.toLowerCase();
    document.getElementById('t-amount').value='';
    document.getElementById('t-preview').textContent='Enter an amount to see preview';
    document.getElementById('t-result').style.display='none';
    document.getElementById('toggle-usd').classList.add('active');
    document.getElementById('toggle-units').classList.remove('active');
    document.getElementById('t-input-label').textContent='AMOUNT (USD)';
    document.getElementById('t-amount').placeholder='0.00';
    document.getElementById('trade-modal').classList.add('open');
}}
function setMode(mode){{
    tradeMode=mode;
    document.getElementById('toggle-usd').classList.toggle('active',mode==='usd');
    document.getElementById('toggle-units').classList.toggle('active',mode==='units');
    document.getElementById('t-input-label').textContent=mode==='usd'?'AMOUNT (USD)':'SHARES / COINS';
    document.getElementById('t-amount').placeholder=mode==='usd'?'0.00':'0.000000';
    document.getElementById('t-amount').value='';
    document.getElementById('t-preview').textContent='Enter an amount to see preview';
}}
function updatePreview(){{
    var val=parseFloat(document.getElementById('t-amount').value);
    if(!val||val<=0){{document.getElementById('t-preview').textContent='Enter an amount to see preview';return;}}
    var usd=tradeMode==='usd'?val:val*tradePrice;
    var units=tradeMode==='usd'?val/tradePrice:val;
    document.getElementById('t-preview').innerHTML='<span style="color:#e6edf3">'+tradeAction+' <strong>'+units.toFixed(6)+' '+tradeSymbol+'</strong> for <strong>$'+usd.toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}})+'</strong></span>';
}}
function submitTrade(){{
    var val=parseFloat(document.getElementById('t-amount').value);
    if(!val||val<=0){{showResult('Please enter a valid amount.',false);return;}}
    var params='symbol='+encodeURIComponent(tradeSymbol)+'&price='+tradePrice+'&action='+tradeAction+'&mode='+tradeMode+'&amount='+val;
    fetch('http://localhost:8080/trade?'+params).then(r=>r.json()).then(data=>{{
        showResult(data.message,data.success);
        if(data.success)setTimeout(()=>location.reload(),1500);
    }}).catch(()=>showResult('Server not running. Start server.py first.',false));
}}
function showResult(msg,success){{
    var el=document.getElementById('t-result');
    el.textContent=msg;el.style.display='block';
    el.style.background=success?'#3fb95022':'#f8514922';
    el.style.color=success?'#3fb950':'#f85149';
    el.style.border='1px solid '+(success?'#3fb950':'#f85149');
}}
document.querySelectorAll('.sparkline').forEach(canvas=>{{
    var prices=JSON.parse(canvas.dataset.prices.replace(/&quot;/g,'"'));
    if(!prices||prices.length<2)return;
    var isUp=canvas.dataset.up==='true';
    var color=isUp?'#3fb950':'#f85149';
    var colorDim=isUp?'#3fb95025':'#f8514925';
    canvas.width=90;canvas.height=36;
    var ctx=canvas.getContext('2d');
    var min=Math.min(...prices),max=Math.max(...prices);
    var range=max-min||1;
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=1.5;
    prices.forEach((p,i)=>{{
        var x=(i/(prices.length-1))*90;
        var y=36-((p-min)/range)*(36-6)-3;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }});
    ctx.stroke();
    ctx.lineTo(90,36);ctx.lineTo(0,36);ctx.closePath();
    ctx.fillStyle=colorDim;ctx.fill();
}});
</script>
</body>
</html>"""

    with open("/root/alphai/dashboard.html", "w") as f:
        f.write(html)
    print("Dashboard built!")

if __name__ == "__main__":
    build_dashboard()
