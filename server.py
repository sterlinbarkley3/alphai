from flask import Flask, request, jsonify, send_file, send_from_directory
from dashboard import build_dashboard
import json
import os

app = Flask(__name__)

# Load portfolio functions (assuming portfolio.py exists)
try:
    import sys
    sys.path.insert(0, '/root/alphai')
    from portfolio import buy_asset, sell_asset, get_portfolio_summary
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False
    print("Warning: portfolio.py not found. Trades will be simulated.")

@app.route('/')
def index():
    build_dashboard()
    return send_file('dashboard.html')

@app.route('/trade')
def trade():
    try:
        symbol = request.args.get('symbol')
        price = float(request.args.get('price'))
        action = request.args.get('action')  # BUY or SELL
        mode = request.args.get('mode')      # usd or units
        amount = float(request.args.get('amount'))

        if PORTFOLIO_AVAILABLE:
            # Call your existing portfolio logic
            result = execute_trade(symbol, price, action, mode, amount)
            return jsonify({"success": True, "message": result.get("message", f"{action} {symbol} executed")})
        else:
            # Simulate success for testing
            return jsonify({"success": True, "message": f"SIMULATED: {action} {amount} {mode} of {symbol} at ${price}"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
