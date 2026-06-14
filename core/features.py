#!/usr/bin/env python3
"""
features.py — Advanced feature engineering
Computes RSI, MACD, Bollinger Bands, volume proxies, and multi-timeframe signals
"""

import pandas as pd
import numpy as np

def compute_rsi(prices, period=14):
    deltas = pd.Series(prices).diff()
    gain = deltas.clip(lower=0).rolling(period).mean()
    loss = (-deltas.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).iloc[-1]

def compute_macd(prices, fast=12, slow=26, signal=9):
    s = pd.Series(prices)
    ema_fast = s.ewm(span=fast).mean()
    ema_slow = s.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def compute_bollinger(prices, period=20):
    s = pd.Series(prices[-period:])
    mid = s.mean()
    std = s.std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    current = prices[-1]
    # Position within bands: -1 = at lower, 0 = at mid, 1 = at upper
    if upper == lower:
        return 0.0
    return (current - mid) / (upper - mid + 1e-10)

def compute_features(prices):
    """
    Takes a list of prices, returns a dict of features for ML.
    Requires at least 50 prices for full feature set.
    """
    if len(prices) < 50:
        return None

    p = prices  # shorthand

    # --- Moving averages ---
    ma5   = sum(p[-5:])   / 5
    ma10  = sum(p[-10:])  / 10
    ma20  = sum(p[-20:])  / 20
    ma50  = sum(p[-50:])  / 50

    current = p[-1]

    ma5_cross   = (ma5  - ma20) / ma20  * 100
    ma10_cross  = (ma10 - ma20) / ma20  * 100
    ma20_cross  = (ma20 - ma50) / ma50  * 100
    price_vs_ma20 = (current - ma20) / ma20 * 100
    price_vs_ma50 = (current - ma50) / ma50 * 100

    # --- Momentum across timeframes ---
    mom5  = (p[-1] - p[-6])  / p[-6]  * 100 if p[-6]  != 0 else 0
    mom10 = (p[-1] - p[-11]) / p[-11] * 100 if p[-11] != 0 else 0
    mom20 = (p[-1] - p[-21]) / p[-21] * 100 if p[-21] != 0 else 0

    # --- Volatility across timeframes ---
    def vol(window):
        w = p[-window:]
        mn = min(w)
        if mn == 0: return 0
        return (max(w) - mn) / mn * 100

    vol10 = vol(10)
    vol20 = vol(20)
    vol_ratio = vol10 / vol20 if vol20 != 0 else 1  # short vs long volatility

    # --- RSI ---
    rsi14 = compute_rsi(p, 14)
    rsi = rsi14 if not pd.isna(rsi14) else 50.0

    # --- MACD ---
    macd_line, signal_line, macd_hist = compute_macd(p)
    macd_cross = float(macd_line - signal_line)
    macd_hist  = float(macd_hist)

    # --- Bollinger Band position ---
    bb_position = compute_bollinger(p, 20)

    # --- Rate of change ---
    roc5  = (p[-1] - p[-6])  / p[-6]  * 100 if len(p) > 5  and p[-6]  != 0 else 0
    roc10 = (p[-1] - p[-11]) / p[-11] * 100 if len(p) > 10 and p[-11] != 0 else 0

    # --- Old rule-based score (keep as a feature) ---
    score = 0
    if ma5 > ma20:   score += 2
    elif ma5 < ma20: score -= 2
    if mom5 > 2:     score += 2
    elif mom5 > 0.5: score += 1
    elif mom5 < -2:  score -= 2
    elif mom5 < -0.5: score -= 1
    if vol10 > 10:   score -= 1

    return {
        # MA features
        "ma5_cross":      round(ma5_cross,   4),
        "ma10_cross":     round(ma10_cross,  4),
        "ma20_cross":     round(ma20_cross,  4),
        "price_vs_ma20":  round(price_vs_ma20, 4),
        "price_vs_ma50":  round(price_vs_ma50, 4),

        # Momentum
        "mom5":   round(mom5,  4),
        "mom10":  round(mom10, 4),
        "mom20":  round(mom20, 4),

        # Volatility
        "vol10":     round(vol10,     4),
        "vol20":     round(vol20,     4),
        "vol_ratio": round(vol_ratio, 4),

        # Indicators
        "rsi":         round(float(rsi),        4),
        "macd_cross":  round(macd_cross,        4),
        "macd_hist":   round(macd_hist,         4),
        "bb_position": round(float(bb_position),4),

        # Rate of change
        "roc5":  round(roc5,  4),
        "roc10": round(roc10, 4),

        # Legacy
        "score": score,
    }
