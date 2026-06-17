#!/usr/bin/env python3
"""
retrain.py — Auto-retrains models with latest data
Crypto: runs daily at 6am
Stocks: runs Mon/Wed/Fri at 6am

Usage:
    python3 retrain.py --crypto     retrain crypto model only
    python3 retrain.py --stocks     retrain stock model only
    python3 retrain.py --force      retrain both regardless of age
    python3 retrain.py              retrain both if stale
"""

import os, sys, subprocess, json
from datetime import datetime

PROJECT_DIR  = "/root/alphai"
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
TRAINING_DIR = os.path.join(PROJECT_DIR, "training")
SCRIPTS_DIR  = os.path.join(PROJECT_DIR, "scripts")
LOG_PATH     = os.path.join(PROJECT_DIR, "logs", "retrain_log.txt")
PYTHON       = sys.executable

CRYPTO_MAX_AGE = 1   # retrain crypto if older than 1 day
STOCKS_MAX_AGE = 3   # retrain stocks if older than 3 days

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def run(script, label):
    log(f"Starting: {label}")
    result = subprocess.run(
        [PYTHON, script],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        log(f"Done: {label}")
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            if line.strip():
                log(f"  {line}")
    else:
        log(f"ERROR in {label}:")
        log(result.stderr[-500:])
        return False
    return True

def get_model_age(key):
    """Returns age in days for crypto or stocks model"""
    info_path = os.path.join(MODELS_DIR, "model_info_v3.json")
    if not os.path.exists(info_path):
        return 999
    with open(info_path) as f:
        info = json.load(f)
    trained_on_str = info.get(key, {}).get("trained_on")
    if not trained_on_str:
        trained_on_str = info.get("trained_on")
    if not trained_on_str:
        return 999
    trained_on = datetime.fromisoformat(trained_on_str)
    age_hours = (datetime.now() - trained_on).total_seconds() / 3600
    age_days = age_hours / 24
    return round(age_days, 1)

def save_timestamp(key):
    """Update trained_on timestamp for crypto or stocks"""
    info_path = os.path.join(MODELS_DIR, "model_info_v3.json")
    if not os.path.exists(info_path):
        return
    with open(info_path) as f:
        info = json.load(f)
    if key in info:
        info[key]["trained_on"] = datetime.now().isoformat()
    info["trained_on"] = datetime.now().isoformat()
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

def retrain_crypto():
    log("--- CRYPTO RETRAIN ---")
    age = get_model_age("crypto")
    log(f"Crypto model age: {age} day(s)")

    if age < CRYPTO_MAX_AGE and "--force" not in sys.argv:
        log(f"Crypto model is fresh. Skipping.")
        return True

    steps = [
        (os.path.join(SCRIPTS_DIR, "fetch_ohlcv.py"),   "Fetch latest OHLCV data"),
        (os.path.join(SCRIPTS_DIR, "label_data_v4.py"),  "Generate labeled data"),
        (os.path.join(SCRIPTS_DIR, "train_model_v3.py"), "Train crypto model"),
    ]
    for script, label in steps:
        if not run(script, label):
            log(f"Crypto retrain failed at: {label}")
            return False

    save_timestamp("crypto")
    log("Crypto retrain complete.")
    return True

def retrain_stocks():
    log("--- STOCKS RETRAIN ---")
    age = get_model_age("stocks")
    log(f"Stocks model age: {age} day(s)")

    if age < STOCKS_MAX_AGE and "--force" not in sys.argv:
        log(f"Stocks model is fresh. Skipping.")
        return True

    steps = [
        (os.path.join(SCRIPTS_DIR, "fetch_ohlcv.py"),   "Fetch latest OHLCV data"),
        (os.path.join(SCRIPTS_DIR, "label_data_v4.py"),  "Generate labeled data"),
        (os.path.join(SCRIPTS_DIR, "train_model_v3.py"), "Train stocks model"),
    ]
    for script, label in steps:
        if not run(script, label):
            log(f"Stocks retrain failed at: {label}")
            return False

    save_timestamp("stocks")
    log("Stocks retrain complete.")
    return True

def main():
    log("=" * 50)
    log("STERLIN AI — AUTO RETRAIN")
    log("=" * 50)

    args = sys.argv[1:]
    force = "--force" in args

    if "--crypto" in args:
        retrain_crypto()
    elif "--stocks" in args:
        retrain_stocks()
    else:
        # Run both
        retrain_crypto()
        retrain_stocks()

    # Print current model stats
    info_path = os.path.join(MODELS_DIR, "model_info_v3.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        log("=" * 50)
        log(f"Crypto: {info['crypto']['model']} — {info['crypto']['accuracy']}% acc, +{info['crypto']['edge']}% edge")
        log(f"Stocks: {info['stocks']['model']} — {info['stocks']['accuracy']}% acc, +{info['stocks']['edge']}% edge")
        log("=" * 50)

if __name__ == "__main__":
    main()
