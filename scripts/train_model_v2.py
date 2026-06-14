#!/usr/bin/env python3
import os, json, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


PROJECT_DIR = "/Users/mythreeboyz/pythonuh/ai trader"

FEATURES = [
    "ma5_cross","ma10_cross","ma20_cross","price_vs_ma20","price_vs_ma50",
    "mom5","mom10","mom20",
    "vol10","vol20","vol_ratio",
    "rsi","macd_cross","macd_hist","bb_position",
    "roc5","roc10",
    "atr_pct","obv_pct","vol_trend","hl_position",
    "score",
]

def train_and_evaluate(name, X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=15,
            random_state=42, n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
        ),
    }

    print(f"\n  Training {name} models...")
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}\n")

    results = {}
    for model_name, model in models.items():
        print(f"  [{model_name}]", end="", flush=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        cv    = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        print(f" accuracy={acc*100:.1f}%  cv={cv.mean()*100:.1f}% +/-{cv.std()*100:.1f}%")
        results[model_name] = {"model": model, "accuracy": acc, "cv_mean": cv.mean(), "cv_std": cv.std()}

    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best      = results[best_name]
    model     = best["model"]

    print(f"\n  Winner: {best_name} ({best['accuracy']*100:.1f}% accuracy, {best['cv_mean']*100:.1f}% cv)")

    if hasattr(model, "feature_importances_"):
        print(f"\n  Top features for {name}:")
        importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
        for feat, imp in importances[:8]:
            bar = "=" * int(imp * 50)
            print(f"    {feat:<20} {bar} {imp*100:.1f}%")

    preds    = model.predict(X_test)
    baseline = y_test.mean() * 100
    print(f"\n  Baseline: {baseline:.1f}%  Model: {best['accuracy']*100:.1f}%  Edge: +{best['accuracy']*100 - baseline:.1f}%")
    print(classification_report(y_test, preds, target_names=['Wrong','Correct']))

    return model, best_name, best["accuracy"], best["cv_mean"], baseline

def main():
    print("\n" + "="*60)
    print("  STERLIN AI - MODEL TRAINING v2")
    print("="*60)

    print("\n  Loading crypto training data...")
    df_crypto = pd.read_csv(os.path.join(PROJECT_DIR, "training_crypto_v3.csv"))
    print(f"  {len(df_crypto):,} samples  |  {df_crypto['label'].mean()*100:.1f}% correct signals")

    X_c = df_crypto[FEATURES].fillna(0)
    y_c = df_crypto["label"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
    )
    model_crypto, name_c, acc_c, cv_c, base_c = train_and_evaluate(
        "CRYPTO", X_train_c, X_test_c, y_train_c, y_test_c
    )

    print("\n" + "="*60)
    print("\n  Loading stock training data...")
    df_stocks = pd.read_csv(os.path.join(PROJECT_DIR, "training_stocks_v3.csv"))
    print(f"  {len(df_stocks):,} samples  |  {df_stocks['label'].mean()*100:.1f}% correct signals")

    X_s = df_stocks[FEATURES].fillna(0)
    y_s = df_stocks["label"]
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
    )
    model_stocks, name_s, acc_s, cv_s, base_s = train_and_evaluate(
        "STOCKS", X_train_s, X_test_s, y_train_s, y_test_s
    )

    crypto_path = os.path.join(PROJECT_DIR, "model_crypto.pkl")
    stocks_path = os.path.join(PROJECT_DIR, "model_stocks.pkl")
    info_path   = os.path.join(PROJECT_DIR, "model_info_v2.json")

    with open(crypto_path, "wb") as f: pickle.dump(model_crypto, f)
    with open(stocks_path, "wb") as f: pickle.dump(model_stocks, f)

    info = {
        "crypto": {"model": name_c, "accuracy": round(acc_c*100,2), "cv": round(cv_c*100,2), "baseline": round(base_c,2), "edge": round(acc_c*100-base_c,2)},
        "stocks": {"model": name_s, "accuracy": round(acc_s*100,2), "cv": round(cv_s*100,2), "baseline": round(base_s,2), "edge": round(acc_s*100-base_s,2)},
        "features": FEATURES,
        "trained_on": pd.Timestamp.now().isoformat(),
    }
    with open(info_path, "w") as f: json.dump(info, f, indent=2)

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  Crypto model:  {name_c:<20} {acc_c*100:.1f}% acc  +{acc_c*100-base_c:.1f}% edge")
    print(f"  Stocks model:  {name_s:<20} {acc_s*100:.1f}% acc  +{acc_s*100-base_s:.1f}% edge")
    print(f"\n  Saved: model_crypto.pkl  model_stocks.pkl  model_info_v2.json\n")

if __name__ == "__main__":
    main()
