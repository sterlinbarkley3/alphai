#!/usr/bin/env python3
"""
train_model.py — Trains a real ML model on labeled trading data
Replaces the hardcoded scoring in brain.py with learned predictions.

Usage:
    python3 train_model.py
Output:
    model.pkl — trained model file used by brain.py
"""

import os, json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

DATA_PATH  = "/Users/mythreeboyz/pythonuh/ai trader/training_data.csv"
MODEL_PATH = "/Users/mythreeboyz/pythonuh/ai trader/model.pkl"
INFO_PATH  = "/Users/mythreeboyz/pythonuh/ai trader/model_info.json"

FEATURES = [
    "ma_cross_pct",
    "momentum",
    "volatility",
    "price_vs_long",
    "score",
]

def main():
    print("\n  Loading training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} samples loaded")
    print(f"  Label distribution:")
    print(f"    Correct signals: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"    Wrong signals:   {(1-df['label']).sum():,} ({(1-df['label']).mean()*100:.1f}%)")

    # Split by asset type so we can see performance on each
    print("\n  Accuracy by asset type in training data:")
    for atype in ["CRYPTO", "STOCK"]:
        sub = df[df["asset_type"] == atype]
        print(f"    {atype}: {sub['label'].mean()*100:.1f}% correct signals ({len(sub):,} samples)")

    # Features and labels
    X = df[FEATURES]
    y = df["label"]

    # Train/test split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Training on {len(X_train):,} samples, testing on {len(X_test):,}...")

    # Train Random Forest
    print("\n  [1/2] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"        Accuracy: {rf_acc*100:.1f}%")

    # Train Gradient Boosting
    print("\n  [2/2] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"        Accuracy: {gb_acc*100:.1f}%")

    # Pick the better model
    if rf_acc >= gb_acc:
        best_model = rf
        best_name  = "Random Forest"
        best_acc   = rf_acc
    else:
        best_model = gb
        best_name  = "Gradient Boosting"
        best_acc   = gb_acc

    print(f"\n  Winner: {best_name} ({best_acc*100:.1f}% accuracy)")

    # Feature importance
    print("\n  What the model learned matters most:")
    importances = zip(FEATURES, best_model.feature_importances_)
    for feat, imp in sorted(importances, key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"    {feat:<20} {bar} {imp*100:.1f}%")

    # Detailed report
    print("\n  Detailed performance:")
    print(classification_report(y_test, best_model.predict(X_test),
                                 target_names=["Wrong call", "Correct call"]))

    # Baseline comparison (what if we just always said "correct"?)
    baseline = y_test.mean() * 100
    print(f"  Baseline (always predict correct): {baseline:.1f}%")
    print(f"  Model accuracy:                    {best_acc*100:.1f}%")
    print(f"  Edge over baseline:                +{best_acc*100 - baseline:.1f}%")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    # Save model info
    info = {
        "model":     best_name,
        "accuracy":  round(best_acc * 100, 2),
        "baseline":  round(baseline, 2),
        "edge":      round(best_acc * 100 - baseline, 2),
        "features":  FEATURES,
        "samples":   len(df),
        "trained_on": pd.Timestamp.now().isoformat(),
    }
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Model saved to model.pkl")
    print(f"  Next step: run python3 ai_brain.py to see AI-powered signals\n")

if __name__ == "__main__":
    main()
