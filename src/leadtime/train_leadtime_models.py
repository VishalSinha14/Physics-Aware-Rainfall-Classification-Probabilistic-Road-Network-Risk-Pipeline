"""
Lead-Time Model Training
--------------------------
Trains three bootstrap RF models (10 models each for speed) on:
  - 0h label  → current heavy rain
  - 30m label → rain 1 ERA5 step ahead
  - 60m label → rain 2 ERA5 steps ahead

Uses identical temporal split as Phase 6 (train: Jun-Jul, test: Aug 16-31).

Outputs:
  models/leadtime/rf_*h_model_*.pkl
  results/leadtime/leadtime_metrics.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix
)

os.makedirs("results/leadtime", exist_ok=True)
os.makedirs("models/leadtime", exist_ok=True)

FEATURES = [
    "t2m_c", "wind_speed", "sp", "swvl1",
    "rain_lag1", "rain_lag2", "rain_roll3", "rain_roll6"
]
N_MODELS = 10  # smaller ensemble for speed (still robust)

LEAD_FILES = {
    "0h":  "data/processed/era5_leadtime_0h.csv",
    "30m": "data/processed/era5_leadtime_30m.csv",
    "60m": "data/processed/era5_leadtime_60m.csv",
}


def temporal_split(df):
    df["time"] = pd.to_datetime(df["time"])
    train = df[df["time"] < "2022-08-01"]
    test  = df[df["time"] >= "2022-08-16"]
    return train, test


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()
    pod   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far   = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi   = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    return dict(
        POD=float(pod), FAR=float(far), CSI=float(csi),
        ROC_AUC=float(roc_auc_score(y_true, y_prob)),
        PR_AUC=float(average_precision_score(y_true, y_prob)),
        Brier=float(brier_score_loss(y_true, y_prob)),
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
        train_positive_rate=None,   # filled below
        test_positive_rate=None,
    )


def train_leadtime_model(lead_name, csv_path):
    print(f"\n{'─'*50}")
    print(f"Lead = {lead_name}  |  {csv_path}")
    if not os.path.exists(csv_path):
        print(f"  ❌ File not found — run engineer_lead_labels.py first")
        return None

    df = pd.read_csv(csv_path)
    train, test = temporal_split(df)

    # Filter features that exist
    feats = [f for f in FEATURES if f in df.columns]
    X_train = train[feats].values
    y_train = train["heavy_rain"].values
    X_test  = test[feats].values
    y_test  = test["heavy_rain"].values

    print(f"  Train: {len(X_train):,}  pos={y_train.mean():.4f}")
    print(f"  Test : {len(X_test):,}  pos={y_test.mean():.4f}")

    all_probs = []
    for i in range(N_MODELS):
        print(f"  Training model {i+1}/{N_MODELS}...", end="\r")
        boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[boot_idx]
        y_boot = y_train[boot_idx]

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            class_weight="balanced", n_jobs=-1,
            random_state=i
        )
        rf.fit(X_boot, y_boot)
        model_path = f"models/leadtime/rf_{lead_name}_model_{i+1}.pkl"
        joblib.dump(rf, model_path)
        probs = rf.predict_proba(X_test)[:, 1]
        all_probs.append(probs)

    print()  # newline after \r
    mean_prob = np.mean(all_probs, axis=0)
    std_prob  = np.std(all_probs, axis=0)

    m = compute_metrics(y_test, mean_prob)
    m["train_positive_rate"] = float(y_train.mean())
    m["test_positive_rate"]  = float(y_test.mean())
    m["mean_uncertainty_std"] = float(std_prob.mean())

    print(f"  ROC-AUC: {m['ROC_AUC']:.4f}  CSI: {m['CSI']:.4f}  "
          f"Brier: {m['Brier']:.6f}  POD: {m['POD']:.4f}")
    return m


def main():
    print("=" * 60)
    print("Lead-Time Model Training")
    print("=" * 60)

    all_metrics = {}
    for lead_name, csv_path in LEAD_FILES.items():
        m = train_leadtime_model(lead_name, csv_path)
        if m:
            all_metrics[lead_name] = m

    with open("results/leadtime/leadtime_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✅ Saved → results/leadtime/leadtime_metrics.json")
    print(f"✅ Models saved to models/leadtime/ ({N_MODELS * len(all_metrics)} total)")

    # Print summary table
    print("\n--- Summary ---")
    print(f"{'Lead':>6}  {'POD':>7}  {'FAR':>7}  {'CSI':>7}  {'ROC-AUC':>9}  {'Brier':>10}")
    for lead, m in all_metrics.items():
        print(f"{lead:>6}  {m['POD']:>7.4f}  {m['FAR']:>7.4f}  {m['CSI']:>7.4f}  "
              f"{m['ROC_AUC']:>9.4f}  {m['Brier']:>10.6f}")


if __name__ == "__main__":
    main()
