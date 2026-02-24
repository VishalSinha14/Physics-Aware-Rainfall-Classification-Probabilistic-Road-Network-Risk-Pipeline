"""
Reliability Curves — All Models
---------------------------------
Plots calibration (reliability) diagrams for:
  - Logistic Regression
  - Ensemble RF (0h, nowcast)
  - Ensemble RF (30m)
  - Ensemble RF (60m)

Each curve compares predicted probabilities to actual observed frequencies.
A perfectly calibrated model follows the diagonal line.

Outputs → results/scientific/reliability_all_models.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

os.makedirs("results/scientific", exist_ok=True)

FEATURES = [
    "t2m_c", "wind_speed", "sp", "swvl1",
    "rain_lag1", "rain_lag2", "rain_roll3", "rain_roll6"
]


def temporal_test_split(df):
    df["time"] = pd.to_datetime(df["time"])
    return df[df["time"] >= "2022-08-16"]


def get_ensemble_probs(model_dir, n_models, X_test):
    """Load saved RF models and average probabilities."""
    all_probs = []
    for i in range(1, n_models + 1):
        path = os.path.join(model_dir, f"rf_model_{i}.pkl")
        if not os.path.exists(path):
            break
        rf = joblib.load(path)
        feats_needed = [f for f in FEATURES if f in rf.feature_names_in_ if hasattr(rf,"feature_names_in_")]
        if not feats_needed:
            # use all features
            probs = rf.predict_proba(X_test)[:, 1]
        else:
            probs = rf.predict_proba(X_test[:, :rf.n_features_in_])[:, 1]
        all_probs.append(probs)
    if all_probs:
        return np.mean(all_probs, axis=0)
    return None


def main():
    print("=" * 60)
    print("Reliability Curves — All Models")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Reliability (Calibration) Diagrams\n"
                 "Perfect calibration = diagonal line",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    models_config = [
        {
            "name": "Logistic Regression",
            "color": "#E74C3C",
            "csv": "data/processed/era5_leadtime_0h.csv",
            "model_type": "lr",
        },
        {
            "name": "Ensemble RF (0h, nowcast)",
            "color": "#2ECC71",
            "csv": "data/processed/era5_leadtime_0h.csv",
            "model_type": "rf",
            "model_dir": "models/bootstrap_models",
            "n_models": 5,  # use first 5 for speed
        },
        {
            "name": "Ensemble RF (+30 min)",
            "color": "#3498DB",
            "csv": "data/processed/era5_leadtime_30m.csv",
            "model_type": "rf",
            "model_dir": "models/leadtime",
            "n_models": 5,
            "model_prefix": "rf_30m_model_",
        },
        {
            "name": "Ensemble RF (+60 min)",
            "color": "#9B59B6",
            "csv": "data/processed/era5_leadtime_60m.csv",
            "model_type": "rf",
            "model_dir": "models/leadtime",
            "n_models": 5,
            "model_prefix": "rf_60m_model_",
        },
    ]

    results = {}

    for idx, cfg in enumerate(models_config):
        ax = axes_flat[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect", alpha=0.5)

        csv = cfg["csv"]
        if not os.path.exists(csv):
            ax.set_title(f"{cfg['name']}\n(data not found)")
            ax.text(0.5, 0.5, "Run engineer_lead_labels.py first",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        df = pd.read_csv(csv)
        test = temporal_test_split(df)
        feats = [f for f in FEATURES if f in df.columns]
        X_test = test[feats].values
        y_test = test["heavy_rain"].values

        if cfg["model_type"] == "lr":
            # Train LR inline on test-split train data for reliability
            train = df[pd.to_datetime(df["time"]) < "2022-08-01"]
            X_tr = train[feats].values
            y_tr = train["heavy_rain"].values
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_test)
            lr = LogisticRegression(C=0.1, class_weight="balanced",
                                    max_iter=1000, solver="lbfgs")
            lr.fit(X_tr_s, y_tr)
            probs = lr.predict_proba(X_te_s)[:, 1]
        else:
            # Load RF models
            model_dir = cfg["model_dir"]
            prefix = cfg.get("model_prefix", "rf_model_")
            all_probs = []
            for i in range(1, cfg["n_models"] + 1):
                path = os.path.join(model_dir, f"{prefix}{i}.pkl")
                if not os.path.exists(path):
                    continue
                rf = joblib.load(path)
                n_feat = rf.n_features_in_
                p = rf.predict_proba(X_test[:, :n_feat])[:, 1]
                all_probs.append(p)
            if not all_probs:
                ax.set_title(f"{cfg['name']}\n(models not found)")
                continue
            probs = np.mean(all_probs, axis=0)

        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_test, probs)

        ax.plot(mean_pred, frac_pos, "o-", color=cfg["color"],
                lw=2.5, ms=8, markeredgecolor="white", markeredgewidth=1.5,
                label=f"Brier={brier:.5f}")
        ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.1, color=cfg["color"])
        ax.set_xlabel("Mean Predicted Probability", fontsize=9)
        ax.set_ylabel("Observed Frequency", fontsize=9)
        ax.set_title(cfg["name"], fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        results[cfg["name"]] = {
            "brier": float(brier),
            "n_bins": len(frac_pos),
        }
        print(f"  {cfg['name']}: Brier={brier:.5f}")

    plt.tight_layout()
    plt.savefig("results/scientific/reliability_all_models.png", dpi=120, bbox_inches="tight")
    plt.close()

    with open("results/scientific/reliability_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Saved → results/scientific/reliability_all_models.png")
    print("✅ Saved → results/scientific/reliability_results.json")


if __name__ == "__main__":
    main()
