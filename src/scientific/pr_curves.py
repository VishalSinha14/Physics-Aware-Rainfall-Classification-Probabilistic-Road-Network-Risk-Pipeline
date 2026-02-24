"""
Precision-Recall Curves — All Models
--------------------------------------
PR curves are more informative than ROC curves under severe class imbalance.

Plots PR curves and optimal decision thresholds for:
  - Logistic Regression  (baseline)
  - Ensemble RF (0h)
  - Ensemble RF (+30m)
  - Ensemble RF (+60m)

Outputs → results/scientific/pr_curves_comparison.png
          results/scientific/pr_results.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

os.makedirs("results/scientific", exist_ok=True)

FEATURES = [
    "t2m_c", "wind_speed", "sp", "swvl1",
    "rain_lag1", "rain_lag2", "rain_roll3", "rain_roll6"
]


def temporal_split(df):
    df["time"] = pd.to_datetime(df["time"])
    train = df[df["time"] < "2022-08-01"]
    test  = df[df["time"] >= "2022-08-16"]
    return train, test


def optimal_threshold(precision, recall, thresholds):
    """Threshold that maximises F1 score."""
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = np.argmax(f1)
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx]), float(f1[best_idx])


def main():
    print("=" * 60)
    print("Precision-Recall Curves — All Models")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Precision-Recall Curves (Better than ROC for Imbalanced Data)",
                 fontsize=13, fontweight="bold")

    ax_pr   = axes[0]  # PR curves overlaid
    ax_thr  = axes[1]  # Optimal threshold comparison

    models_config = [
        {
            "name": "Logistic Regression",
            "color": "#E74C3C",
            "csv": "data/processed/era5_leadtime_0h.csv",
            "type": "lr",
        },
        {
            "name": "Ensemble RF (0h)",
            "color": "#2ECC71",
            "csv": "data/processed/era5_leadtime_0h.csv",
            "type": "rf",
            "model_dir": "models/bootstrap_models",
            "model_prefix": "rf_model_",
            "n": 5,
        },
        {
            "name": "Ensemble RF (+30m)",
            "color": "#3498DB",
            "csv": "data/processed/era5_leadtime_30m.csv",
            "type": "rf",
            "model_dir": "models/leadtime",
            "model_prefix": "rf_30m_model_",
            "n": 5,
        },
        {
            "name": "Ensemble RF (+60m)",
            "color": "#9B59B6",
            "csv": "data/processed/era5_leadtime_60m.csv",
            "type": "rf",
            "model_dir": "models/leadtime",
            "model_prefix": "rf_60m_model_",
            "n": 5,
        },
    ]

    results = {}
    opt_data = []

    for cfg in models_config:
        csv = cfg["csv"]
        if not os.path.exists(csv):
            print(f"  ⚠️  {cfg['name']}: {csv} not found")
            continue

        df = pd.read_csv(csv)
        train, test = temporal_split(df)
        feats = [f for f in FEATURES if f in df.columns]
        X_te = test[feats].values
        y_te = test["heavy_rain"].values

        # Get probabilities
        if cfg["type"] == "lr":
            X_tr = train[feats].values
            y_tr = train["heavy_rain"].values
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            lr = LogisticRegression(C=0.1, class_weight="balanced",
                                    max_iter=1000)
            lr.fit(X_tr_s, y_tr)
            probs = lr.predict_proba(X_te_s)[:, 1]
        else:
            all_p = []
            for i in range(1, cfg["n"] + 1):
                path = os.path.join(cfg["model_dir"], f"{cfg['model_prefix']}{i}.pkl")
                if not os.path.exists(path):
                    continue
                rf = joblib.load(path)
                all_p.append(rf.predict_proba(X_te[:, :rf.n_features_in_])[:, 1])
            if not all_p:
                print(f"  ⚠️  {cfg['name']}: no models loaded")
                continue
            probs = np.mean(all_p, axis=0)

        pr_auc = average_precision_score(y_te, probs)
        precision, recall, thresholds = precision_recall_curve(y_te, probs)
        opt_thr, opt_pr, opt_rc, opt_f1 = optimal_threshold(precision, recall, thresholds)

        # Plot PR curve
        ax_pr.plot(recall, precision, color=cfg["color"], lw=2.5,
                   label=f"{cfg['name']} (PR-AUC={pr_auc:.4f})")
        ax_pr.scatter([opt_rc], [opt_pr], color=cfg["color"], s=100,
                      zorder=5, edgecolor="black", linewidth=1)

        results[cfg["name"]] = {
            "PR_AUC": float(pr_auc),
            "optimal_threshold": opt_thr,
            "optimal_precision": opt_pr,
            "optimal_recall": opt_rc,
            "optimal_f1": opt_f1,
        }
        opt_data.append({
            "name": cfg["name"],
            "color": cfg["color"],
            "PR_AUC": pr_auc,
            "opt_f1": opt_f1,
            "opt_thr": opt_thr,
        })
        print(f"  {cfg['name']}: PR-AUC={pr_auc:.4f}, "
              f"optimal threshold={opt_thr:.3f}, F1={opt_f1:.4f}")

    # Finalize PR plot
    ax_pr.set_xlabel("Recall", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    # Add "area = random" baseline
    pos_rate = 0.002  # approximate
    ax_pr.axhline(y=pos_rate, color="gray", linestyle=":", lw=1.5,
                  label=f"Baseline (no skill) ≈ {pos_rate:.3f}")

    # Bar chart: PR-AUC per model
    if opt_data:
        names  = [d["name"].replace(" ", "\n") for d in opt_data]
        prauc  = [d["PR_AUC"] for d in opt_data]
        colors = [d["color"] for d in opt_data]
        bars = ax_thr.bar(range(len(names)), prauc, color=colors, alpha=0.85,
                          edgecolor="white", lw=1.5)
        for bar, v, d in zip(bars, prauc, opt_data):
            ax_thr.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"PR-AUC={v:.4f}\nF1={d['opt_f1']:.4f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax_thr.set_xticks(range(len(names)))
        ax_thr.set_xticklabels(names, fontsize=8)
        ax_thr.set_ylabel("PR-AUC (↑ better)")
        ax_thr.set_title("PR-AUC Comparison\n(higher = better for rare events)")
        ax_thr.grid(True, axis="y", alpha=0.3)
        ax_thr.set_ylim(0, max(prauc) * 1.35)

    plt.tight_layout()
    plt.savefig("results/scientific/pr_curves_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()

    with open("results/scientific/pr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Saved → results/scientific/pr_curves_comparison.png")
    print("✅ Saved → results/scientific/pr_results.json")


if __name__ == "__main__":
    main()
