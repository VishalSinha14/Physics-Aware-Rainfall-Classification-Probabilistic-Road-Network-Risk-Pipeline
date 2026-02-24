"""
Logistic Regression Baseline
------------------------------
Trains a Logistic Regression model on the same temporal split as the
ensemble RF for fair comparison. Computes POD, FAR, CSI, ROC-AUC,
PR-AUC, Brier Score, and reliability diagram. Runs McNemar test vs
ensemble RF predictions.

Outputs → results/baselines/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, precision_recall_curve,
    roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RESULTS_DIR = "results/baselines"
THRESHOLD = 10  # mm/hr heavy rain

FEATURES = [
    "t2m_c", "wind_speed", "sp", "swvl1",
    "rain_lag1", "rain_lag2", "rain_roll3", "rain_roll6"
]

os.makedirs(RESULTS_DIR, exist_ok=True)


def temporal_split(df):
    """Same temporal split as Phase 6."""
    df["time"] = pd.to_datetime(df["time"])
    train = df[df["time"] < "2022-08-01"]
    test  = df[df["time"] >= "2022-08-16"]
    return train, test


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute POD, FAR, CSI, ROC-AUC, PR-AUC, Brier."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()
    pod   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far   = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi   = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    roc   = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return dict(POD=pod, FAR=far, CSI=csi, ROC_AUC=roc,
                PR_AUC=prauc, Brier=brier,
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar test: a=correct & b=wrong vs a=wrong & b=correct."""
    a_right = (pred_a == y_true)
    b_right = (pred_b == y_true)
    n01 = ((a_right) & (~b_right)).sum()   # A right, B wrong
    n10 = ((~a_right) & (b_right)).sum()   # A wrong, B right
    if (n01 + n10) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_val = 1 - chi2.cdf(chi2_stat, df=1)
    return {"statistic": float(chi2_stat), "p_value": float(p_val),
            "significant": bool(p_val < 0.05)}


def main():
    print("=" * 60)
    print("Logistic Regression Baseline")
    print("=" * 60)

    print("\nLoading data...")
    df = pd.read_csv("data/processed/era5_2022_monsoon_temporal.csv")
    df["heavy_rain"] = (df["tp_mm"] >= THRESHOLD).astype(int)

    train, test = temporal_split(df)
    print(f"  Train: {len(train):,}  Test: {len(test):,}")
    print(f"  Train positive rate: {train['heavy_rain'].mean():.4f}")
    print(f"  Test  positive rate: {test['heavy_rain'].mean():.4f}")

    X_train = train[FEATURES].values
    y_train = train["heavy_rain"].values
    X_test  = test[FEATURES].values
    y_test  = test["heavy_rain"].values

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Cross-validate C parameter ──────────────────────────────
    print("\nCross-validating regularization strength (C)...")
    C_grid = [0.01, 0.1, 1.0, 10.0]
    cv = StratifiedKFold(n_splits=3, shuffle=False)
    best_C, best_auc = 0.1, 0.0
    for C in C_grid:
        aucs = []
        for fold_tr, fold_va in cv.split(X_train_s, y_train):
            m = LogisticRegression(
                C=C, class_weight="balanced",
                max_iter=1000, solver="lbfgs"
            )
            m.fit(X_train_s[fold_tr], y_train[fold_tr])
            p = m.predict_proba(X_train_s[fold_va])[:, 1]
            aucs.append(roc_auc_score(y_train[fold_va], p))
        mean_auc = np.mean(aucs)
        print(f"  C={C:.2f}  CV ROC-AUC={mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc, best_C = mean_auc, C
    print(f"  → Best C = {best_C}")

    # ── Train final model ────────────────────────────────────────
    print("\nTraining final Logistic Regression...")
    lr = LogisticRegression(
        C=best_C, class_weight="balanced",
        max_iter=1000, solver="lbfgs"
    )
    lr.fit(X_train_s, y_train)
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, lr_prob)
    print("\n  Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # ── Feature coefficients ─────────────────────────────────────
    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": lr.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\n  Feature coefficients:")
    print(coef_df.to_string(index=False))

    # ── McNemar test vs ensemble RF ──────────────────────────────
    print("\nRunning McNemar test vs Ensemble RF...")
    ensemble_pred_path = "data/processed/monsoon_ensemble_predictions_10mm.csv"
    mcnemar = None
    if os.path.exists(ensemble_pred_path):
        ens_df = pd.read_csv(ensemble_pred_path)
        # Align on last len(y_test) rows (test set)
        if "true_label" in ens_df.columns:
            ens_test = ens_df.tail(len(y_test))
            ens_pred = (ens_test["mean_probability"].values >= 0.5).astype(int)
            mcnemar = mcnemar_test(y_test, lr_pred, ens_pred)
            print(f"  χ² = {mcnemar['statistic']:.4f}, p = {mcnemar['p_value']:.6f}")
            print(f"  Significant difference: {mcnemar['significant']}")
    else:
        print("  Ensemble predictions not found — skipping McNemar test")

    # -- Save results FIRST (before any matplotlib that might crash) ------
    out = {
        "model": "Logistic Regression",
        "best_C": best_C,
        "metrics": metrics,
        "coefficients": coef_df.to_dict(orient="records"),
        "mcnemar_vs_ensemble": mcnemar
    }
    with open(f"{RESULTS_DIR}/lr_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n==> Saved: {RESULTS_DIR}/lr_results.json")

    # -- Generate plot (wrap in try/except - matplotlib can crash on Windows)
    try:
        frac_pos, mean_pred = calibration_curve(y_test, lr_prob, n_bins=10, strategy="uniform")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Logistic Regression Baseline Analysis", fontsize=14, fontweight="bold")

        axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        axes[0].plot(mean_pred, frac_pos, "o-", color="#E74C3C", lw=2, ms=7, label="Logistic Regression")
        axes[0].set_xlabel("Mean Predicted Probability")
        axes[0].set_ylabel("Observed Frequency")
        axes[0].set_title("Reliability Diagram")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        fpr, tpr, _ = roc_curve(y_test, lr_prob)
        axes[1].plot(fpr, tpr, color="#E74C3C", lw=2,
                     label=f"LR  AUC={metrics['ROC_AUC']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/lr_analysis.png", dpi=120, bbox_inches="tight")
        plt.close("all")
        print(f"==> Saved: {RESULTS_DIR}/lr_analysis.png")
    except Exception as e:
        print(f"WARNING: Plot failed ({e}). JSON was already saved.")

    return out


if __name__ == "__main__":
    main()
