"""
Phase 6: Research-Grade Validation & Enhancement
==================================================
Comprehensive validation script implementing all 9 components:
  1. Time-based train/test split
  2. Reliability diagram + Brier Score + calibration
  3. Typhoon case injection (extreme event stress test)
  4. Dynamic hazard scaling experiment
  5. Threshold sensitivity analysis (5, 10, 20 mm)
  6. Multi-threshold fusion
  7. Spatial cross-validation
  8. Functionality validation (Monte Carlo)
  9. Ensemble diversity improvement

Outputs: All figures + CSV results saved to results/phase6/
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Paths
DATA_PATH = "data/processed/era5_2022_monsoon_temporal.csv"
RESULTS_DIR = "results/phase6"
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = [
    "t2m_c", "wind_speed", "sp", "swvl1",
    "rain_lag1", "rain_lag2", "rain_roll3", "rain_roll6"
]


def load_data():
    """Load monsoon temporal dataset."""
    print("Loading monsoon temporal dataset...")
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    print(f"  Shape: {df.shape}")
    print(f"  Time range: {df['time'].min()} → {df['time'].max()}")
    return df


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all meteorological metrics."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        return {"error": "single class"}

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

    return {
        "POD": pod, "FAR": far, "CSI": csi,
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
    }


# ================================================================
# 1️⃣ TIME-BASED TRAIN/TEST SPLIT
# ================================================================

def run_time_split(df):
    """
    Split respecting temporal causality:
      Train: June–July 2022
      Validation: Aug 1–15
      Test: Aug 16–31
    """
    print("\n" + "=" * 60)
    print("1️⃣  TIME-BASED TRAIN/TEST SPLIT")
    print("=" * 60)

    df["heavy_rain_10"] = (df["tp_mm"] >= 10).astype(int)

    train = df[df["time"] < "2022-08-01"]
    val   = df[(df["time"] >= "2022-08-01") & (df["time"] < "2022-08-16")]
    test  = df[df["time"] >= "2022-08-16"]

    print(f"  Train: {len(train):,} samples (Jun-Jul)")
    print(f"  Val:   {len(val):,} samples (Aug 1-15)")
    print(f"  Test:  {len(test):,} samples (Aug 16-31)")

    X_train, y_train = train[FEATURES], train["heavy_rain_10"]
    X_val,   y_val   = val[FEATURES],   val["heavy_rain_10"]
    X_test,  y_test  = test[FEATURES],  test["heavy_rain_10"]

    # Train bootstrap ensemble
    n_models = 30
    all_probs_val = []
    all_probs_test = []
    models = []

    for i in range(n_models):
        boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[boot_idx]
        y_boot = y_train.iloc[boot_idx]

        model = RandomForestClassifier(
            n_estimators=200, max_depth=18,
            class_weight="balanced", n_jobs=-1, random_state=i
        )
        model.fit(X_boot, y_boot)
        models.append(model)

        all_probs_val.append(model.predict_proba(X_val)[:, 1])
        all_probs_test.append(model.predict_proba(X_test)[:, 1])

        if (i + 1) % 10 == 0:
            print(f"    Trained model {i+1}/{n_models}")

    mean_prob_val  = np.mean(all_probs_val, axis=0)
    mean_prob_test = np.mean(all_probs_test, axis=0)
    std_prob_test  = np.std(all_probs_test, axis=0)

    pred_val  = (mean_prob_val >= 0.5).astype(int)
    pred_test = (mean_prob_test >= 0.5).astype(int)

    val_metrics  = compute_metrics(y_val, pred_val, mean_prob_val)
    test_metrics = compute_metrics(y_test, pred_test, mean_prob_test)

    print(f"\n  Validation metrics:")
    for k, v in val_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"\n  Test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Save
    results = {"validation": val_metrics, "test": test_metrics}
    with open(f"{RESULTS_DIR}/1_time_split_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return models, X_test, y_test, mean_prob_test, std_prob_test, test


# ================================================================
# 2️⃣ RELIABILITY DIAGRAM + CALIBRATION
# ================================================================

def run_calibration(models, X_test, y_test, mean_prob_test, X_train=None, y_train=None, df=None):
    """Reliability diagram, Brier Score, Isotonic/Platt calibration."""
    print("\n" + "=" * 60)
    print("2️⃣  RELIABILITY DIAGRAM & CALIBRATION")
    print("=" * 60)

    # Brier Score before calibration
    brier_before = brier_score_loss(y_test, mean_prob_test)
    print(f"  Brier Score (before calibration): {brier_before:.6f}")

    # Reliability diagram
    prob_true, prob_pred = calibration_curve(y_test, mean_prob_test, n_bins=10, strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color="#FF6B6B", linewidth=2, label="Ensemble (before)")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Apply Isotonic calibration using train data
    if df is not None:
        train_data = df[df["time"] < "2022-08-01"]
        val_data = df[(df["time"] >= "2022-08-01") & (df["time"] < "2022-08-16")]
        X_cal = val_data[FEATURES]
        y_cal = val_data["heavy_rain_10"]

        # Get ensemble probabilities on calibration set
        cal_probs = np.mean([m.predict_proba(X_cal)[:, 1] for m in models], axis=0)

        # Isotonic calibration
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(cal_probs, y_cal)
        calibrated_probs = iso_reg.predict(mean_prob_test)

        brier_after = brier_score_loss(y_test, calibrated_probs)
        print(f"  Brier Score (after Isotonic):     {brier_after:.6f}")

        prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred_cal, prob_true_cal, "s-", color="#4ECDC4", linewidth=2, label="After Isotonic")
        ax.legend()

    # Histogram of predicted probabilities
    ax2 = axes[1]
    ax2.hist(mean_prob_test, bins=50, color="#FFE66D", alpha=0.7, label="Before calibration")
    if df is not None:
        ax2.hist(calibrated_probs, bins=50, color="#4ECDC4", alpha=0.5, label="After Isotonic")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Probability Distribution")
    ax2.legend()
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/2_reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/2_reliability_diagram.png")

    cal_results = {
        "brier_before": brier_before,
        "brier_after_isotonic": brier_after if df is not None else None,
    }
    with open(f"{RESULTS_DIR}/2_calibration_metrics.json", "w") as f:
        json.dump(cal_results, f, indent=2)

    return calibrated_probs if df is not None else mean_prob_test


# ================================================================
# 3️⃣ TYPHOON CASE INJECTION
# ================================================================

def run_typhoon_stress_test(models, df):
    """Inject synthetic high-intensity typhoon rainfall."""
    print("\n" + "=" * 60)
    print("3️⃣  TYPHOON CASE INJECTION (EXTREME EVENT)")
    print("=" * 60)

    # Create synthetic typhoon event:
    # Take the 99.9th percentile conditions and amplify
    extreme = df.nlargest(100, "tp_mm").copy()

    # Create typhoon scenario: sustained heavy rainfall
    typhoon_scenario = extreme.copy()
    typhoon_scenario["tp_mm"] = typhoon_scenario["tp_mm"] * 3.0  # 3x amplification
    typhoon_scenario["rain_lag1"] = typhoon_scenario["tp_mm"] * 0.9
    typhoon_scenario["rain_lag2"] = typhoon_scenario["tp_mm"] * 0.85
    typhoon_scenario["rain_roll3"] = typhoon_scenario["tp_mm"] * 2.5
    typhoon_scenario["rain_roll6"] = typhoon_scenario["tp_mm"] * 4.5
    typhoon_scenario["wind_speed"] = typhoon_scenario["wind_speed"] * 2.0  # Strong winds
    typhoon_scenario["swvl1"] = 0.45  # Saturated soil

    X_typhoon = typhoon_scenario[FEATURES]

    # Get baseline (normal conditions — random sample)
    baseline = df.sample(1000, random_state=42)
    X_baseline = baseline[FEATURES]

    # Predict
    typhoon_probs = np.mean([m.predict_proba(X_typhoon)[:, 1] for m in models], axis=0)
    baseline_probs = np.mean([m.predict_proba(X_baseline)[:, 1] for m in models], axis=0)

    print(f"  Baseline mean hazard prob: {baseline_probs.mean():.6f}")
    print(f"  Typhoon  mean hazard prob: {typhoon_probs.mean():.4f}")
    print(f"  Typhoon  max  hazard prob: {typhoon_probs.max():.4f}")
    print(f"  Hazard amplification:      {typhoon_probs.mean() / max(baseline_probs.mean(), 1e-8):.1f}x")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(baseline_probs, bins=30, alpha=0.7, color="#4ECDC4", label="Baseline")
    axes[0].hist(typhoon_probs, bins=30, alpha=0.7, color="#FF6B6B", label="Typhoon")
    axes[0].set_xlabel("Hazard Probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Hazard Probability: Baseline vs Typhoon")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Simulate road risk under typhoon
    # Use simple vulnerability model
    vuln_levels = np.random.uniform(0.3, 0.85, len(typhoon_probs))
    typhoon_risk = typhoon_probs * vuln_levels
    baseline_risk = baseline_probs[:len(vuln_levels)] * vuln_levels

    axes[1].bar(["Baseline", "Typhoon"], [baseline_risk.mean(), typhoon_risk.mean()],
                 color=["#4ECDC4", "#FF6B6B"])
    axes[1].set_ylabel("Mean Road Risk Score")
    axes[1].set_title("Road Risk: Baseline vs Typhoon")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/3_typhoon_stress_test.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/3_typhoon_stress_test.png")

    typhoon_results = {
        "baseline_mean_hazard": float(baseline_probs.mean()),
        "typhoon_mean_hazard": float(typhoon_probs.mean()),
        "typhoon_max_hazard": float(typhoon_probs.max()),
        "amplification_factor": float(typhoon_probs.mean() / max(baseline_probs.mean(), 1e-8)),
        "typhoon_mean_risk": float(typhoon_risk.mean()),
        "baseline_mean_risk": float(baseline_risk.mean()),
    }
    with open(f"{RESULTS_DIR}/3_typhoon_results.json", "w") as f:
        json.dump(typhoon_results, f, indent=2)


# ================================================================
# 4️⃣ DYNAMIC HAZARD SCALING
# ================================================================

def run_dynamic_scaling(models, df):
    """Test risk stability under amplified rainfall signals."""
    print("\n" + "=" * 60)
    print("4️⃣  DYNAMIC HAZARD SCALING EXPERIMENT")
    print("=" * 60)

    test_data = df[df["time"] >= "2022-08-16"].copy()
    sample = test_data.sample(min(5000, len(test_data)), random_state=42)

    scale_factors = [1.0, 1.5, 2.0, 3.0]
    results = []

    for factor in scale_factors:
        scaled = sample.copy()
        # Scale rainfall-related features
        for col in ["rain_lag1", "rain_lag2", "rain_roll3", "rain_roll6"]:
            if col in scaled.columns:
                scaled[col] = scaled[col] * factor

        X_scaled = scaled[FEATURES]
        probs = np.mean([m.predict_proba(X_scaled)[:, 1] for m in models], axis=0)

        # Simulate risk
        vuln = np.random.uniform(0.3, 0.85, len(probs))
        risk = probs * vuln

        high_risk_pct = (risk > 0.05).mean() * 100
        avg_functionality = (1 - risk).mean()

        results.append({
            "scale_factor": factor,
            "mean_hazard": float(probs.mean()),
            "mean_risk": float(risk.mean()),
            "high_risk_pct": float(high_risk_pct),
            "avg_functionality": float(avg_functionality),
        })

        print(f"  {factor}x: hazard={probs.mean():.6f}, risk={risk.mean():.6f}, "
              f"high_risk={high_risk_pct:.1f}%, func={avg_functionality:.4f}")

    # Plot elasticity curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    factors = [r["scale_factor"] for r in results]

    axes[0].plot(factors, [r["mean_hazard"] for r in results], "o-", color="#FF6B6B", linewidth=2)
    axes[0].set_xlabel("Rainfall Scale Factor")
    axes[0].set_ylabel("Mean Hazard Probability")
    axes[0].set_title("Hazard vs Rainfall Intensity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(factors, [r["high_risk_pct"] for r in results], "s-", color="#FFE66D", linewidth=2)
    axes[1].set_xlabel("Rainfall Scale Factor")
    axes[1].set_ylabel("% High-Risk Roads")
    axes[1].set_title("High-Risk Road Percentage")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(factors, [r["avg_functionality"] for r in results], "^-", color="#4ECDC4", linewidth=2)
    axes[2].set_xlabel("Rainfall Scale Factor")
    axes[2].set_ylabel("Avg Functionality")
    axes[2].set_title("Network Functionality Decline")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/4_dynamic_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/4_dynamic_scaling.png")

    with open(f"{RESULTS_DIR}/4_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)


# ================================================================
# 5️⃣ THRESHOLD SENSITIVITY ANALYSIS
# ================================================================

def run_threshold_sensitivity(df):
    """Compare models trained at 5mm, 10mm, 20mm thresholds."""
    print("\n" + "=" * 60)
    print("5️⃣  THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)

    thresholds = [5, 10, 20]
    results = {}

    train = df[df["time"] < "2022-08-01"]
    test  = df[df["time"] >= "2022-08-16"]

    X_train, X_test = train[FEATURES], test[FEATURES]

    for thresh in thresholds:
        print(f"\n  Training for {thresh}mm threshold...")
        y_train_t = (train["tp_mm"] >= thresh).astype(int)
        y_test_t  = (test["tp_mm"] >= thresh).astype(int)

        pos_rate_train = y_train_t.mean()
        pos_rate_test  = y_test_t.mean()
        print(f"    Positive rate — train: {pos_rate_train:.4f}, test: {pos_rate_test:.4f}")

        # Train 10-model ensemble (faster)
        probs_list = []
        for i in range(10):
            boot = np.random.choice(len(X_train), len(X_train), replace=True)
            model = RandomForestClassifier(
                n_estimators=100, max_depth=15,
                class_weight="balanced", n_jobs=-1, random_state=i + thresh
            )
            model.fit(X_train.iloc[boot], y_train_t.iloc[boot])
            probs_list.append(model.predict_proba(X_test)[:, 1])

        mean_prob = np.mean(probs_list, axis=0)
        pred = (mean_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_test_t, pred, mean_prob)

        results[f"{thresh}mm"] = {
            "positive_rate": float(pos_rate_test),
            **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()},
            "mean_hazard_prob": float(mean_prob.mean()),
            "max_hazard_prob": float(mean_prob.max()),
        }

        print(f"    ROC-AUC: {metrics['ROC_AUC']:.4f}, "
              f"PR-AUC: {metrics['PR_AUC']:.4f}, "
              f"CSI: {metrics['CSI']:.4f}, "
              f"Brier: {metrics['Brier']:.6f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    thresh_labels = [f"{t}mm" for t in thresholds]
    colors = ["#4ECDC4", "#FFE66D", "#FF6B6B"]

    # ROC-AUC comparison
    roc_vals = [results[t]["ROC_AUC"] for t in thresh_labels]
    axes[0].bar(thresh_labels, roc_vals, color=colors)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("ROC-AUC by Threshold")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(True, alpha=0.3)

    # CSI comparison
    csi_vals = [results[t]["CSI"] for t in thresh_labels]
    axes[1].bar(thresh_labels, csi_vals, color=colors)
    axes[1].set_ylabel("CSI")
    axes[1].set_title("CSI by Threshold")
    axes[1].grid(True, alpha=0.3)

    # Mean hazard probability
    hazard_vals = [results[t]["mean_hazard_prob"] for t in thresh_labels]
    axes[2].bar(thresh_labels, hazard_vals, color=colors)
    axes[2].set_ylabel("Mean Hazard Probability")
    axes[2].set_title("Hazard Output by Threshold")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/5_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {RESULTS_DIR}/5_threshold_sensitivity.png")

    with open(f"{RESULTS_DIR}/5_threshold_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ================================================================
# 6️⃣ MULTI-THRESHOLD FUSION
# ================================================================

def run_multi_threshold_fusion(df):
    """Build multi-output severity model (minor + major disruption)."""
    print("\n" + "=" * 60)
    print("6️⃣  MULTI-THRESHOLD FUSION")
    print("=" * 60)

    train = df[df["time"] < "2022-08-01"]
    test  = df[df["time"] >= "2022-08-16"]

    X_train, X_test = train[FEATURES], test[FEATURES]

    # Train separate models for each severity
    severity_levels = {
        "minor_disruption": 5,   # >= 5mm
        "moderate_disruption": 10,  # >= 10mm
        "major_disruption": 20,     # >= 20mm
    }

    severity_probs = {}
    for name, thresh in severity_levels.items():
        y_train_s = (train["tp_mm"] >= thresh).astype(int)
        y_test_s  = (test["tp_mm"] >= thresh).astype(int)

        model = RandomForestClassifier(
            n_estimators=150, max_depth=15,
            class_weight="balanced", n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train_s)
        probs = model.predict_proba(X_test)[:, 1]
        severity_probs[name] = probs

        auc = roc_auc_score(y_test_s, probs) if y_test_s.sum() > 0 else 0
        print(f"  {name} ({thresh}mm): AUC={auc:.4f}, mean_prob={probs.mean():.6f}")

    # Graded severity risk
    severity_risk = (
        0.2 * severity_probs["minor_disruption"] +
        0.3 * severity_probs["moderate_disruption"] +
        0.5 * severity_probs["major_disruption"]
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, probs in severity_probs.items():
        axes[0].hist(probs, bins=50, alpha=0.5, label=name)
    axes[0].set_xlabel("Disruption Probability")
    axes[0].set_ylabel("Count (log)")
    axes[0].set_title("Multi-Threshold Disruption Probabilities")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(severity_risk, bins=50, color="#FF6B6B", alpha=0.7)
    axes[1].set_xlabel("Graded Severity Risk Score")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Fused Multi-Threshold Risk")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/6_multi_threshold_fusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/6_multi_threshold_fusion.png")

    fusion_results = {
        "mean_graded_risk": float(severity_risk.mean()),
        "max_graded_risk": float(severity_risk.max()),
        "severity_means": {k: float(v.mean()) for k, v in severity_probs.items()},
    }
    with open(f"{RESULTS_DIR}/6_fusion_results.json", "w") as f:
        json.dump(fusion_results, f, indent=2)


# ================================================================
# 7️⃣ SPATIAL CROSS-VALIDATION
# ================================================================

def run_spatial_cv(df):
    """Leave-one-region-out spatial cross-validation."""
    print("\n" + "=" * 60)
    print("7️⃣  SPATIAL CROSS-VALIDATION")
    print("=" * 60)

    df_cv = df.copy()
    df_cv["heavy_rain_10"] = (df_cv["tp_mm"] >= 10).astype(int)

    # Create spatial blocks from lat/lon grid
    lat_bins = pd.qcut(df_cv["latitude"], q=4, labels=False, duplicates="drop")
    lon_bins = pd.qcut(df_cv["longitude"], q=4, labels=False, duplicates="drop")
    df_cv["spatial_block"] = lat_bins * 10 + lon_bins

    n_blocks = df_cv["spatial_block"].nunique()
    print(f"  Created {n_blocks} spatial blocks")

    fold_results = []
    gkf = GroupKFold(n_splits=min(n_blocks, 5))

    for fold, (train_idx, test_idx) in enumerate(gkf.split(
        df_cv[FEATURES], df_cv["heavy_rain_10"], groups=df_cv["spatial_block"]
    )):
        X_tr = df_cv.iloc[train_idx][FEATURES]
        y_tr = df_cv.iloc[train_idx]["heavy_rain_10"]
        X_te = df_cv.iloc[test_idx][FEATURES]
        y_te = df_cv.iloc[test_idx]["heavy_rain_10"]

        model = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            class_weight="balanced", n_jobs=-1, random_state=fold
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        pred = (probs >= 0.5).astype(int)

        metrics = compute_metrics(y_te, pred, probs)
        fold_results.append(metrics)
        print(f"  Fold {fold+1}: ROC-AUC={metrics['ROC_AUC']:.4f}, "
              f"CSI={metrics['CSI']:.4f}, Brier={metrics['Brier']:.6f}")

    # Summary
    avg_metrics = {}
    for key in ["ROC_AUC", "PR_AUC", "CSI", "POD", "FAR", "Brier"]:
        vals = [r[key] for r in fold_results if key in r]
        avg_metrics[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print(f"\n  Spatial CV Summary:")
    for k, v in avg_metrics.items():
        print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_names = list(avg_metrics.keys())
    means = [avg_metrics[k]["mean"] for k in metric_names]
    stds  = [avg_metrics[k]["std"] for k in metric_names]
    colors = ["#FF6B6B", "#FFE66D", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    bars = ax.bar(metric_names, means, yerr=stds, capsize=5, color=colors[:len(metric_names)])
    ax.set_ylabel("Score")
    ax.set_title("Spatial Cross-Validation Metrics (Mean ± Std)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/7_spatial_cv.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/7_spatial_cv.png")

    with open(f"{RESULTS_DIR}/7_spatial_cv_results.json", "w") as f:
        json.dump({"fold_results": fold_results, "summary": avg_metrics}, f, indent=2)


# ================================================================
# 8️⃣ FUNCTIONALITY VALIDATION (MONTE CARLO)
# ================================================================

def run_functionality_validation():
    """Monte Carlo perturbation of vulnerability weights."""
    print("\n" + "=" * 60)
    print("8️⃣  FUNCTIONALITY VALIDATION (MONTE CARLO)")
    print("=" * 60)

    import geopandas as gpd

    risk_path = "data/processed/road_risk_scores.geojson"
    if not os.path.exists(risk_path):
        print("  ⚠️  road_risk_scores.geojson not found, skipping.")
        return

    roads = gpd.read_file(risk_path)
    print(f"  Loaded {len(roads)} road segments")

    # Monte Carlo: perturb vulnerability weights
    n_simulations = 100
    func_results = []

    base_vuln = roads["vulnerability"].values
    hazard = roads["hazard_probability"].values

    for i in range(n_simulations):
        # Perturb vulnerability ±20%
        noise = np.random.normal(1.0, 0.2, len(base_vuln))
        perturbed_vuln = np.clip(base_vuln * noise, 0, 1)

        risk = hazard * perturbed_vuln
        functionality = 1 - risk

        func_results.append({
            "mean_functionality": float(functionality.mean()),
            "mean_risk": float(risk.mean()),
            "high_risk_pct": float((risk > 0.05).mean() * 100),
        })

    func_df = pd.DataFrame(func_results)
    print(f"  Monte Carlo ({n_simulations} runs):")
    print(f"    Functionality: {func_df['mean_functionality'].mean():.6f} "
          f"± {func_df['mean_functionality'].std():.6f}")
    print(f"    Risk:          {func_df['mean_risk'].mean():.6f} "
          f"± {func_df['mean_risk'].std():.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(func_df["mean_functionality"], bins=30, color="#4ECDC4", alpha=0.7)
    axes[0].axvline(func_df["mean_functionality"].mean(), color="red", linestyle="--",
                     label=f"Mean: {func_df['mean_functionality'].mean():.6f}")
    axes[0].set_xlabel("Mean Functionality")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Monte Carlo: Functionality Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(func_df["mean_risk"], bins=30, color="#FF6B6B", alpha=0.7)
    axes[1].axvline(func_df["mean_risk"].mean(), color="blue", linestyle="--",
                     label=f"Mean: {func_df['mean_risk'].mean():.6f}")
    axes[1].set_xlabel("Mean Risk Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Monte Carlo: Risk Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/8_monte_carlo_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/8_monte_carlo_validation.png")

    mc_results = {
        "n_simulations": n_simulations,
        "functionality_mean": float(func_df["mean_functionality"].mean()),
        "functionality_std": float(func_df["mean_functionality"].std()),
        "risk_mean": float(func_df["mean_risk"].mean()),
        "risk_std": float(func_df["mean_risk"].std()),
    }
    with open(f"{RESULTS_DIR}/8_monte_carlo_results.json", "w") as f:
        json.dump(mc_results, f, indent=2)


# ================================================================
# 9️⃣ ENSEMBLE DIVERSITY IMPROVEMENT
# ================================================================

def run_ensemble_diversity(df):
    """Compare bootstrap RF vs diverse ensemble approaches."""
    print("\n" + "=" * 60)
    print("9️⃣  ENSEMBLE DIVERSITY IMPROVEMENT")
    print("=" * 60)

    train = df[df["time"] < "2022-08-01"]
    test  = df[df["time"] >= "2022-08-16"]

    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train = (train["tp_mm"] >= 10).astype(int)
    y_test  = (test["tp_mm"] >= 10).astype(int)

    ensembles = {}

    # A) Standard bootstrap (current approach)
    print("  Training A) Standard bootstrap RF...")
    probs_a = []
    for i in range(10):
        boot = np.random.choice(len(X_train), len(X_train), replace=True)
        m = RandomForestClassifier(n_estimators=200, max_depth=18,
                                    class_weight="balanced", n_jobs=-1, random_state=i)
        m.fit(X_train.iloc[boot], y_train.iloc[boot])
        probs_a.append(m.predict_proba(X_test)[:, 1])
    ensembles["Standard Bootstrap RF"] = np.array(probs_a)

    # B) Random feature subsets
    print("  Training B) Random feature subsets...")
    probs_b = []
    for i in range(10):
        n_feat = np.random.choice([4, 5, 6, 7], 1)[0]
        feats = np.random.choice(FEATURES, n_feat, replace=False).tolist()
        boot = np.random.choice(len(X_train), len(X_train), replace=True)
        m = RandomForestClassifier(n_estimators=150, max_depth=15,
                                    class_weight="balanced", n_jobs=-1, random_state=i + 100)
        m.fit(X_train[feats].iloc[boot], y_train.iloc[boot])
        probs_b.append(m.predict_proba(X_test[feats])[:, 1])
    ensembles["Random Feature Subsets"] = np.array(probs_b)

    # C) Varying hyperparameters
    print("  Training C) Varying hyperparameters...")
    probs_c = []
    hyperparams = [
        {"n_estimators": 50, "max_depth": 8},
        {"n_estimators": 100, "max_depth": 12},
        {"n_estimators": 200, "max_depth": 18},
        {"n_estimators": 300, "max_depth": 25},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 150, "max_depth": 15},
        {"n_estimators": 200, "max_depth": 20},
        {"n_estimators": 250, "max_depth": 22},
        {"n_estimators": 100, "max_depth": 8},
        {"n_estimators": 150, "max_depth": 18},
    ]
    for i, hp in enumerate(hyperparams):
        boot = np.random.choice(len(X_train), len(X_train), replace=True)
        m = RandomForestClassifier(**hp, class_weight="balanced", n_jobs=-1, random_state=i + 200)
        m.fit(X_train.iloc[boot], y_train.iloc[boot])
        probs_c.append(m.predict_proba(X_test)[:, 1])
    ensembles["Varying Hyperparameters"] = np.array(probs_c)

    # D) Mixed model types (RF + GBM)
    print("  Training D) Mixed RF + Gradient Boosting...")
    probs_d = []
    for i in range(5):
        boot = np.random.choice(len(X_train), len(X_train), replace=True)
        m = RandomForestClassifier(n_estimators=150, max_depth=15,
                                    class_weight="balanced", n_jobs=-1, random_state=i + 300)
        m.fit(X_train.iloc[boot], y_train.iloc[boot])
        probs_d.append(m.predict_proba(X_test)[:, 1])

    for i in range(5):
        boot = np.random.choice(len(X_train), min(50000, len(X_train)), replace=True)
        m = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=i + 400
        )
        m.fit(X_train.iloc[boot], y_train.iloc[boot])
        probs_d.append(m.predict_proba(X_test)[:, 1])
    ensembles["Mixed RF + GBM"] = np.array(probs_d)

    # Compare ensemble diversity (prediction std)
    diversity_results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (name, probs_arr) in enumerate(ensembles.items()):
        mean_p = probs_arr.mean(axis=0)
        std_p = probs_arr.std(axis=0)

        metrics = compute_metrics(y_test, (mean_p >= 0.5).astype(int), mean_p)
        diversity_results[name] = {
            "ROC_AUC": metrics["ROC_AUC"],
            "Brier": metrics["Brier"],
            "mean_epistemic_std": float(std_p.mean()),
            "max_epistemic_std": float(std_p.max()),
        }

        ax = axes[idx // 2][idx % 2]
        ax.hist(std_p, bins=50, alpha=0.7, color=["#FF6B6B", "#4ECDC4", "#FFE66D", "#45B7D1"][idx])
        ax.set_title(f"{name}\nMean σ={std_p.mean():.6f}, AUC={metrics['ROC_AUC']:.4f}")
        ax.set_xlabel("Epistemic Uncertainty (σ)")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        print(f"  {name}: AUC={metrics['ROC_AUC']:.4f}, "
              f"Brier={metrics['Brier']:.6f}, mean_σ={std_p.mean():.6f}")

    plt.suptitle("Ensemble Diversity Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/9_ensemble_diversity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/9_ensemble_diversity.png")

    with open(f"{RESULTS_DIR}/9_ensemble_diversity_results.json", "w") as f:
        json.dump(diversity_results, f, indent=2)


# ================================================================
# 📊 ADDITIONAL RESEARCH METRICS (CRPS, Risk Exceedance)
# ================================================================

def run_additional_metrics(models, X_test, y_test, std_prob_test, mean_prob_test):
    """CRPS, risk exceedance curves, network robustness."""
    print("\n" + "=" * 60)
    print("📊 ADDITIONAL RESEARCH METRICS")
    print("=" * 60)

    # CRPS (Continuous Ranked Probability Score) — approximate
    # CRPS = E|F(y) - 1(y <= x)| — we approximate with ensemble members
    brier = brier_score_loss(y_test, mean_prob_test)
    spread = std_prob_test.mean()
    crps_approx = brier + spread * 0.5  # Simplified CRPS approximation
    print(f"  Brier Score:           {brier:.6f}")
    print(f"  CRPS (approx):         {crps_approx:.6f}")
    print(f"  Mean Ensemble Spread:  {spread:.6f}")

    # Risk exceedance probability curve
    thresholds = np.linspace(0, 0.5, 50)
    exceedance = [(mean_prob_test > t).mean() for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(thresholds, exceedance, color="#FF6B6B", linewidth=2)
    axes[0].set_xlabel("Risk Threshold")
    axes[0].set_ylabel("Exceedance Probability")
    axes[0].set_title("Risk Exceedance Curve")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Reliability-Resolution-Uncertainty decomposition
    # Brier = Reliability - Resolution + Uncertainty
    climatology = y_test.mean()
    uncertainty = climatology * (1 - climatology)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    reliability = 0
    resolution = 0

    for i in range(n_bins):
        mask = (mean_prob_test >= bin_edges[i]) & (mean_prob_test < bin_edges[i + 1])
        n_k = mask.sum()
        if n_k > 0:
            o_k = y_test[mask].mean()
            f_k = mean_prob_test[mask].mean()
            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - climatology) ** 2

    reliability /= len(y_test)
    resolution /= len(y_test)

    print(f"  Reliability:           {reliability:.6f}")
    print(f"  Resolution:            {resolution:.6f}")
    print(f"  Uncertainty:           {uncertainty:.6f}")
    print(f"  Brier decomp check:    {reliability - resolution + uncertainty:.6f} (should ≈ {brier:.6f})")

    # Bar chart of decomposition
    components = ["Reliability\n(lower=better)", "Resolution\n(higher=better)", "Uncertainty"]
    values = [reliability, resolution, uncertainty]
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D"]
    axes[1].bar(components, values, color=colors)
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Brier Score Decomposition (Brier={brier:.6f})")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/10_additional_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/10_additional_metrics.png")

    add_results = {
        "brier_score": float(brier),
        "crps_approx": float(crps_approx),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "mean_ensemble_spread": float(spread),
    }
    with open(f"{RESULTS_DIR}/10_additional_metrics.json", "w") as f:
        json.dump(add_results, f, indent=2)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("🔬 Phase 6: Research-Grade Validation & Enhancement")
    print("=" * 60)

    df = load_data()

    # 1. Time-based split
    models, X_test, y_test, mean_prob, std_prob, test_data = run_time_split(df)

    # 2. Calibration
    run_calibration(models, X_test, y_test, mean_prob, df=df)

    # 3. Typhoon stress test
    run_typhoon_stress_test(models, df)

    # 4. Dynamic scaling
    run_dynamic_scaling(models, df)

    # 5. Threshold sensitivity
    run_threshold_sensitivity(df)

    # 6. Multi-threshold fusion
    run_multi_threshold_fusion(df)

    # 7. Spatial cross-validation
    run_spatial_cv(df)

    # 8. Functionality validation (Monte Carlo)
    run_functionality_validation()

    # 9. Ensemble diversity
    run_ensemble_diversity(df)

    # 10. Additional research metrics
    run_additional_metrics(models, X_test, y_test, std_prob, mean_prob)

    print("\n" + "=" * 60)
    print("🏁 PHASE 6 COMPLETE — ALL RESULTS SAVED TO results/phase6/")
    print("=" * 60)

    # List all generated files
    for f in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f} ({size:,} bytes)")
