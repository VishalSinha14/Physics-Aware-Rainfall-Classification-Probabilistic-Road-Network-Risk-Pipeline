"""
Evaluation Metrics for Rainfall Hazard & Road Risk Pipeline
------------------------------------------------------------
Meteorological: POD, FAR, CSI, ROC-AUC, PR-AUC
Infrastructure: Average functionality, critical road %
Uncertainty: Ensemble prediction variance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


# ================================================================
# METEOROLOGICAL METRICS
# ================================================================

def compute_meteorological_metrics(y_true, y_pred, y_prob=None):
    """
    Compute POD, FAR, CSI from binary predictions.
    Optionally compute ROC-AUC and PR-AUC if probabilities are provided.

    Returns a dict of metric_name: value.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Handle case where confusion matrix might be 1x1 (all same class)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        return {"error": "Cannot compute metrics — only one class present"}

    # Probability of Detection (Hit Rate / Recall)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # False Alarm Ratio
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Critical Success Index (Threat Score)
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

    # Bias Score
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "Bias": bias,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }

    if y_prob is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        metrics["PR_AUC"] = average_precision_score(y_true, y_prob)

    return metrics


def compute_roc_curve(y_true, y_prob):
    """Compute ROC curve data for plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def compute_pr_curve(y_true, y_prob):
    """Compute Precision-Recall curve data for plotting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


# ================================================================
# INFRASTRUCTURE METRICS
# ================================================================

def compute_infrastructure_metrics(roads_gdf):
    """
    Compute network-level infrastructure metrics from risk-scored roads.

    Parameters:
        roads_gdf: GeoDataFrame with 'functionality', 'risk_score', 'highway' columns.

    Returns dict with infrastructure metrics.
    """
    metrics = {}

    # Average network functionality
    metrics["avg_functionality"] = float(roads_gdf["functionality"].mean())

    # Average risk score
    metrics["avg_risk"] = float(roads_gdf["risk_score"].mean())

    # Percentage of roads by risk category
    total = len(roads_gdf)
    high_risk = (roads_gdf["risk_score"] > 0.05).sum()
    med_risk = ((roads_gdf["risk_score"] > 0.01) & (roads_gdf["risk_score"] <= 0.05)).sum()
    low_risk = (roads_gdf["risk_score"] <= 0.01).sum()

    metrics["pct_high_risk"] = float(100 * high_risk / total) if total > 0 else 0.0
    metrics["pct_medium_risk"] = float(100 * med_risk / total) if total > 0 else 0.0
    metrics["pct_low_risk"] = float(100 * low_risk / total) if total > 0 else 0.0

    # Critical roads (motorway, trunk, primary) affected
    critical_types = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}
    if "highway" in roads_gdf.columns:
        critical_mask = roads_gdf["highway"].apply(
            lambda x: x in critical_types if isinstance(x, str)
            else (x[0] in critical_types if isinstance(x, list) else False)
        )
        critical_roads = roads_gdf[critical_mask]
        if len(critical_roads) > 0:
            metrics["critical_roads_count"] = int(len(critical_roads))
            metrics["critical_avg_functionality"] = float(critical_roads["functionality"].mean())
            metrics["critical_avg_risk"] = float(critical_roads["risk_score"].mean())
        else:
            metrics["critical_roads_count"] = 0

    return metrics


# ================================================================
# UNCERTAINTY METRICS
# ================================================================

def compute_uncertainty_metrics(roads_gdf):
    """Compute uncertainty-related metrics from ensemble predictions."""
    if "risk_uncertainty" not in roads_gdf.columns:
        return {}

    metrics = {
        "mean_risk_uncertainty": float(roads_gdf["risk_uncertainty"].mean()),
        "max_risk_uncertainty": float(roads_gdf["risk_uncertainty"].max()),
        "std_risk_uncertainty": float(roads_gdf["risk_uncertainty"].std()),
    }

    if "hazard_uncertainty" in roads_gdf.columns:
        metrics["mean_hazard_uncertainty"] = float(roads_gdf["hazard_uncertainty"].mean())
        metrics["max_hazard_uncertainty"] = float(roads_gdf["hazard_uncertainty"].max())

    return metrics


# ================================================================
# FULL REPORT
# ================================================================

def generate_full_report(roads_gdf, y_true=None, y_pred=None, y_prob=None):
    """
    Generate a comprehensive evaluation report.

    Parameters:
        roads_gdf: Risk-scored road GeoDataFrame
        y_true, y_pred, y_prob: Optional meteorological predictions for model eval

    Returns a dict containing all metric categories.
    """
    report = {}

    if y_true is not None and y_pred is not None:
        report["meteorological"] = compute_meteorological_metrics(y_true, y_pred, y_prob)

    report["infrastructure"] = compute_infrastructure_metrics(roads_gdf)
    report["uncertainty"] = compute_uncertainty_metrics(roads_gdf)

    return report


def print_report(report):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    if "meteorological" in report:
        print("\n--- Meteorological Metrics ---")
        for k, v in report["meteorological"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if "infrastructure" in report:
        print("\n--- Infrastructure Metrics ---")
        for k, v in report["infrastructure"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if "uncertainty" in report:
        print("\n--- Uncertainty Metrics ---")
        for k, v in report["uncertainty"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    print("=" * 60)
