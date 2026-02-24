"""
Class Imbalance Analysis
--------------------------
Documents the class imbalance in the dataset for each rainfall threshold,
explains mathematically why accuracy is misleading, and shows the
mitigation strategies used in this project.

Outputs → results/scientific/imbalance_report.json
          results/scientific/imbalance_analysis.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/scientific", exist_ok=True)


def main():
    print("=" * 60)
    print("Class Imbalance Analysis")
    print("=" * 60)

    print("\nLoading data...")
    df = pd.read_csv("data/processed/era5_2022_monsoon_temporal.csv")
    total = len(df)

    thresholds = [5, 10, 20]
    records = []
    for t in thresholds:
        pos = (df["tp_mm"] >= t).sum()
        neg = total - pos
        pos_pct = 100 * pos / total
        neg_pct = 100 * neg / total
        # Naive majority classifier accuracy
        naive_acc = max(pos_pct, neg_pct)
        # Skilless CSI (all predicted negative)
        csi_naive = 0.0
        records.append({
            "threshold_mm": t,
            "positive_count": int(pos),
            "negative_count": int(neg),
            "positive_pct": round(pos_pct, 4),
            "negative_pct": round(neg_pct, 4),
            "imbalance_ratio": round(neg / pos, 1) if pos > 0 else None,
            "naive_accuracy_pct": round(naive_acc, 4),
            "naive_csi": csi_naive,
            "naive_pod": 0.0,
        })
        print(f"\n  Threshold = {t} mm/hr:")
        print(f"    Positive (heavy rain): {pos:>10,}  ({pos_pct:.4f}%)")
        print(f"    Negative (no rain)  : {neg:>10,}  ({neg_pct:.4f}%)")
        print(f"    Imbalance ratio     : {neg/pos:.0f}:1" if pos > 0 else "")
        print(f"    Naive accuracy      : {naive_acc:.4f}%")

    # ── Why accuracy fails — mathematical breakdown ──────────────
    print("\n--- Why accuracy is misleading ---")
    print(f"  Total samples: {total:,}")
    pos10 = (df['tp_mm'] >= 10).sum()
    neg10 = total - pos10
    print(f"  Heavy rain (>=10mm): {pos10:,}  ({100*pos10/total:.4f}%)")
    print(f"  A model that always predicts NO RAIN gets:")
    print(f"    Accuracy = {100*neg10/total:.2f}%  (looks great, but is useless)")
    print(f"    POD  = 0.00  (misses all events)")
    print(f"    CSI  = 0.00  (no skill)")

    # ── Visualization ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Class Imbalance: Why Accuracy Is Misleading", fontsize=14, fontweight="bold")

    # Plot 1: Imbalance pie charts
    for i, rec in enumerate(records):
        ax = axes[i]
        sizes = [rec["positive_pct"], rec["negative_pct"]]
        labels = [f"Heavy Rain\n({rec['positive_pct']:.3f}%)",
                  f"No Heavy Rain\n({rec['negative_pct']:.2f}%)"]
        colors = ["#E74C3C", "#3498DB"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title(f"Threshold ≥ {rec['threshold_mm']} mm/hr\n"
                     f"Imbalance: {rec['imbalance_ratio']}:1\n"
                     f"Naive Accuracy: {rec['naive_accuracy_pct']:.2f}%",
                     fontsize=10)

    plt.tight_layout()
    plt.savefig("results/scientific/imbalance_analysis.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── Save JSON ────────────────────────────────────────────────
    report = {
        "total_samples": total,
        "thresholds": records,
        "why_accuracy_fails": {
            "explanation": (
                "Accuracy = (TP+TN)/(TP+TN+FP+FN). With severe class imbalance, "
                "TN dominates. A trivial model that always predicts 'No Heavy Rain' "
                "achieves >99.8% accuracy but detects zero events. "
                "We use POD, FAR, CSI, and Brier Score instead."
            ),
            "naive_10mm_accuracy": round(100 * neg10 / total, 4),
            "naive_10mm_pod": 0.0,
            "naive_10mm_csi": 0.0,
        },
        "mitigation_strategies": [
            "class_weight='balanced' in all classifiers",
            "Bootstrap ensemble with stratified sampling",
            "Primary metrics: POD, FAR, CSI, PR-AUC, Brier Score",
            "Accuracy reported in appendix only",
        ]
    }
    with open("results/scientific/imbalance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Saved → results/scientific/imbalance_analysis.png")
    print("✅ Saved → results/scientific/imbalance_report.json")


if __name__ == "__main__":
    main()
