"""
Lead-Time Degradation Curve
-----------------------------
Loads all three lead-time model metrics and plots the performance
degradation as lead time increases (0h → 30m → 60m).

This converts the project from a detection system to an
early-warning framework.

Outputs → results/leadtime/degradation_curve.png
          results/leadtime/degradation_summary.json
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/leadtime", exist_ok=True)


def main():
    print("=" * 60)
    print("Lead-Time Degradation Curve")
    print("=" * 60)

    metrics_path = "results/leadtime/leadtime_metrics.json"
    if not os.path.exists(metrics_path):
        print("❌ Run train_leadtime_models.py first")
        return

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # Lead-time order and labels
    order = ["0h", "30m", "60m"]
    labels = ["0 hr\n(nowcast)", "+30 min\n(short-range)", "+60 min\n(medium-range)"]
    x = np.arange(len(order))

    available = [l for l in order if l in all_metrics]
    avail_labels = [labels[order.index(l)] for l in available]
    x_avail = np.arange(len(available))

    metrics_to_plot = [
        ("POD",     "POD (Prob. of Detection)",    "#E74C3C",  True),
        ("CSI",     "CSI (Critical Success Index)", "#E67E22",  True),
        ("ROC_AUC", "ROC-AUC",                     "#3498DB",  True),
        ("PR_AUC",  "PR-AUC",                       "#9B59B6",  True),
        ("Brier",   "Brier Score (↓ better)",       "#27AE60",  False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Performance Degradation with Lead Time\n"
                 "(Ensemble RF — 10 bootstrap models)", fontsize=14, fontweight="bold")

    axes_flat = axes.flatten()

    for idx, (mkey, mname, color, higher_better) in enumerate(metrics_to_plot):
        ax = axes_flat[idx]
        values = [all_metrics[lt][mkey] for lt in available if mkey in all_metrics[lt]]
        ax.plot(x_avail[:len(values)], values, "o-", color=color,
                lw=2.5, ms=9, markeredgecolor="white", markeredgewidth=1.5)
        ax.fill_between(x_avail[:len(values)], values, alpha=0.1, color=color)
        for xi, v in zip(x_avail[:len(values)], values):
            ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x_avail[:len(values)])
        ax.set_xticklabels(avail_labels[:len(values)], fontsize=9)
        ax.set_ylabel(mname)
        ax.set_title(mname)
        ax.grid(True, alpha=0.3)
        # Add direction annotation
        arrow = "↑ better" if higher_better else "↓ better"
        ax.annotate(arrow, xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", fontsize=8, color="gray")

    # Panel 6: Summary table
    ax_t = axes_flat[5]
    ax_t.axis("off")
    table_rows = []
    for lt in available:
        m = all_metrics[lt]
        table_rows.append([
            lt,
            f"{m.get('POD', 0):.4f}",
            f"{m.get('FAR', 0):.4f}",
            f"{m.get('CSI', 0):.4f}",
            f"{m.get('ROC_AUC', 0):.4f}",
            f"{m.get('Brier', 0):.6f}",
        ])
    tbl = ax_t.table(
        cellText=table_rows,
        colLabels=["Lead", "POD↑", "FAR↓", "CSI↑", "AUC↑", "Brier↓"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for j in range(6):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    row_colors = ["#EBF5FB", "#E8F8F5", "#FDFEFE"]
    for i in range(1, len(table_rows) + 1):
        for j in range(6):
            tbl[i, j].set_facecolor(row_colors[(i - 1) % len(row_colors)])
    ax_t.set_title("Summary Table", fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/leadtime/degradation_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── Compute degradation rates ────────────────────────────────
    summary = {"lead_times": available, "metrics": {}}
    for mkey, mname, _, higher_better in metrics_to_plot:
        values = [all_metrics[lt].get(mkey) for lt in available
                  if all_metrics[lt].get(mkey) is not None]
        if len(values) >= 2:
            drop = values[0] - values[-1]
            drop_pct = 100 * drop / abs(values[0]) if values[0] != 0 else 0
            summary["metrics"][mkey] = {
                "values_by_lead": dict(zip(available, values)),
                "total_change": float(drop),
                "total_change_pct": float(drop_pct),
                "direction": "decrease" if drop > 0 else "increase",
                "higher_better": higher_better
            }

    with open("results/leadtime/degradation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Saved → results/leadtime/degradation_curve.png")
    print("✅ Saved → results/leadtime/degradation_summary.json")
    print("\n--- Degradation Summary ---")
    for mkey, info in summary["metrics"].items():
        better_word = "drop" if info["higher_better"] else "improvement"
        print(f"  {mkey}: {info['total_change_pct']:.2f}% {better_word} from 0h → 60m")


if __name__ == "__main__":
    main()
